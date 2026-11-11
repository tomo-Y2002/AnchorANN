#!/usr/bin/env python3
"""Full-base float64 PCA and exhaustive IP/L2 truth for MS MARCO."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import numpy as np

N = 8_841_792
DIM = 768
COMPONENTS = 192
QUERIES = 1000


def vecs(path, dimension):
    path = Path(path)
    if path.stat().st_size % (4 * (dimension + 1)):
        raise ValueError('truncated vecs')
    rows = np.memmap(path, dtype='<f4', mode='r').reshape(-1, dimension + 1)
    if not np.all(rows[:, 0].view('<i4') == dimension):
        raise ValueError('vecs dimensions')
    return rows[:, 1:]


def write_vecs(path, values, integer=False):
    dtype = '<i4' if integer else '<f4'
    with Path(path).open('wb') as f:
        for start in range(0, len(values), 8192):
            x = values[start:start + 8192]
            rows = np.empty((len(x), x.shape[1] + 1), dtype=dtype)
            rows[:, 0].view('<i4')[:] = x.shape[1]
            rows[:, 1:] = x
            f.write(rows.tobytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def principal_components(base, count, block=8192):
    # Merge centered block covariances, avoiding cancellation in E[xx']-mu mu'.
    mean = np.zeros(base.shape[1], dtype=np.float64)
    scatter = np.zeros((base.shape[1], base.shape[1]), dtype=np.float64)
    n = 0
    for start in range(0, len(base), block):
        x = np.asarray(base[start:start + block], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite base')
        batch_mean = x.mean(axis=0)
        delta = batch_mean - mean
        centered = x - batch_mean
        scatter += centered.T @ centered + np.outer(delta, delta) * n * len(x) / (n + len(x))
        mean += delta * len(x) / (n + len(x))
        n += len(x)
    covariance = scatter / (n - 1)
    eigenvalues, matrix = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    matrix = matrix[:, order[:count]]
    # Fix the otherwise arbitrary eigenvector signs for reproducibility.
    signs = np.sign(matrix[np.argmax(np.abs(matrix), axis=0), np.arange(count)])
    matrix *= signs
    return mean, matrix, float(eigenvalues[order[:count]].sum() / eigenvalues.sum())


def top(scores, ids, k, largest):
    key = -scores if largest else scores
    selected = np.argpartition(key, k - 1)[:k]
    boundary = key[selected].max()
    # Include all ties at the boundary before resolving by original row ID.
    selected = np.flatnonzero(key <= boundary)
    selected = selected[np.lexsort((ids[selected], key[selected]))[:k]]
    return scores[selected], ids[selected]


def exact_truth(base, query, k=100, block=65536):
    """Independent exhaustive IP and actual saved-float32 L2 top k."""
    ip_ids, l2_ids = [], []
    # Batched matrix multiplication keeps full-N truth practical without N*q RAM.
    for begin in range(0, len(query), 25):
        q = np.asarray(query[begin:begin + 25], dtype=np.float64)
        ips = [np.empty(0) for _ in q]
        dists = [np.empty(0) for _ in q]
        ii = [np.empty(0, dtype=np.int64) for _ in q]
        di = [np.empty(0, dtype=np.int64) for _ in q]
        for start in range(0, len(base), block):
            x = np.asarray(base[start:start + block], dtype=np.float64)
            dot = q[:, :-1] @ x[:, :-1].T
            distance = np.square(x).sum(axis=1)[None, :] + np.square(q).sum(axis=1)[:, None] - 2 * dot
            ids = np.arange(start, start + len(x))
            for j in range(len(q)):
                values, indices = top(dot[j], ids, min(k, len(x)), True)
                ips[j], ii[j] = top(np.concatenate((ips[j], values)), np.concatenate((ii[j], indices)), min(k, len(ii[j]) + len(indices)), True)
                values, indices = top(distance[j], ids, min(k, len(x)), False)
                dists[j], di[j] = top(np.concatenate((dists[j], values)), np.concatenate((di[j], indices)), min(k, len(di[j]) + len(indices)), False)
        for j in range(len(q)):
            if set(ii[j]) != set(di[j]):
                # Allow only permutations of genuinely equal saved L2 distances.
                candidates = np.unique(np.concatenate((ii[j], di[j])))
                x = np.asarray(base[candidates], dtype=np.float64)
                distances = np.square(x - q[j]).sum(axis=1)
                lookup = dict(zip(candidates.tolist(), distances.tolist()))
                if sorted(lookup[int(id)] for id in ii[j]) != sorted(lookup[int(id)] for id in di[j]):
                    raise ValueError(f'IP/L2 ranking differs for query {begin+j}')
            ip_ids.append(ii[j]); l2_ids.append(di[j])
        print('exact queries complete', begin + len(q), flush=True)
    return np.array(ip_ids, dtype='<i4'), np.array(l2_ids, dtype='<i4')


def qrels(source, query_ids, passage_ids, output):
    ids = {line.split('\t')[1]: int(line.split('\t')[0]) for line in Path(passage_ids).read_text().splitlines()[1:]}
    queries = [line.split('\t')[1] for line in Path(query_ids).read_text().splitlines()[1:]][:QUERIES]
    wanted = set(queries)
    relevant = {qid: [] for qid in queries}
    lost = set()
    for line in Path(source).read_text().splitlines():
        qid, _, pid, relevance = line.split()
        if qid not in wanted or int(relevance) <= 0:
            continue
        if pid in ids:
            relevant[qid].append(ids[pid])
        else:
            lost.add(qid)
    Path(output).write_text(json.dumps([relevant[qid] for qid in queries]) + '\n')
    return len(lost)


def prepare(source, output):
    started = time.monotonic()
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('completed inputs exist; keep immutable')
    base = vecs(source / 'msmarco_base.fvecs', DIM + 1)
    query = vecs(source / 'msmarco_query.fvecs', DIM + 1)[:QUERIES]
    if len(base) != N or len(query) != QUERIES:
        raise ValueError('requires full retained base and first1000 queries')
    original = json.loads((source / 'base-manifest.json').read_text())
    totals_min, totals_max = np.inf, -np.inf
    for start in range(0, N, 8192):
        x = np.asarray(base[start:start + 8192], dtype=np.float64)
        norm = np.square(x).sum(axis=1)
        totals_min = min(totals_min, norm.min()); totals_max = max(totals_max, norm.max())
    # The source transform's extra coordinate is float32; quantify its rounding.
    tolerance = 8 * np.finfo(np.float32).eps * max(1, totals_max)
    if totals_max - totals_min > tolerance:
        raise ValueError('source augmented norms are not constant within float32 rounding')
    print('source norms verified', totals_min, totals_max, flush=True)
    mean, matrix, variance = principal_components(base[:, :DIM], COMPONENTS)
    np.save(output / 'pca_mean.npy', mean)
    np.save(output / 'pca_matrix.npy', matrix)
    projected = np.lib.format.open_memmap(output / 'projected.npy', mode='w+', dtype='<f4', shape=(N, COMPONENTS))
    m2 = 0.
    for start in range(0, N, 8192):
        x = (np.asarray(base[start:start + 8192, :DIM], dtype=np.float64) @ matrix).astype('<f4')
        projected[start:start + len(x)] = x
        m2 = max(m2, np.square(x.astype(np.float64)).sum(axis=1).max())
    projected.flush()
    with (output / 'base.fvecs').open('wb') as f:
        for start in range(0, N, 8192):
            x = projected[start:start + 8192]
            rows = np.empty((len(x), COMPONENTS + 2), dtype='<f4')
            rows[:, 0].view('<i4')[:] = COMPONENTS + 1
            rows[:, 1:-1] = x
            rows[:, -1] = np.sqrt(np.maximum(0, m2 - np.square(x.astype(np.float64)).sum(axis=1)))
            f.write(rows.tobytes())
    q = np.column_stack(((np.asarray(query[:, :DIM], dtype=np.float64) @ matrix).astype('<f4'), np.zeros(QUERIES, dtype='<f4')))
    write_vecs(output / 'query.fvecs', q)
    print('projection complete', variance, flush=True)
    truth, _ = exact_truth(vecs(output / 'base.fvecs', COMPONENTS + 1), q)
    write_vecs(output / 'gt.ivecs', truth, integer=True)
    qrels_source = source / 'qrels.dev.small.tsv'
    if not qrels_source.exists():
        qrels_source = output / 'qrels.dev.small.tsv'
        urllib.request.urlretrieve('https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv', qrels_source)
    lost = qrels(qrels_source, source / 'query_ids.tsv', source / 'passage_ids.tsv', output / 'qrels.json')
    names = ['base.fvecs', 'query.fvecs', 'gt.ivecs', 'qrels.json', 'pca_matrix.npy', 'pca_mean.npy']
    manifest = dict(n=N, d=COMPONENTS + 1, q=QUERIES, k=100, pca_fit='full_base_centered_float64_covariance', projection_centered=False,
                    explained_variance_ratio=variance, m2=float(m2), source_squared_norm_min=float(totals_min), source_squared_norm_max=float(totals_max),
                    source_norm_tolerance=float(tolerance), source_manifest=original, lost_relevant_queries=lost, ip_l2_top100_verified=QUERIES,
                    seconds=time.monotonic() - started, files={name:dict(sha256=sha(output/name), bytes=(output/name).stat().st_size) for name in names})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
