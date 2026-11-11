#!/usr/bin/env python3
"""Create TAS-B MS MARCO fvecs and exact IP ground truth.

All numerical preparation/search uses NumPy. Torch and transformers are
imported only by the query encoding stage, when requested.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import struct
import time

import numpy as np

try:
    from .download_msmarco import (BASE_COUNT, DIMENSION, MODEL, MODEL_REVISION, SOURCE_COUNT,
                         file_record, save_json)
except ImportError:
    from download_msmarco import (BASE_COUNT, DIMENSION, MODEL, MODEL_REVISION, SOURCE_COUNT,
                        file_record, save_json)

TIE_RULE = 'float64 score, then ascending zero-based output passage number; no tie tolerance'
ROUNDING_RELATIVE_TOLERANCE = 64 * np.finfo(np.float64).eps


def read_npids(path: Path, expected_count: int) -> np.ndarray:
    """Read the distribution's NPID v1 integer sequence, with no pickle/code load."""
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:4] != b'NPID':
        raise ValueError('unsupported NPID header; direct reader needs review')
    offset, reserved, length = struct.unpack_from('<QQI', raw, 4)
    if reserved != 0 or offset != 24 + length or raw[24:offset] != b'{"version": 1}':
        raise ValueError('unsupported NPID version/header')
    nodes = []
    for expected_kind in (0, 1):
        if offset + 24 > len(raw):
            raise ValueError('truncated NPID node')
        kind, next_offset, count, length = struct.unpack_from('<IqQI', raw, offset)
        end = offset + 24 + length
        if end > len(raw) or kind != expected_kind or count != expected_count:
            raise ValueError('NPID node kind/count/length mismatch')
        node = json.loads(raw[offset + 24:end])
        nodes.append(node)
        if kind == 0 and next_offset != end or kind == 1 and (next_offset != -1 or end != len(raw)):
            raise ValueError('unsupported NPID tree layout')
        offset = end
    if nodes != [dict(format='intsequence', prefix='', start=0),
                 dict(format='intsequence', prefix='', start=0, count=expected_count)]:
        raise ValueError('unsupported NPID sequence; do not guess passage IDs')
    return np.arange(expected_count, dtype=np.int64)


def retained_indices(ids: np.ndarray, remove: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove the largest numeric passage IDs, preserving the original row order."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('passage IDs must be unique nonnegative integers')
    if not 0 <= remove < len(ids):
        raise ValueError('invalid exclusion count')
    excluded = np.sort(ids)[-remove:][::-1] if remove else np.empty(0, dtype=ids.dtype)
    return np.flatnonzero(~np.isin(ids, excluded)), excluded


def squared_norms(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('vectors must be a finite matrix')
    result = np.einsum('ij,ij->i', values, values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite squared norm')
    return result


def augment_base(vectors: np.ndarray, maximum_norm: float) -> tuple[np.ndarray, dict]:
    if not math.isfinite(maximum_norm) or maximum_norm < 0:
        raise ValueError('maximum norm must be finite and nonnegative')
    maximum_squared = maximum_norm * maximum_norm
    if not math.isfinite(maximum_squared):
        raise ValueError('maximum squared norm overflow')
    norms = squared_norms(vectors)
    tolerance = ROUNDING_RELATIVE_TOLERANCE * max(1., maximum_squared)
    radicand = maximum_squared - norms
    if np.any(radicand < -tolerance):
        raise ValueError('negative augmentation radicand exceeds rounding tolerance')
    clamped = int(np.count_nonzero(radicand < 0))
    minimum = float(radicand.min()) if len(radicand) else 0.
    output = np.empty((len(vectors), vectors.shape[1] + 1), dtype=np.float32)
    output[:, :-1] = vectors
    output[:, -1] = np.sqrt(np.maximum(radicand, 0.))
    if not np.isfinite(output).all():
        raise ValueError('nonfinite float32 augmentation')
    return output, dict(rounding_relative_tolerance=ROUNDING_RELATIVE_TOLERANCE,
                        rounding_absolute_tolerance=tolerance, clamped_count=clamped,
                        minimum_radicand=minimum)


def augment_queries(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('queries must be a finite matrix')
    return np.pad(values, ((0, 0), (0, 1)))


def write_vecs(path: Path, values: np.ndarray, integer=False) -> None:
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('vecs must be a finite matrix')
    if integer and (values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values > np.iinfo(np.int32).max)):
        raise ValueError('ivecs values must be nonnegative int32 IDs')
    dtype = '<i4' if integer else '<f4'
    with path.open('xb') as stream:
        rows = np.empty((len(values), values.shape[1] + 1), dtype=dtype)
        rows.view('<i4')[:, 0] = values.shape[1]
        rows[:, 1:] = values
        rows.tofile(stream)


def read_vecs(path: Path, dimension: int, count: int, integer=False) -> np.ndarray:
    if path.stat().st_size != count * (dimension + 1) * 4:
        raise ValueError('vecs size/shape mismatch')
    rows = np.memmap(path, mode='r', dtype='<i4' if integer else '<f4', shape=(count, dimension + 1))
    for start in range(0, count, 65536):
        if np.any(rows.view('<i4')[start:start + 65536, 0] != dimension):
            raise ValueError('invalid vecs dimension header')
    return rows[:, 1:]


def topk(values: np.ndarray, ids: np.ndarray, k: int, largest=True) -> tuple[np.ndarray, np.ndarray]:
    """Exact partition with deterministic IDs even for ties across the kth boundary."""
    values = np.asarray(values, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all() or not 0 < k <= len(ids):
        raise ValueError('invalid top-k inputs')
    order_values = -values if largest else values
    candidates = np.argpartition(order_values, k - 1)[:k]
    boundary = order_values[candidates].max()
    strict = np.flatnonzero(order_values < boundary)
    tied = np.flatnonzero(order_values == boundary)
    needed = k - len(strict)
    if len(tied) > needed:
        tied = tied[np.argpartition(ids[tied], needed - 1)[:needed]]
    candidates = np.concatenate((strict, tied))
    order = np.lexsort((ids[candidates], order_values[candidates]))
    selected = candidates[order]
    return ids[selected], values[selected]


def exact_search(base: np.ndarray, queries: np.ndarray, k: int, block_rows: int = 65536,
                 metric='ip', progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive float64 search; score matrix memory bounded by base block size.

    L2 uses norms of the actual saved float32 vectors, including coordinate 769.
    No constant-M substitution is used in this independent verification.
    """
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1] or not 0 < k <= len(base) or block_rows <= 0:
        raise ValueError('invalid search shape/k/block size')
    if metric not in ('ip', 'l2'):
        raise ValueError('unsupported metric')
    query_values = np.asarray(queries, dtype=np.float64)
    if not np.isfinite(query_values).all():
        raise ValueError('nonfinite query')
    query_norms = squared_norms(query_values) if metric == 'l2' else None
    best_ids = [np.empty(0, dtype=np.int64) for _ in queries]
    best_values = [np.empty(0, dtype=np.float64) for _ in queries]
    for start in range(0, len(base), block_rows):
        stop = min(start + block_rows, len(base))
        block = np.asarray(base[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError('nonfinite base vector')
        scores = query_values @ block.T
        if metric == 'l2':
            scores *= -2.
            scores += squared_norms(block)[None, :]
            scores += query_norms[:, None]
        ids = np.arange(start, stop, dtype=np.int64)
        for qi, row in enumerate(scores):
            local_ids, local_values = topk(row, ids, min(k, len(ids)), largest=metric == 'ip')
            merged_ids = np.concatenate((best_ids[qi], local_ids))
            merged_values = np.concatenate((best_values[qi], local_values))
            best_ids[qi], best_values[qi] = topk(merged_values, merged_ids, min(k, len(merged_ids)), largest=metric == 'ip')
        if progress:
            progress(stop, len(base))
    return np.stack(best_ids), np.stack(best_values)


def verify_rankings(base: np.ndarray, queries: np.ndarray, query_ids: list[str], k=10,
                    block_rows=65536) -> dict:
    ip_ids, ip_scores = exact_search(base[:, :-1], queries[:, :-1], k, block_rows)
    l2_ids, l2_distances = exact_search(base, queries, k, block_rows, metric='l2')
    mismatches = []
    for qi in range(len(queries)):
        if np.array_equal(ip_ids[qi], l2_ids[qi]):
            continue
        entries = []
        for pid in sorted(set(ip_ids[qi]) | set(l2_ids[qi])):
            vector = np.asarray(base[pid], dtype=np.float64)
            query = np.asarray(queries[qi], dtype=np.float64)
            entries.append(dict(passage_number=int(pid), inner_product=float(vector[:-1] @ query[:-1]),
                                squared_l2=float(np.sum((vector - query) ** 2))))
        mismatches.append(dict(query_row=qi, query_id=query_ids[qi],
                               ip_topk=ip_ids[qi].tolist(), l2_topk=l2_ids[qi].tolist(), candidates=entries))
    return dict(query_count=len(queries), k=k, passed=not mismatches, tie_rule=TIE_RULE,
                ip_topk=ip_ids.tolist(), ip_scores=ip_scores.tolist(),
                l2_topk=l2_ids.tolist(), squared_l2=l2_distances.tolist(), mismatches=mismatches)


def read_queries(path: Path, expected_count: int = 6980) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with path.open(encoding='utf-8') as stream:
        for row in stream:
            fields = row.rstrip('\r\n').split('\t', 1)
            if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
                raise ValueError('malformed dev small query row')
            ids.append(fields[0])
            texts.append(fields[1])
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError('dev small query count/uniqueness mismatch')
    return ids, texts


def encode_queries(texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, dict]:
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=MODEL_REVISION)
    model.eval()
    vectors = np.empty((len(texts), DIMENSION), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                              max_length=512, return_tensors='pt')
            # The model card specifies the CLS vector, shared query/passage layers,
            # and dot-product scoring. There is no normalization or mean pooling.
            output = model(**batch).last_hidden_state[:, 0, :]
            vectors[start:start + len(output)] = output.cpu().numpy()
            print(f'encoded_queries={min(start + batch_size, len(texts))}/{len(texts)}', flush=True)
    return vectors, dict(model=MODEL, revision=MODEL_REVISION, pooling='CLS', normalized=False,
                         max_length=512, batch_size=batch_size, device='cpu',
                         torch_version=torch.__version__, transformers_version=transformers.__version__)


def prepare_base(directory: Path, block_rows: int) -> dict:
    ids = read_npids(directory / 'docnos.npids', SOURCE_COUNT)
    rows, excluded = retained_indices(ids, SOURCE_COUNT - BASE_COUNT)
    source = directory / 'vecs.f4'
    if source.stat().st_size != SOURCE_COUNT * DIMENSION * 4:
        raise ValueError('source vector length mismatch')
    vectors = np.memmap(source, dtype='<f4', mode='r', shape=(SOURCE_COUNT, DIMENSION))
    maximum_squared = 0.
    for start in range(0, len(rows), block_rows):
        maximum_squared = max(maximum_squared, float(squared_norms(vectors[rows[start:start + block_rows]]).max()))
    maximum = math.sqrt(maximum_squared)
    target = directory / 'msmarco_base.fvecs'
    if target.exists() or target.with_suffix('.fvecs.partial').exists():
        raise ValueError('base output already exists; no implicit overwrite')
    partial = target.with_suffix('.fvecs.partial')
    output = np.memmap(partial, mode='w+', dtype='<f4', shape=(BASE_COUNT, DIMENSION + 2))
    clamped = 0
    minimum = 0.
    for start in range(0, len(rows), block_rows):
        augmented, numerical = augment_base(vectors[rows[start:start + block_rows]], maximum)
        output.view('<i4')[start:start + len(augmented), 0] = DIMENSION + 1
        output[start:start + len(augmented), 1:] = augmented
        clamped += numerical['clamped_count']
        minimum = min(minimum, numerical['minimum_radicand'])
    output.flush()
    del output
    partial.replace(target)
    mapping = directory / 'passage_ids.tsv'
    with mapping.open('x') as stream:
        stream.write('row\tpassage_id\n')
        for row, original_row in enumerate(rows):
            stream.write(f'{row}\t{ids[original_row]}\n')
    excluded_path = directory / 'excluded_passage_ids.txt'
    with excluded_path.open('x') as stream:
        stream.writelines(f'{pid}\n' for pid in excluded)
    record = dict(count=BASE_COUNT, dimension=DIMENSION + 1, source_count=SOURCE_COUNT,
                  maximum_norm=maximum, maximum_squared_norm=maximum_squared,
                  augmentation_squared_M=maximum * maximum, clamped_count=clamped,
                  minimum_radicand=minimum,
                  rounding_relative_tolerance=ROUNDING_RELATIVE_TOLERANCE,
                  rounding_absolute_tolerance=numerical['rounding_absolute_tolerance'],
                  excluded_passage_ids=excluded.tolist(), output=file_record(target),
                  passage_ids=file_record(mapping), exclusions=file_record(excluded_path))
    save_json(directory / 'base-manifest.json', record)
    return record


def prepare_queries(directory: Path, batch_size: int) -> dict:
    ids, texts = read_queries(directory / 'queries.dev.small.tsv')
    vectors, model = encode_queries(texts, batch_size)
    target = directory / 'msmarco_query.fvecs'
    write_vecs(target, augment_queries(vectors))
    mapping = directory / 'query_ids.tsv'
    with mapping.open('x') as stream:
        stream.write('row\tquery_id\n')
        stream.writelines(f'{row}\t{qid}\n' for row, qid in enumerate(ids))
    first = directory / 'first100_query_ids.txt'
    with first.open('x') as stream:
        stream.writelines(f'{qid}\n' for qid in ids[:100])
    record = dict(count=len(ids), dimension=DIMENSION + 1, model=model,
                  row_order='queries.dev.small.tsv distribution order', output=file_record(target),
                  query_ids=file_record(mapping), first100_query_ids=file_record(first))
    save_json(directory / 'query-manifest.json', record)
    return record


def prepare_ground_truth(directory: Path, block_rows: int) -> dict:
    ids, _ = read_queries(directory / 'queries.dev.small.tsv')
    base = read_vecs(directory / 'msmarco_base.fvecs', DIMENSION + 1, BASE_COUNT)
    queries = read_vecs(directory / 'msmarco_query.fvecs', DIMENSION + 1, len(ids))
    if np.any(queries[:, -1] != 0):
        raise ValueError('saved query augmentation must be zero')
    validation = verify_rankings(base, queries[:100], ids[:100], block_rows=block_rows)
    save_json(directory / 'ranking-validation.json', validation)
    if not validation['passed']:
        raise ValueError('saved float32 IP/L2 rankings differ; see ranking-validation.json; stop for decision')
    print(f'saved_float32_ip_l2_top10={validation["query_count"]}/{len(queries[:100])}', flush=True)
    started = time.monotonic()

    def progress(done, total):
        print(json.dumps(dict(exact_ip_rows=done, total_rows=total,
                              elapsed_seconds=time.monotonic() - started)), flush=True)

    gt, scores = exact_search(base[:, :-1], queries[:, :-1], 100, block_rows, progress=progress)
    # Full-query GEMM and 100-query verification GEMM may use different BLAS
    # tiles; require the same top ten under the recorded score/ID rule.
    if not np.array_equal(gt[:100, :10], validation['ip_topk']):
        save_json(directory / 'ground-truth-mismatch.json', dict(
            query_ids=ids[:100], verification_top10=validation['ip_topk'],
            full_top10=gt[:100, :10].tolist(), full_scores=scores[:100, :10].tolist()))
        raise ValueError('full ground truth differs from verified IP top10; stop for decision')
    target = directory / 'msmarco_gt.ivecs'
    write_vecs(target, gt, integer=True)
    record = dict(query_count=len(ids), base_count=BASE_COUNT, k=100, metric='768-dimensional inner product',
                  arithmetic='float64 exhaustive block matrix multiplication of saved float32 coordinates',
                  tie_rule=TIE_RULE, block_rows=block_rows, elapsed_seconds=time.monotonic() - started,
                  output=file_record(target), validation=file_record(directory / 'ranking-validation.json'))
    save_json(directory / 'gt-manifest.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['base', 'queries', 'ground-truth'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--block-rows', type=int, default=65536)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.block_rows <= 0 or args.batch_size <= 0:
        parser.error('block rows and batch size must be positive')
    if not (args.directory / 'sources.json').is_file():
        raise ValueError('verified source manifest required before preparation')
    started = time.monotonic()
    result = (prepare_base(args.directory, args.block_rows) if args.stage == 'base' else
              prepare_queries(args.directory, args.batch_size) if args.stage == 'queries' else
              prepare_ground_truth(args.directory, args.block_rows))
    result.update(stage=args.stage, wall_seconds=time.monotonic() - started,
                  python=platform.python_version(), numpy=np.__version__,
                  blas_threads={key: os.environ.get(key) for key in
                                ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']})
    save_json(args.directory / f'{args.stage}-execution.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
