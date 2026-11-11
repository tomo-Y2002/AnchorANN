#!/usr/bin/env python3
"""Acquire public image-vector datasets and convert matrices to vecs files."""
import argparse
import gzip
from pathlib import Path
import shutil
import struct
import tarfile
import urllib.request

import numpy as np

TEXMEX = 'ftp://ftp.irisa.fr/local/texmex/corpus/'
DEEP = 'https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/'


def copy_prefix(source, target, count=None):
    remaining = count
    while remaining is None or remaining > 0:
        block = source.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
        if not block:
            if remaining:
                raise ValueError('truncated source')
            break
        target.write(block)
        if remaining is not None:
            remaining -= len(block)


def download(url, output, prefix=None):
    output = Path(output)
    if output.exists():
        raise ValueError(f'output exists: {output}')
    with urllib.request.urlopen(url, timeout=120) as response, output.open('xb') as target:
        if url.endswith('.gz') and not url.endswith('.tar.gz'):
            with gzip.GzipFile(fileobj=response) as source:
                copy_prefix(source, target, prefix)
        else:
            copy_prefix(response, target, prefix)


def matrix(path, limit=None):
    path = Path(path)
    if path.suffix == '.npy':
        values = np.load(path, mmap_mode='r')
    elif path.suffix in {'.fbin', '.ibin'}:
        with path.open('rb') as f:
            rows, dim = struct.unpack('<II', f.read(8))
        dtype = '<f4' if path.suffix == '.fbin' else '<i4'
        values = np.memmap(path, mode='r', dtype=dtype, offset=8, shape=(rows, dim))
    elif path.suffix in {'.fvecs', '.ivecs', '.bvecs'}:
        with path.open('rb') as f:
            dim, = struct.unpack('<I', f.read(4))
        size = dim + 4 if path.suffix == '.bvecs' else 4 * (dim + 1)
        if dim <= 0 or path.stat().st_size % size:
            raise ValueError('invalid vecs record size')
        rows = path.stat().st_size // size
        if path.suffix == '.bvecs':
            raw = np.memmap(path, mode='r', dtype='u1', shape=(rows, size))
            headers = np.ascontiguousarray(raw[:, :4]).view('<u4').ravel()
            values = raw[:, 4:]
        else:
            raw = np.memmap(path, mode='r', dtype='<i4', shape=(rows, dim + 1))
            headers = raw[:, 0]
            values = raw[:, 1:]
            if path.suffix == '.fvecs':
                values = values.view('<f4')
        if not np.all(headers == dim):
            raise ValueError('nonuniform vecs dimensions')
    else:
        raise ValueError('use .npy, .fbin, .ibin, .fvecs, .ivecs or .bvecs')
    if values.ndim != 2 or (limit is not None and len(values) < limit):
        raise ValueError('matrix shape or prefix length differs')
    return values if limit is None else values[:limit]


def write_vecs(output, values, integer=False):
    dtype = '<i4' if integer else '<f4'
    with Path(output).open('xb') as f:
        for start in range(0, len(values), 8192):
            x = values[start:start + 8192]
            records = np.empty((len(x), x.shape[1] + 1), dtype=dtype)
            records[:, 0].view('<i4')[:] = x.shape[1]
            records[:, 1:] = x
            f.write(records.tobytes())


def acquire(dataset, directory, queries=1000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dataset == 'SIFT1M':
        download(TEXMEX + 'bigann_base.bvecs.gz', directory/'base.bvecs', 1_000_000 * 132)
        download(TEXMEX + 'bigann_query.bvecs.gz', directory/'query.bvecs', queries * 132)
        with urllib.request.urlopen(TEXMEX+'bigann_gnd.tar.gz', timeout=120) as response:
            with tarfile.open(fileobj=response, mode='r|gz') as archive:
                for member in archive:
                    if Path(member.name).name == 'idx_1M.ivecs' and member.isfile():
                        with archive.extractfile(member) as source, (directory/'gt.ivecs').open('xb') as target:
                            shutil.copyfileobj(source, target)
                        break
                else:
                    raise ValueError('SIFT1M ground truth missing')
    elif dataset == 'GIST1M':
        with urllib.request.urlopen(TEXMEX+'gist.tar.gz', timeout=120) as response:
            with tarfile.open(fileobj=response, mode='r|gz') as archive:
                for member in archive:
                    names = {'gist_base.fvecs':'base.fvecs', 'gist_query.fvecs':'query.fvecs', 'gist_groundtruth.ivecs':'gt.ivecs'}
                    if member.isfile() and Path(member.name).name in names:
                        with archive.extractfile(member) as source, (directory/names[Path(member.name).name]).open('xb') as target:
                            shutil.copyfileobj(source, target)
        if not all((directory/name).exists() for name in names.values()):
            raise ValueError('GIST archive members missing')
    else:
        # Download only the full header plus the 1M prefix of the 1B base.
        download(DEEP+'base.1B.fbin', directory/'base.fbin', 8 + 1_000_000*96*4)
        with (directory/'base.fbin').open('r+b') as f:
            f.write(struct.pack('<II', 1_000_000, 96))
        download(DEEP+'query.public.10K.fbin', directory/'query.fbin')
        write_vecs(directory/'base.fvecs', matrix(directory/'base.fbin', 1_000_000))
        write_vecs(directory/'query.fvecs', matrix(directory/'query.fbin', queries))
        exact_groundtruth(directory/'base.fvecs', directory/'query.fvecs', directory/'gt.ivecs')


def exact_groundtruth(base_path, query_path, output, k=100):
    base, queries = matrix(base_path), matrix(query_path)
    if base.shape[1] != queries.shape[1] or len(base) < k:
        raise ValueError('ground-truth geometry differs')
    truth = []
    for query in queries:
        best = np.empty(0, dtype=np.int64)
        scores = np.empty(0)
        for start in range(0, len(base), 65536):
            values = np.asarray(base[start:start+65536], dtype=np.float64)
            distance = np.square(values - query.astype(np.float64)).sum(axis=1)
            ids = np.arange(start, start+len(values))
            ids = np.concatenate((best, ids)); distance = np.concatenate((scores, distance))
            order = np.lexsort((ids, distance))[:k]
            best, scores = ids[order], distance[order]
        truth.append(best)
    write_vecs(output, np.asarray(truth), integer=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('download')
    get.add_argument('dataset', choices=['SIFT1M','DEEP1M','GIST1M'])
    get.add_argument('--directory', type=Path, required=True)
    get.add_argument('--queries', type=int, default=1000)
    convert = sub.add_parser('convert')
    convert.add_argument('input', type=Path)
    convert.add_argument('output', type=Path)
    convert.add_argument('--limit', type=int)
    gt = sub.add_parser('ground-truth')
    gt.add_argument('--base', required=True)
    gt.add_argument('--queries', required=True)
    gt.add_argument('--output', required=True)
    gt.add_argument('--k', type=int, default=100)
    args = parser.parse_args()
    if args.command == 'download':
        acquire(args.dataset, args.directory, args.queries)
    elif args.command == 'convert':
        write_vecs(args.output, matrix(args.input, args.limit), integer=args.output.suffix=='.ivecs')
    else:
        exact_groundtruth(args.base, args.queries, args.output, args.k)


if __name__ == '__main__':
    main()
