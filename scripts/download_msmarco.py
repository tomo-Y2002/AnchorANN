#!/usr/bin/env python3
"""Download the pinned MS MARCO Flex artifact without importing PyTerrier.

Uses the system liblz4 to read concatenated LZ4 frames and Python's streaming
tar reader. Only the three documented regular files are extracted.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import ctypes.util
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

DATASET = 'macavaney/msmarco-passage.tasb.flex'
DATASET_COMMIT = '87817c21458b5749d0d37b9006e3d5b903a920ca'
MODEL = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'
MODEL_REVISION = 'a03882cb371476b74ca3557366452cd868ac4f42'
QUERY_ARCHIVE = 'collectionandqueries.tar.gz'
QUERIES_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/' + QUERY_ARCHIVE
SOURCE_COUNT = 8_841_823
BASE_COUNT = 8_841_792
DIMENSION = 768
MIN_FREE_BYTES = 200_000_000_000
CHUNK = 1024 * 1024


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def archive_record(paths: list[Path]) -> dict:
    """Hash the compressed segments in order, as the Flex metadata specifies."""
    digest = hashlib.sha256()
    count = 0
    for path in paths:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(CHUNK), b''):
                digest.update(block)
                count += len(block)
    return dict(bytes=count, sha256=digest.hexdigest())


def save_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as target:
        json.dump(data, target, indent=2, allow_nan=False)
        target.write('\n')
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def free_space(directory: Path, required: int = MIN_FREE_BYTES) -> dict:
    usage = shutil.disk_usage(directory)
    record = dict(directory=str(directory.resolve()), total_bytes=usage.total,
                  used_bytes=usage.used, free_bytes=usage.free, required_free_bytes=required)
    save_json(directory / 'disk-preflight.json', record)
    if usage.free < required:
        raise ValueError(f'insufficient disk space: {usage.free} < {required} bytes; download refused')
    return record


def download(url: str, path: Path, expected: dict | None = None) -> dict:
    """Resume only partial files; verify pinned LFS length and SHA before publish."""
    if path.exists():
        record = file_record(path)
        if expected is None or all(record[key] == expected[key] for key in ('bytes', 'sha256')):
            return dict(record, url=url)
        raise ValueError(f'existing download differs from pinned source: {path.name}')
    partial = path.with_suffix(path.suffix + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
            raise ValueError('server did not honor partial download range; partial retained')
        with partial.open('ab' if offset else 'xb') as target:
            shutil.copyfileobj(response, target, CHUNK)
            target.flush()
            os.fsync(target.fileno())
    record = file_record(partial)
    if expected and any(record[key] != expected[key] for key in ('bytes', 'sha256')):
        raise ValueError(f'checksum/size mismatch for {path.name}; partial retained')
    partial.replace(path)
    return dict(record, file=path.name, url=url)


def lz4_chunks(paths: list[Path]):
    """Decode concatenated frames with bounded memory, including empty frames."""
    name = ctypes.util.find_library('lz4')
    if not name:
        raise ValueError('system liblz4 unavailable; install no alternative reader silently')
    lib = ctypes.CDLL(name)
    lib.LZ4F_createDecompressionContext.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
    lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
    lib.LZ4F_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    lib.LZ4F_decompress.restype = ctypes.c_size_t
    lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
    lib.LZ4F_isError.restype = ctypes.c_uint
    context = ctypes.c_void_p()
    if lib.LZ4F_isError(lib.LZ4F_createDecompressionContext(ctypes.byref(context), 100)):
        raise ValueError('cannot create LZ4 decompression context')
    output = ctypes.create_string_buffer(CHUNK)
    result = 0
    try:
        for path in paths:
            with path.open('rb') as stream:
                for raw in iter(lambda: stream.read(CHUNK), b''):
                    source = ctypes.create_string_buffer(raw)
                    offset = 0
                    while offset < len(raw):
                        consumed = ctypes.c_size_t(len(raw) - offset)
                        produced = ctypes.c_size_t(CHUNK)
                        result = lib.LZ4F_decompress(context, output, ctypes.byref(produced),
                                                    ctypes.byref(source, offset), ctypes.byref(consumed), None)
                        if lib.LZ4F_isError(result):
                            raise ValueError('invalid LZ4 frame or checksum')
                        if consumed.value == 0 and produced.value == 0:
                            raise ValueError('LZ4 decoder made no progress')
                        offset += consumed.value
                        if produced.value:
                            yield output.raw[:produced.value]
        # Flush any output buffered after the final input byte.
        while result:
            consumed = ctypes.c_size_t(0)
            produced = ctypes.c_size_t(CHUNK)
            result = lib.LZ4F_decompress(context, output, ctypes.byref(produced), None,
                                        ctypes.byref(consumed), None)
            if lib.LZ4F_isError(result) or not produced.value:
                raise ValueError('truncated LZ4 frame')
            yield output.raw[:produced.value]
    finally:
        lib.LZ4F_freeDecompressionContext(context)


class ChunkReader(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.pending = memoryview(b'')

    def readable(self):
        return True

    def readinto(self, buffer):
        if not self.pending:
            self.pending = memoryview(next(self.chunks, b''))
        count = min(len(buffer), len(self.pending))
        buffer[:count] = self.pending[:count]
        self.pending = self.pending[count:]
        return count


def extract_member(stream, target: Path, expected_size: int) -> dict:
    """Reuse complete members only after comparing their bytes to the archive."""
    existing = target.exists()
    partial = target.with_suffix(target.suffix + '.partial')
    digest = hashlib.sha256()
    count = 0
    with contextlib.nullcontext(None) if existing else partial.open('xb') as output:
        for block in iter(lambda: stream.read(CHUNK), b''):
            digest.update(block)
            count += len(block)
            if output is not None:
                output.write(block)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    record = dict(file=target.name, bytes=count, sha256=digest.hexdigest())
    if count != expected_size:
        raise ValueError('truncated extracted member')
    if existing:
        if file_record(target) != record:
            raise ValueError(f'existing extracted file differs from archive: {target.name}')
    else:
        partial.replace(target)
    return record


def extract_flex(paths: list[Path], directory: Path, contents: list[dict]) -> dict:
    expected = {item['path']: item['size'] for item in contents}
    if set(expected) != {'docnos.npids', 'pt_meta.json', 'vecs.f4'}:
        raise ValueError('unexpected Flex members; direct reader needs review')
    records = {}
    chunks = lz4_chunks(paths)
    reader = io.BufferedReader(ChunkReader(chunks), CHUNK)
    with tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in records:
                raise ValueError('unexpected, duplicate or nonregular Flex member')
            if member.size != expected[member.name]:
                raise ValueError('Flex member size mismatch')
            target = directory / member.name
            with archive.extractfile(member) as stream:
                records[member.name] = extract_member(stream, target, member.size)
    # tar stops at its EOF marker; consume the decoder to check all frames.
    for _ in iter(lambda: reader.read(CHUNK), b''):
        pass
    if set(records) != set(expected):
        raise ValueError('Flex archive is missing members')
    return records


def acquire(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    disk = free_space(directory)
    raw_root = f'https://huggingface.co/datasets/{DATASET}/raw/{DATASET_COMMIT}/'
    root = raw_root.replace('/raw/', '/resolve/')
    downloads = []
    names = ['artifact.tar.lz4.json'] + [f'artifact.tar.lz4.{i}' for i in range(6)]
    for name in names:
        with urllib.request.urlopen(raw_root + name, timeout=60) as response:
            pointer = response.read(4096).decode('ascii')
        match = re.fullmatch(r'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n', pointer)
        if not match:
            raise ValueError('unexpected pinned LFS pointer')
        downloads.append(download(root + name, directory / name,
                                  dict(sha256=match[1], bytes=int(match[2]))))
        print(json.dumps(downloads[-1]), flush=True)
    layout = json.loads((directory / names[0]).read_text())
    parts = [directory / name for name in names[1:]]
    compressed_record = archive_record(parts)
    if compressed_record['sha256'] != layout['expected_sha256']:
        raise ValueError('combined compressed archive checksum mismatch')
    records = extract_flex(parts, directory, layout['contents'])
    metadata = json.loads((directory / 'pt_meta.json').read_text())
    if metadata != dict(type='dense_index', format='flex', vec_size=DIMENSION, doc_count=SOURCE_COUNT):
        raise ValueError('unexpected Flex shape/metadata')
    query_record = download(QUERIES_URL, directory / QUERY_ARCHIVE)
    query_path = directory / 'queries.dev.small.tsv'
    with tarfile.open(directory / QUERY_ARCHIVE, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.name == 'queries.dev.small.tsv']
        if len(members) != 1 or not members[0].isfile():
            raise ValueError('dev small query distribution missing or ambiguous')
        with archive.extractfile(members[0]) as stream:
            extract_member(stream, query_path, members[0].size)
    manifest = dict(dataset=DATASET, dataset_commit=DATASET_COMMIT, model=MODEL,
                    model_revision=MODEL_REVISION, disk_preflight=disk, downloads=downloads,
                    extracted=records, archive_sha256=compressed_record['sha256'], archive_bytes=compressed_record['bytes'],
                    query_archive=query_record, queries=file_record(query_path))
    save_json(directory / 'sources.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.directory), indent=2))
