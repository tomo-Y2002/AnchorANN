import ctypes
import ctypes.util
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

import prepare_msmarco as prepare
import download_msmarco as source


class PreparationTest(unittest.TestCase):
    @staticmethod
    def npids_bytes(count):
        header = b'{"version": 1}'
        metadata = json.dumps(dict(format='intsequence', prefix='', start=0)).encode()
        leaf = json.dumps(dict(format='intsequence', prefix='', start=0, count=count)).encode()
        offset = 24 + len(header)
        return (b'NPID' + struct.pack('<QQI', offset, 0, len(header)) + header +
                struct.pack('<IqQI', 0, offset + 24 + len(metadata), count, len(metadata)) + metadata +
                struct.pack('<IqQI', 1, -1, count, len(leaf)) + leaf)

    def test_exclusions_use_numeric_id_and_preserve_rows(self):
        rows, excluded = prepare.retained_indices(np.array([1, 100, 5, 99, 20]), 2)
        np.testing.assert_array_equal(rows, [0, 2, 4])
        np.testing.assert_array_equal(excluded, [100, 99])
        for ids, count in [(np.array([1, 1]), 0), (np.array([-1, 2]), 0),
                           (np.array([1., 2.]), 0), (np.array([1, 2]), 2)]:
            with self.assertRaises(ValueError):
                prepare.retained_indices(ids, count)

    def test_npids_integer_sequence_and_reject_unknown_layout(self):
        raw = self.npids_bytes(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ids.npids'
            path.write_bytes(raw)
            np.testing.assert_array_equal(prepare.read_npids(path, 4), np.arange(4))
            with self.assertRaises(ValueError):
                prepare.read_npids(path, 5)
            for broken in [raw[:30], raw + b'x', raw.replace(b'intsequence', b'notsequence')]:
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    prepare.read_npids(path, 4)

    def test_augmentation_and_float32_roundtrip(self):
        base = np.array([[3, 4], [1, 2], [0, 0]], np.float32)
        augmented, metadata = prepare.augment_base(base, 5.)
        np.testing.assert_array_equal(augmented[:, :-1], base)
        np.testing.assert_allclose(prepare.squared_norms(augmented), [25, 25, 25], rtol=1e-7)
        self.assertEqual(metadata['clamped_count'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'base.fvecs'
            prepare.write_vecs(path, augmented)
            np.testing.assert_array_equal(prepare.read_vecs(path, 3, 3), augmented)
            self.assertEqual(path.read_bytes()[:4], struct.pack('<i', 3))
            with self.assertRaises(FileExistsError):
                prepare.write_vecs(path, augmented)
            with path.open('r+b') as stream:
                stream.seek(16)
                stream.write(struct.pack('<i', 2))
            with self.assertRaises(ValueError):
                prepare.read_vecs(path, 3, 3)
        np.testing.assert_array_equal(prepare.augment_queries(base)[:, -1], 0)

    def test_rounding_only_clamps_small_negatives(self):
        augmented, metadata = prepare.augment_base(np.ones((1, 3), np.float32), np.sqrt(3.))
        self.assertEqual(metadata['clamped_count'], 1)
        self.assertEqual(augmented[0, -1], 0.)
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            prepare.augment_base(np.array([[3, 4]], np.float32), 4.999)
        for bad in [float('nan'), float('inf'), -1., 1e308]:
            with self.assertRaises(ValueError):
                prepare.augment_base(np.ones((1, 2)), bad)
        for bad in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                prepare.augment_base(np.array([[bad]]), 2.)
            with self.assertRaises(ValueError):
                prepare.augment_queries(np.array([[bad]]))

    def test_ground_truth_matches_full_brute_force_across_blocks(self):
        rng = np.random.default_rng(868)
        base = rng.normal(size=(39, 7)).astype(np.float32)
        queries = rng.normal(size=(6, 7)).astype(np.float32)
        scores = queries.astype(np.float64) @ base.astype(np.float64).T
        expected = np.stack([np.lexsort((np.arange(len(base)), -row))[:10] for row in scores])
        for block_rows in [1, 4, 12, 100]:
            ids, result = prepare.exact_search(base, queries, 10, block_rows)
            np.testing.assert_array_equal(ids, expected)
            np.testing.assert_allclose(result, np.take_along_axis(scores, expected, axis=1), rtol=1e-14)
        maximum = np.sqrt(prepare.squared_norms(base).max())
        augmented, _ = prepare.augment_base(base, maximum)
        with tempfile.TemporaryDirectory() as tmp:
            bp, qp = Path(tmp) / 'base.fvecs', Path(tmp) / 'query.fvecs'
            prepare.write_vecs(bp, augmented)
            prepare.write_vecs(qp, prepare.augment_queries(queries))
            saved_base = prepare.read_vecs(bp, 8, len(base))
            saved_queries = prepare.read_vecs(qp, 8, len(queries))
            validation = prepare.verify_rankings(saved_base, saved_queries, [str(i) for i in range(6)], block_rows=9)
            self.assertTrue(validation['passed'])
            distances = np.sum((saved_queries.astype(np.float64)[:, None, :] - saved_base.astype(np.float64)[None, :, :]) ** 2, axis=2)
            expected_l2 = np.stack([np.lexsort((np.arange(len(base)), row))[:10] for row in distances])
            np.testing.assert_array_equal(validation['l2_topk'], expected_l2)

    def test_ties_use_smallest_number_even_across_partition_boundary(self):
        ids, scores = prepare.topk(np.array([2., 3., 3., 3., 3.]), np.array([10, 9, 8, 1, 7]), 3)
        np.testing.assert_array_equal(ids, [1, 7, 8])
        np.testing.assert_array_equal(scores, [3, 3, 3])
        base = np.ones((19, 4), np.float32)
        for metric in ['ip', 'l2']:
            ids, _ = prepare.exact_search(base, np.ones((2, 4)), 7, 3, metric)
            np.testing.assert_array_equal(ids, np.tile(np.arange(7), (2, 1)))

    def test_rank_mismatch_records_query_and_both_scores(self):
        base = np.array([[2, 10], [1, 0]], np.float32)
        query = np.array([[1, 0]], np.float32)
        result = prepare.verify_rankings(base, query, ['query-42'], k=1, block_rows=1)
        self.assertFalse(result['passed'])
        mismatch = result['mismatches'][0]
        self.assertEqual(mismatch['query_id'], 'query-42')
        self.assertEqual(mismatch['ip_topk'], [0])
        self.assertEqual(mismatch['l2_topk'], [1])
        self.assertEqual(mismatch['candidates'][0]['inner_product'], 2.)
        self.assertEqual(mismatch['candidates'][0]['squared_l2'], 101.)

    def test_ivecs_and_invalid_search_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gt.ivecs'
            values = np.array([[3, 2, 1], [0, 2, 5]])
            prepare.write_vecs(path, values, integer=True)
            np.testing.assert_array_equal(prepare.read_vecs(path, 3, 2, integer=True), values)
            with self.assertRaises(ValueError):
                prepare.write_vecs(Path(tmp) / 'bad.ivecs', np.array([[-1]]), integer=True)
        for kwargs in [dict(k=0), dict(k=3), dict(k=1, block_rows=0), dict(k=1, metric='cosine')]:
            with self.assertRaises(ValueError):
                prepare.exact_search(np.ones((2, 3)), np.ones((1, 3)), **kwargs)
        with self.assertRaises(ValueError):
            prepare.exact_search(np.array([[np.inf, 0]]), np.ones((1, 2)), 1)
        with self.assertRaises(ValueError):
            prepare.exact_search(np.ones((2, 3)), np.full((1, 3), np.nan), 1)

    def test_query_distribution_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'queries.tsv'
            path.write_text('99\tsecond?\n1\tfirst?\n42\tthird?\n')
            ids, text = prepare.read_queries(path, expected_count=3)
            self.assertEqual(ids, ['99', '1', '42'])
            self.assertEqual(text, ['second?', 'first?', 'third?'])
            for broken in ['1\tquery\n1\tduplicate\n', 'x\tquery\n', '1\t\n']:
                path.write_text(broken)
                with self.assertRaises(ValueError):
                    prepare.read_queries(path, expected_count=2)

    def test_complete_small_pipeline_outputs_share_id_order_and_exact_truth(self):
        rng = np.random.default_rng(764)
        vectors = rng.normal(size=(123, 3)).astype(np.float32)
        queries = rng.normal(size=(4, 3)).astype(np.float32)
        distribution_ids = ['99', '1', '42', '3']
        reader = prepare.read_queries
        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(
                prepare, SOURCE_COUNT=123, BASE_COUNT=120, DIMENSION=3), mock.patch.object(
                prepare, 'read_queries', side_effect=lambda path: reader(path, 4)), mock.patch.object(
                prepare, 'encode_queries', return_value=(queries, {'model': 'fixture'})):
            directory = Path(tmp)
            (directory / 'docnos.npids').write_bytes(self.npids_bytes(123))
            vectors.tofile(directory / 'vecs.f4')
            (directory / 'queries.dev.small.tsv').write_text(''.join(f'{qid}\tquery {qid}\n' for qid in distribution_ids))
            base_record = prepare.prepare_base(directory, 17)
            query_record = prepare.prepare_queries(directory, 2)
            gt_record = prepare.prepare_ground_truth(directory, 23)
            self.assertEqual(base_record['excluded_passage_ids'], [122, 121, 120])
            self.assertEqual(query_record['count'], 4)
            self.assertEqual(gt_record['k'], 100)
            self.assertEqual((directory / 'first100_query_ids.txt').read_text().splitlines(), distribution_ids)
            self.assertEqual((directory / 'query_ids.tsv').read_text().splitlines(),
                             ['row\tquery_id', '0\t99', '1\t1', '2\t42', '3\t3'])
            gt = prepare.read_vecs(directory / 'msmarco_gt.ivecs', 100, 4, integer=True)
            scores = queries.astype(np.float64) @ vectors[:120].astype(np.float64).T
            expected = np.stack([np.lexsort((np.arange(120), -row))[:100] for row in scores])
            np.testing.assert_array_equal(gt, expected)
            for record in [base_record['output'], query_record['output'], gt_record['output']]:
                self.assertEqual(source.file_record(directory / record['file']), record)


class SourceTest(unittest.TestCase):
    @staticmethod
    def compress(raw):
        name = ctypes.util.find_library('lz4')
        if not name:
            raise AssertionError('system liblz4 is required for the direct artifact reader tests')
        lib = ctypes.CDLL(name)
        lib.LZ4F_compressFrameBound.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        lib.LZ4F_compressFrameBound.restype = ctypes.c_size_t
        lib.LZ4F_compressFrame.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        lib.LZ4F_compressFrame.restype = ctypes.c_size_t
        output = ctypes.create_string_buffer(lib.LZ4F_compressFrameBound(len(raw), None))
        size = lib.LZ4F_compressFrame(output, len(output), raw, len(raw), None)
        return output.raw[:size]

    def test_concatenated_empty_and_split_lz4_frames(self):
        expected = b'test content' * 100000
        compressed = self.compress(b'') + self.compress(expected[:600000]) + self.compress(expected[600000:])
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / 'part0', Path(tmp) / 'part1']
            paths[0].write_bytes(compressed[:27])
            paths[1].write_bytes(compressed[27:])
            self.assertEqual(b''.join(source.lz4_chunks(paths)), expected)
            paths[1].write_bytes(compressed[27:-2])
            with self.assertRaises(ValueError):
                b''.join(source.lz4_chunks(paths))

    def test_safe_streaming_extraction_and_member_rejection(self):
        files = {'docnos.npids': b'ids', 'pt_meta.json': b'{}', 'vecs.f4': b'\0' * 24}
        for extra in [None, '../escape', 'unknown']:
            tar = io.BytesIO()
            with tarfile.open(fileobj=tar, mode='w') as archive:
                for name, contents in dict(files, **({extra: b'x'} if extra else {})).items():
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    archive.addfile(info, io.BytesIO(contents))
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                path = directory / 'part'
                path.write_bytes(self.compress(tar.getvalue()))
                layout = [dict(path=name, size=len(contents)) for name, contents in files.items()]
                if extra:
                    with self.assertRaises(ValueError):
                        source.extract_flex([path], directory, layout)
                else:
                    records = source.extract_flex([path], directory, layout)
                    for name, contents in files.items():
                        self.assertEqual((directory / name).read_bytes(), contents)
                        self.assertEqual(records[name]['bytes'], len(contents))
                    self.assertEqual(source.extract_flex([path], directory, layout), records)
                    (directory / 'vecs.f4').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError, 'differs from archive'):
                        source.extract_flex([path], directory, layout)

    def test_disk_gate_records_and_refuses_insufficient_capacity(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(source.shutil, 'disk_usage') as disk:
            disk.return_value = source.shutil._ntuple_diskusage(1000, 950, 50)
            with self.assertRaisesRegex(ValueError, 'download refused'):
                source.free_space(Path(tmp), 200)
            record = json.loads((Path(tmp) / 'disk-preflight.json').read_text())
            self.assertEqual(record['free_bytes'], 50)
            self.assertEqual(record['required_free_bytes'], 200)

    def test_acquisition_validates_compressed_archive_hash_and_exact_query_member(self):
        files = {'docnos.npids': b'ids',
                 'pt_meta.json': json.dumps(dict(type='dense_index', format='flex', vec_size=3, doc_count=5)).encode(),
                 'vecs.f4': np.arange(15, dtype='<f4').tobytes()}
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode='w') as archive:
            for name, contents in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        compressed = self.compress(raw_tar.getvalue())
        queries = b'99\tfirst query\n1\tsecond query\n'
        query_tar = io.BytesIO()
        with tarfile.open(fileobj=query_tar, mode='w:gz') as archive:
            for name, contents in [('collection.tsv', b'ignored'), ('queries.dev.small.tsv', queries)]:
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        for use_compressed_hash in [True, False]:
            payloads = {source.QUERY_ARCHIVE: query_tar.getvalue()}
            split = len(compressed) // 6
            for i in range(6):
                payloads[f'artifact.tar.lz4.{i}'] = compressed[i * split:(i + 1) * split] if i < 5 else compressed[i * split:]
            expected = hashlib.sha256(compressed if use_compressed_hash else b''.join(files.values())).hexdigest()
            payloads['artifact.tar.lz4.json'] = json.dumps(dict(
                expected_sha256=expected, contents=[dict(path=name, size=len(contents)) for name, contents in files.items()])).encode()

            def fetch(request, timeout):
                url = request if isinstance(request, str) else request.full_url
                name = url.rsplit('/', 1)[-1]
                payload = payloads[name]
                if '/raw/' in url:
                    payload = (f'version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(payload).hexdigest()}\nsize {len(payload)}\n').encode()
                response = io.BytesIO(payload)
                response.status = 200
                return response

            with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(source, SOURCE_COUNT=5, DIMENSION=3), \
                    mock.patch.object(source, 'free_space', return_value={'free_bytes': 200_000_000_000}), \
                    mock.patch.object(source.urllib.request, 'urlopen', side_effect=fetch), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                if use_compressed_hash:
                    manifest = source.acquire(Path(tmp))
                    self.assertEqual(manifest['archive_sha256'], expected)
                    self.assertEqual(manifest['archive_bytes'], len(compressed))
                    self.assertEqual((Path(tmp) / 'queries.dev.small.tsv').read_bytes(), queries)
                    self.assertFalse((Path(tmp) / 'collection.tsv').exists())
                else:
                    with self.assertRaisesRegex(ValueError, 'compressed archive checksum'):
                        source.acquire(Path(tmp))


if __name__ == '__main__':
    unittest.main()
