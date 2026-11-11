import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import datasets


class DatasetTest(unittest.TestCase):
    def test_vecs_conversion_and_exhaustive_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = np.array([[1,0],[0,1],[1,0]], dtype='<f4')
            query = np.array([[1,0]], dtype='<f4')
            datasets.write_vecs(root/'base.fvecs', base)
            datasets.write_vecs(root/'query.fvecs', query)
            np.testing.assert_array_equal(datasets.matrix(root/'base.fvecs'), base)
            datasets.exact_groundtruth(root/'base.fvecs',root/'query.fvecs',root/'gt.ivecs',2)
            np.testing.assert_array_equal(datasets.matrix(root/'gt.ivecs'),[[0,2]])
            with self.assertRaises(ValueError):
                datasets.matrix(root/'base.fvecs',4)
            with self.assertRaises(FileExistsError):
                datasets.write_vecs(root/'base.fvecs',base)

    def test_bvecs_and_fbin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'base.bvecs').write_bytes(struct.pack('<I',2)+bytes([1,2]))
            np.testing.assert_array_equal(datasets.matrix(root/'base.bvecs'),[[1,2]])
            (root/'base.fbin').write_bytes(struct.pack('<II',1,2)+np.array([3,4],dtype='<f4').tobytes())
            np.testing.assert_array_equal(datasets.matrix(root/'base.fbin'),[[3,4]])
            with self.assertRaises(ValueError):
                datasets.copy_prefix(io.BytesIO(b'12'),io.BytesIO(),3)


if __name__ == '__main__':
    unittest.main()
