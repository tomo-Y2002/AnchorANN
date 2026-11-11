import tempfile
from pathlib import Path
import unittest
import numpy as np
import prepare_msmarco_pca as m


class PCATest(unittest.TestCase):
    def test_top_set_agreement_and_boundary_mismatch(self):
        e=np.finfo(np.float32).eps
        base=np.array([[1,0,1],[1+e,0,1+e]],dtype='<f4')
        q=np.array([[1,0,0]],dtype='<f4')
        ip,l2=m.exact_truth(base,q,k=2,block=2)
        np.testing.assert_array_equal(ip,[[1,0]])
        np.testing.assert_array_equal(l2,[[0,1]])
        with self.assertRaises(ValueError):m.exact_truth(base,q,k=1,block=2)
    def test_covariance_and_uncentered_projection(self):
        x = np.random.default_rng(711).normal(size=(53, 7)) + 5
        mean, matrix, ratio = m.principal_components(x, 3, block=8)
        eigenvalues, _ = np.linalg.eigh(np.cov(x.T))
        np.testing.assert_allclose(mean, x.mean(axis=0))
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(ratio, eigenvalues[-3:].sum()/eigenvalues.sum())
        self.assertFalse(np.allclose(x @ matrix, (x-mean) @ matrix))

    def test_exhaustive_ip_l2_ties_and_vecs(self):
        x = np.random.default_rng(4).normal(size=(70, 6)).astype('<f4')
        x[1] = x[0]
        q = x[:4]
        m2 = np.square(x.astype(np.float64)).sum(axis=1).max()
        base = np.column_stack((x, np.sqrt(m2-np.square(x.astype(np.float64)).sum(axis=1)))).astype('<f4')
        query = np.column_stack((q, np.zeros(4))).astype('<f4')
        a, b = m.exact_truth(base, query, k=10, block=13)
        expected = np.array([np.lexsort((np.arange(len(x)), -(x.astype(np.float64)@r)))[:10] for r in q])
        np.testing.assert_array_equal(a, expected)
        np.testing.assert_array_equal(a, b)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'base.fvecs'
            m.write_vecs(p, base)
            np.testing.assert_array_equal(m.vecs(p, 7), base)
            m.write_vecs(Path(tmp)/'gt.ivecs', a, integer=True)
            self.assertEqual((Path(tmp)/'gt.ivecs').stat().st_size, 4*11*4)

    def test_qrels_remaps_retained_ids_and_counts_lost_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            (r/'passages').write_text('row\tpassage_id\n0\t42\n1\t99\n')
            (r/'queries').write_text('row\tquery_id\n0\t7\n1\t3\n')
            (r/'source').write_text('7 0 42 1\n7 0 100 1\n7 0 101 1\n3 0 99 1\n8 0 100 1\n')
            self.assertEqual(m.qrels(r/'source',r/'queries',r/'passages',r/'out'),1)
            self.assertEqual((r/'out').read_text(),'[[0], [1]]\n')


if __name__ == '__main__':
    unittest.main()
