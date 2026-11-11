import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
spec=importlib.util.spec_from_file_location('network_model',Path(__file__).with_name('network_model.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ModelTest(unittest.TestCase):
    def result(self):
        return dict(dataset='SIFT1M',method='anchorann',parameters={'m':128,'s':32,'variant':'full'},q=2,one_recall=.5,
                    queries=[dict(online_seconds=.1,communication_rounds=1,up_bytes=125000,down_bytes=125000)]*2,
                    regeneration=dict(count=2,seconds=4,down_bytes=250000000,interval_queries=10))
    def test_units_and_stationary_amortization(self):
        r=self.result();online,am=m.modeled(r,'LAN')
        self.assertAlmostEqual(online,1/.107)
        self.assertAlmostEqual(am,1/(.107+.3))
        self.assertAlmostEqual(m.modeled(r,'WAN')[0],1/.17)
        r['regeneration']['count']=0
        self.assertEqual(*m.modeled(r,'LAN'))
    def test_csv_columns_duplicate_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'point.json';p.write_text(json.dumps(self.result()))
            m.export([p],tmp)
            with (Path(tmp)/'recall_qps.csv').open() as f:
                rows=list(csv.reader(f))
            self.assertEqual(rows[0],list(m.COLUMNS));self.assertEqual(len(rows),3)
            with self.assertRaises(ValueError):m.export([p,p],tmp)
        r=self.result();r['q']=1000
        with self.assertRaises(ValueError):m.modeled(r,'LAN')

if __name__=='__main__':unittest.main()
