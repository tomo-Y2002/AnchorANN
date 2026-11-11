#!/usr/bin/env python3
"""Derive online and stationary amortized QPS from single-host measurements."""
import argparse
import csv
import json
import math
from pathlib import Path

# RTT seconds, bandwidth bits/second. Convert bandwidth to bytes/second below.
LAN = (0.005, 1_000_000_000)
WAN = (0.050, 100_000_000)
SETTINGS = {'LAN': LAN, 'WAN': WAN}
COLUMNS = ('setting','dataset','method','order','one_recall','qps')
METHODS = {'anchorann':'AnchorANN','pacmann':'Pacmann','amora':'Amora','nsg':'NSG'}


def modeled(result, setting):
    rtt, bandwidth = SETTINGS[setting]
    bps = bandwidth/8
    rows = result['queries']
    if len(rows) != result['q'] or not rows:
        raise ValueError('incomplete run')
    online = sum(r['online_seconds'] + r['communication_rounds']*rtt +
                 (r['up_bytes']+r['down_bytes'])/bps for r in rows)/len(rows)
    refresh = result['regeneration']
    extra = 0
    if refresh['count']:
        interval = refresh['interval_queries']
        if not interval or interval <= 0:
            raise ValueError('refresh interval missing')
        mean_refresh = (refresh['seconds']+refresh['down_bytes']/bps)/refresh['count']
        extra = mean_refresh/interval
    if not math.isfinite(online+extra) or online <= 0:
        raise ValueError('invalid modeled time')
    return 1/online, 1/(online+extra)


def sort_key(result):
    p=result['parameters']
    variable={'anchorann':'m','pacmann':'step','amora':'t','nsg':'l_search'}[result['method']]
    return (result['dataset'],result['method'],p.get('s',0),p.get('variant',''),p[variable])


def export(paths, output):
    results=sorted([json.loads(Path(p).read_text()) for p in paths], key=sort_key)
    seen=set()
    online,amortized=[],[]
    for setting in SETTINGS:
        counts={}
        for result in results:
            key=(setting,result['dataset'],result['method'],json.dumps(result['parameters'],sort_keys=True))
            if key in seen:
                raise ValueError('duplicate measurement point')
            seen.add(key)
            group=(result['dataset'],result['method'])
            counts[group]=counts.get(group,0)+1
            prefix=(setting,result['dataset'],METHODS[result['method']],counts[group],result['one_recall'])
            qps,amqps=modeled(result,setting)
            online.append(prefix+(qps,));amortized.append(prefix+(amqps,))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('recall_qps.csv',online),('recall_qps_amortized.csv',amortized)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(COLUMNS);writer.writerows(rows)
    return online,amortized


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',nargs='+',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    args=parser.parse_args()
    paths=[]
    for path in args.results:
        paths.extend(path.rglob('*.json') if path.is_dir() else [path])
    export(paths,args.output_dir)

if __name__=='__main__':main()
