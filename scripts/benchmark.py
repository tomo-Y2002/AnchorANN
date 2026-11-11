#!/usr/bin/env python3
"""Run an AnchorANN probe sweep on localhost and write network-model inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'paper/scripts'))
from network_model import export


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(records, report, dataset, probe, size, width):
    if not records or [r['query'] for r in records] != list(range(len(records))):
        raise ValueError('incomplete query prefix')
    rows = []
    for record in records:
        r = record['result']
        if r.get('error'):
            raise ValueError('query failed')
        if len(r['ids']) != 10 or len(set(r['ids'])) != 10:
            raise ValueError('missing or duplicate answer IDs')
        rows.append(dict(query=record['query'], ids=r['ids'], online_seconds=r['timing']['online_seconds'],
                         communication_rounds=r['rpcs'], up_bytes=r['online_up_bytes'], down_bytes=r['online_down_bytes']))
    p = report['n_probe'][0]
    run = p['runs'][0]
    if report['s'] != size or run['n_probe'] != probe or run['l_search'] != width:
        raise ValueError('measured configuration differs')
    if len(records) != report['queries'] or run['one_recall_at_k'] is None:
        raise ValueError('ground truth or measured queries missing')
    return dict(dataset=dataset, method='anchorann', parameters=dict(m=probe, s=size, l_search=width, variant='full'),
                q=len(rows), one_recall=run['one_recall_at_k'], queries=rows,
                regeneration=dict(count=run['refreshes'], seconds=run['refresh_seconds'],
                                  down_bytes=run['refresh_down_bytes'], up_bytes=run['refresh_up_bytes'],
                                  interval_queries=p['queries_per_refresh']))


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, GOMAXPROCS='1')
    def binary(name):
        candidate = args.bin_dir/name
        if candidate.is_file():
            return str(candidate.resolve())
        installed = shutil.which(name)
        if installed:
            return installed
        candidate.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['go','build','-o',str(candidate),'./anchorann/cmd/'+name], cwd=ROOT, env=env, check=True)
        return str(candidate.resolve())
    server, client = binary('search-server'), binary('search-client')
    endpoint = f'127.0.0.1:{args.port}'
    def pin(cpu):
        return [] if cpu is None else ['taskset','-c',str(cpu)]
    command = [*pin(args.server_cpu),server,'-addr',endpoint,'-vectors',str(args.base.resolve()),
               '-artifact',str(args.artifact.resolve()),'-bundle',str(args.bundle.resolve())]
    server_log = output/'server.json'
    with server_log.open('w') as log, (output/'server.stderr.log').open('w') as errors:
        # nohup, a separate process group and file logs keep the server in the background.
        process = subprocess.Popen(['nohup',*command], stdout=log, stderr=errors, env=env, start_new_session=True)
        try:
            deadline = time.monotonic()+args.ready_timeout
            while True:
                time.sleep(0.1)
                if process.poll() is not None:
                    raise RuntimeError('server exited; see server.stderr.log')
                try:
                    with socket.create_connection(('127.0.0.1',args.port), timeout=0.2):
                        if server_log.stat().st_size:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('server readiness timed out')
            results = []
            for probe in args.probes:
                prefix = output/f'm{probe}'
                client_cmd = [*pin(args.client_cpu),client,'-addr',endpoint,'-queries',str(args.queries.resolve()),'-gt',str(args.gt.resolve()),
                              '-query-limit',str(args.query_count),'-k','10','-n-probe',str(probe),'-l-search',str(args.l_search),
                              '-pir-rpc','batch','-per-query',str(prefix.with_suffix('.jsonl'))]
                with prefix.with_suffix('.report.json').open('w') as report_log:
                    subprocess.run(client_cmd, stdout=report_log, check=True, env=env, timeout=args.timeout)
                report = json.loads(prefix.with_suffix('.report.json').read_text())
                raw = [json.loads(line) for line in prefix.with_suffix('.jsonl').read_text().splitlines()]
                result = normalize(raw, report, args.dataset, probe, args.size, args.l_search)
                result['provenance'] = dict(server_command=command, client_command=client_cmd,
                    inputs={key:sha256(getattr(args,key)) for key in ('base','queries','gt','artifact','bundle')},
                    binaries={'server':sha256(server),'client':sha256(client)})
                path = prefix.with_suffix('.json')
                path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                results.append(path)
            export(results, output/'model')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    print(output/'model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['SIFT1M','DEEP1M','GIST1M','MSMARCO'])
    for key in ('base','queries','gt','artifact','bundle','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=ROOT/'bin')
    parser.add_argument('--probes', default='4,8,16,32,64,128')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--l-search', type=int, default=128)
    parser.add_argument('--query-count', type=int, default=1000)
    parser.add_argument('--port', type=int, default=50091)
    parser.add_argument('--ready-timeout', type=float, default=3600)
    parser.add_argument('--timeout', type=float, default=86400)
    parser.add_argument('--server-cpu', type=int, help='pin the server to this CPU with taskset')
    parser.add_argument('--client-cpu', type=int, help='pin the client to this CPU with taskset')
    args = parser.parse_args()
    args.probes = [int(p) for p in args.probes.split(',')]
    if not args.probes or min(args.probes) <= 0 or max(args.probes) > args.l_search or args.query_count <= 0:
        parser.error('positive probes <= l-search and query-count required')
    run(args)


if __name__ == '__main__':
    main()
