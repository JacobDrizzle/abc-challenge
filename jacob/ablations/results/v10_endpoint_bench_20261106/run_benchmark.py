"""Run the existing team benchmark with isolated output paths on frozen ONNX files."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from bench import cli, data, results

INPUTS = json.loads((OUT / 'benchmark_inputs.json').read_text())
SCRATCH = ROOT / 'bench_runs' / OUT.name
MACHINE = 'local-800-comparison'
BASELINES = {'hold_last':'hold_last','constant_velocity_1step':'constant_velocity','damped_velocity':'damped_velocity'}


def score_one(cohort, name):
    os.chdir(ROOT)
    results.RESULTS = OUT / 'scores'
    results.RUNS = SCRATCH / 'windows'
    source = (INPUTS['new_models'] | INPUTS['timing_comparators']).get(name)
    if source:
        path = source['path']
        assert data.sha256_file(path) == source['sha256']
    else:
        path = 'baseline:' + BASELINES[name]
    expanded = cohort == 'merged-800-test160'
    definition = INPUTS['expanded']['definition'] if expanded else INPUTS['benchmark']
    h5 = INPUTS['expanded']['h5'] if expanded else INPUTS['h5']['path']
    target = results.result_path(cohort, name, MACHINE)
    expected_fingerprint = INPUTS['expanded']['cohort_fingerprint'] if expanded else INPUTS['cohort_fingerprint']
    if target.exists():
        saved = json.loads(target.read_text())
        assert saved['model']['onnx_sha256'] == (source['sha256'] if source else None) and saved['bench']['cohort_fingerprint'] == expected_fingerprint
        assert (not saved['model']['stateful'] or saved['accuracy_cold_start']) and saved['contamination']['status'] == 'clean'
        return
    args = ['score', str(path), '--bench', definition, '--h5', h5, '--name', name, '--machine', MACHINE,
            '--protocol', 'both', '--threads', '1', '--verbose']
    if name in INPUTS['new_models']:
        args += ['--trained-on', INPUTS['trained_on']['path'], '--declared-params', '7555']
    if cli.main(args):
        raise RuntimeError(f'Scoring failed for {cohort}/{name}')


def job(spec):
    cohort, name = spec
    log = SCRATCH / f'{cohort}_{name}.log'
    print(f'Start {cohort}/{name}', flush=True)
    with log.open('a', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), cohort, name], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f'Complete {cohort}/{name}', flush=True)


def measure_latency(filename):
    from bench.latency import session
    timed = INPUTS['new_models'] | INPUTS['timing_comparators']
    assert all(data.sha256_file(item['path']) == item['sha256'] for item in timed.values())
    timing = session({name: item['path'] for name, item in timed.items()}, MACHINE,
                     threads=(1, 4), warmup=200, runs=3000, rounds=3)
    results.write_json(OUT / filename, timing)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        score_one(*sys.argv[1:])
    elif sys.argv[1:] == ['--expanded-comparators']:
        os.chdir(ROOT)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(job, [('merged-800-test160', n) for n in INPUTS['expanded']['extra_models']]))
        while not (OUT/'latency.json').exists():
            time.sleep(10)
        measure_latency('latency_controlled.json')
        print('Expanded comparators and idle timing session complete.', flush=True)
    else:
        os.chdir(ROOT)
        jobs = [(INPUTS['benchmark'], name) for name in INPUTS['new_models']]
        jobs += [('merged-800-test160', name) for name in INPUTS['expanded']['models']]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(job, jobs))
        measure_latency('latency.json')
        print('All score and latency runs complete.', flush=True)
