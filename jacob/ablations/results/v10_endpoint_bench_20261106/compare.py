"""Report frozen benchmark outputs, including the 140 players outside the historical 20."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT))
from bench import data, stats
from bench.scoring import summarise

REFIT = 'k8_800_refit_s44'
ORIGINAL = 'k8_800_original_s44'
GAIN = 'k8_800_gain105_s44'
OLDER = 'v9_k8_combined_ga2'
SMALL, LARGE, EXTRA = 'group-a-v2-test20', 'merged-800-test160', 'additional-140'
LABEL = {REFIT:'800-user k8, endpoint refit', ORIGINAL:'800-user k8, original', GAIN:'800-user k8, previous +5% edit',
         OLDER:'100-user k8', 'v9_k5_combined_ga2':'100-user k5', 'v7_sep_base3_addons_angular_ga2':'v7 sep + angular',
         'v6_tcnn_ga2':'v6 TCNN', 'v5_tcnn_ga2':'v5 TCNN', 'nic_ga2_gru_s45':'Nic GRU, seed 45',
         'nic_ga2_gadst_s42':'Nic GADS-T, seed 42', 'hold_last':'Hold last', 'damped_velocity':'Damped velocity',
         'constant_velocity_1step':'Constant velocity'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False)+'\n', encoding='utf-8')


def records(folder, definition):
    rows = {}
    for file in sorted(folder.glob('*.json')):
        row = read(file)
        if row['contamination']['status'] != 'clean':
            continue
        assert row['bench']['cohort_fingerprint'] == definition['cohort_fingerprint']
        assert row['accuracy']['windows'] == definition['windows']
        assert set(row['accuracy']['per_player']) == set(definition['players'])
        name = row['model']['name']
        if name in rows:
            assert row['model']['onnx_sha256'] == rows[name]['model']['onnx_sha256']
            if row['machine']['label'] != 'jacob-amd':
                continue
        rows[name] = row
    return rows


def subset_scores(row, definition, users):
    """Preserve full-precision player sums. Subset quantiles use archived float32 endpoints."""
    segments = definition['segments']
    wanted = np.array([key.split('/', 1)[0] in users for key, _ in segments])
    count = sum(n for keep, (_, n) in zip(wanted, segments) if keep)
    result = {'model':row['model'],'efficiency':row['efficiency']}
    for protocol in ('accuracy','accuracy_cold_start'):
        original = row[protocol]
        if original is None:
            result[protocol] = None
            continue
        path = ROOT / original['per_window_file']['path']
        assert data.sha256_file(path) == original['per_window_file']['sha256']
        endpoint = np.load(path, mmap_mode='r', allow_pickle=False)
        segment_ids = np.load(path.parent/'segment_id.npy', mmap_mode='r', allow_pickle=False)
        assert len(endpoint) == len(segment_ids) == definition['windows']
        np.testing.assert_array_equal(np.bincount(segment_ids), [n for _,n in segments])
        values = endpoint[wanted[segment_ids]]
        assert len(values) == count
        players = {key:{'windows':v['windows'],'sum_deg_by_horizon':np.array(v['sum_deg_by_horizon'])}
                   for key,v in original['per_player'].items() if key in users}
        result[protocol] = summarise(players, values.astype(np.float64), original['protocol'])
        assert result[protocol]['windows'] == count
    return result


def make_report():
    inputs = read(OUT/'benchmark_inputs.json')
    small = data.load_definition(SMALL)
    large = data.load_definition(ROOT / inputs['expanded']['definition'])
    fresh = records(OUT/'scores'/SMALL, small)
    full = records(OUT/'scores'/LARGE, large)
    history = records(ROOT/'bench/results'/SMALL, small)
    assert set(fresh) == {REFIT,ORIGINAL,GAIN} and set(full) == set(inputs['expanded']['all_models'])
    for name, row in (fresh | full).items():
        if row['model']['kind'] == 'baseline':
            continue
        source = (inputs['timing_comparators'] | inputs['new_models'])[name]
        assert row['model']['onnx_sha256'] == data.sha256_file(ROOT/source['path']) == source['sha256']
    historical20 = history | fresh
    extra_users = set(large['players']) - set(small['players'])
    assert len(extra_users) == 140
    extra = {name:subset_scores(row,large,extra_users) for name,row in full.items()}
    # The historical players nested in the 160-user HDF5 must reproduce their own benchmark.
    agreement = {}
    for name,row in full.items():
        subset = subset_scores(row,large,set(small['players']))
        diffs = []
        for key in ('accuracy','accuracy_cold_start'):
            if subset[key] is None:
                assert historical20[name][key] is None
                continue
            for user,a in subset[key]['per_player'].items():
                b = historical20[name][key]['per_player'][user]
                assert a['windows'] == b['windows']
                diffs.append(max(abs(np.array(a['sum_deg_by_horizon'])-b['sum_deg_by_horizon'])) / a['windows'])
        agreement[name] = float(max(diffs))
        assert agreement[name] < 1e-5, (name,agreement[name])
        assert extra[name]['accuracy']['windows'] == 5428916
    write(OUT/'additional_140_scores.json', {'parent_fingerprint':large['cohort_fingerprint'],
          'users':sorted(extra_users),'windows':5428916,'quantiles':'archived float32 endpoint values', 'scores':extra})
    write(OUT/'nested_cohort_check.json',{'max_player_horizon_difference_deg':agreement,'tolerance_deg':1e-5})
    # Compare independent window/checkpoint evaluations already present in the project.
    previous_dir = ROOT/'jacob/ablations/analyses/generalisation/scores'
    reference_checks = {}
    for name, filename, predictor in ((ORIGINAL,'k8_800_s44','model'),(OLDER,'k8_100_s42','model'),
                                      ('hold_last','baselines','hold_last'),('damped_velocity','baselines','damped_velocity'),
                                      ('constant_velocity_1step','baselines','constant_velocity')):
        ref_path = previous_dir/(filename+'.json'); ref = read(ref_path)
        assert ref['data_sha256'] == inputs['expanded']['data_sha256'] and ref['strides']['test'] == 1
        totals = {}
        for index, seg in enumerate(ref['segments']):
            if seg['split'] != 'test':
                continue
            row = totals.setdefault(seg['user'],np.zeros(3))
            row += [seg['windows_scored'],ref['sums'][predictor]['step10'][index],ref['sums'][predictor]['step20'][index]]
        current = full[name]['accuracy']['per_player']
        assert set(totals) == set(current)
        discrepancies = []
        for user,(n,s10,s20) in totals.items():
            assert n == current[user]['windows']
            discrepancies += [abs(s10-current[user]['sum_deg_by_horizon'][9])/n,
                              abs(s20-current[user]['sum_deg_by_horizon'][19])/n]
        reference_checks[name] = {'reference':str(ref_path.relative_to(ROOT)),'reference_sha256':data.sha256_file(ref_path),
                                  'max_player_difference_deg':float(max(discrepancies))}
        assert max(discrepancies) < 1e-4, (name,max(discrepancies))
    write(OUT/'existing_scores_check.json',reference_checks)
    cohorts = {SMALL:historical20, EXTRA:extra, LARGE:full}
    paired = {}
    for cohort, rows in cohorts.items():
        paired[cohort] = {}
        for name,row in rows.items():
            if name == REFIT:
                continue
            paired[cohort][name] = {'from_start':stats.paired(rows[REFIT]['accuracy'],row['accuracy'])}
            if row.get('accuracy_cold_start'):
                paired[cohort][name]['cold_start'] = stats.paired(rows[REFIT]['accuracy_cold_start'],row['accuracy_cold_start'])
    write(OUT/'paired_comparisons.json', paired)
    timing = read(OUT/inputs['expanded']['timing_session_for_report'])
    for name, t in timing['models'].items():
        assert t['onnx_sha256'] == historical20[name]['model']['onnx_sha256']
    with (OUT/'all_model_scores.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['cohort','model','protocol','players','windows','mean_200ms_deg','mean_100ms_deg','median_deg','p95_deg','p99_deg','equal_player_deg','params','MACs','median_1thread_us','median_4thread_us'])
        for cohort,rows in cohorts.items():
            for name,row in sorted(rows.items(),key=lambda x:x[1]['accuracy']['mean_deg_step20']):
                for key in ('accuracy','accuracy_cold_start'):
                    a=row.get(key)
                    if not a:
                        continue
                    eff=row.get('efficiency') or {}; t=timing['models'].get(name,{}).get('threads',{})
                    writer.writerow([cohort,name,a['protocol'],len(a['per_player']),a['windows'],a['mean_deg_step20'],a['mean_deg_step10'],a['median_deg_step20'],a['p95_deg_step20'],a['p99_deg_step20'],a['equal_player_mean_deg_step20'],row['model']['declared_trainable_parameters'],eff.get('macs_per_call',{}).get('total'),*(1000*t[n]['median_ms'] if n in t else '' for n in ('1','4'))])

    big = paired[LARGE][ORIGINAL]['from_start']
    refit, original, gain = (full[name]['accuracy']['mean_deg_step20'] for name in (REFIT,ORIGINAL,GAIN))
    table=['# Frozen k8 benchmark comparison','',
           f"On all 160 test players the endpoint refit scored **{refit:.5f}°**, the original 800-user model scored **{original:.5f}°**, and the previous +5% edit scored **{gain:.5f}°**. The refit improved {big['players_better']}/160 players versus the original. The paired 95% interval for fitted minus original error was [{big['ci95'][0]:+.5f}°, {big['ci95'][1]:+.5f}°].",'',
           'The headline score is window-weighted mean geodesic error 200 ms ahead, streaming from each segment’s start. Lower is better. All models use their frozen weights. The 800-user refit and original have the same architecture and training cohort. The previous +5% edit uses the same source checkpoint.','',
           '## Direct comparison of the 800-user models','',
           '| Cohort | Windows | Original (°) | +5% edit (°) | Refit (°) | Refit − original (°), 95% interval | Players improved |',
           '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
    short={SMALL:'Historical 20',EXTRA:'Additional 140',LARGE:'Full 160'}
    for cohort,rows in cohorts.items():
        p=paired[cohort][ORIGINAL]['from_start'];lo,hi=p['ci95']; a=rows[REFIT]['accuracy']
        table.append(f"| {short[cohort]} | {a['windows']:,} | {rows[ORIGINAL]['accuracy']['mean_deg_step20']:.5f} | {rows[GAIN]['accuracy']['mean_deg_step20']:.5f} | {a['mean_deg_step20']:.5f} | {p['mean_diff_deg']:+.5f} [{lo:+.5f}, {hi:+.5f}] | {p['players_better']}/{p['players']} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.4,3.8),layout='constrained')
    for y,cohort in enumerate((SMALL,EXTRA,LARGE)):
        p=paired[cohort][ORIGINAL]['from_start']; d=p['mean_diff_deg']; lo,hi=p['ci95']
        ax.plot((lo,hi),(y,y),color='#396da5',linewidth=2)
        ax.plot((lo,hi),(y,y),'|',color='#396da5',markersize=10)
        ax.plot(d,y,'o',color='#184e88')
    ax.axvline(0,color='#943c37',linestyle='--',linewidth=1)
    ax.set_yticks(range(3),[short[c] for c in (SMALL,EXTRA,LARGE)])
    ax.set_ylim(2.45,-.45);ax.set_title('200 ms error change from the endpoint refit',loc='left')
    ax.set_xlabel('Fitted minus original error (degrees). Negative values favour the refit.\nBars show paired 95% player-bootstrap intervals.')
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'paired_test_effect.png',dpi=180);fig.savefig(OUT/'paired_test_effect.svg');plt.close(fig)
    table.extend(['','![Paired endpoint differences](paired_test_effect.png)','',
                  'The full 160 include both smaller groups. The additional 140 provide a separate look at the players outside the team leaderboard. The project had already evaluated earlier k8 models on this cohort. Intervals use 10,000 paired player-bootstrap draws with seed 0. The additional-140 quantiles use the archived float32 endpoint errors. The cohort means and paired comparisons use float64 per-player sums.','',
                  '## Previous iterations on the standard team benchmark','',
                  '| Model | 200 ms (°) | 100 ms (°) | Cold-start 200 ms (°) | Endpoint p95 (°) | Parameters | MACs/call | 1-thread median (µs) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for name in LABEL:
        row=historical20[name]; a=row['accuracy']; cold=row.get('accuracy_cold_start'); eff=row.get('efficiency');t=timing['models'].get(name)
        table.append(f"| {LABEL[name]} | {a['mean_deg_step20']:.5f} | {a['mean_deg_step10']:.5f} | {format(cold['mean_deg_step20'],'.5f') if cold else '—'} | {a['p95_deg_step20']:.4f} | {row['model']['declared_trainable_parameters']:,} | {format(eff['macs_per_call']['total'],',') if eff else '—'} | {format(1000*t['threads']['1']['median_ms'],'.1f') if t else '—'} |")
    table.extend(['','The earlier learned models in this table trained on 100 users. Differences against those models also include changes in training data and recipe. The 800-user original isolates the incremental contribution of the refit. The historical accuracy records reuse the same fingerprint and windows. Latencies were measured together in a new local CPU session.','',
                  '## Full 160-user benchmark','',
                  '| Model | 200 ms (°) | 100 ms (°) | Cold-start 200 ms (°) | Median / p95 / p99 (°) | Equal-player mean (°) |',
                  '| --- | ---: | ---: | ---: | --- | ---: |'])
    for name in sorted(full, key=lambda n:full[n]['accuracy']['mean_deg_step20']):
        a=full[name]['accuracy']; c=full[name]['accuracy_cold_start']
        cold_score = f"{c['mean_deg_step20']:.5f}" if c else '—'
        table.append(f"| {LABEL[name]} | {a['mean_deg_step20']:.5f} | {a['mean_deg_step10']:.5f} | {cold_score} | {a['median_deg_step20']:.3f} / {a['p95_deg_step20']:.3f} / {a['p99_deg_step20']:.3f} | {a['equal_player_mean_deg_step20']:.5f} |")
    table.extend(['','| Refit against comparator on 160 players | Difference (°) | 95% interval (°) | Players improved |',
                  '| --- | ---: | --- | ---: |'])
    for name in (ORIGINAL,GAIN,OLDER):
        p=paired[LARGE][name]['from_start']; lo,hi=p['ci95']
        table.append(f"| {LABEL[name]} | {p['mean_diff_deg']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | {p['players_better']}/160 |")
    eff=full[REFIT]['efficiency']; path=ROOT / inputs['new_models'][REFIT]['path']; test=eff['output_checks']
    assert test['shape_ok'] and test['finite'] and test['deterministic'] and not eff['operators']['issues']
    table.extend(['','## Verification and evidence','',
                  f"The frozen refit ONNX is {path.stat().st_size:,} bytes and has 7,555 declared parameters, {eff['initializer_elements']:,} stored initializer elements and {eff['macs_per_call']['total']:,} multiply-accumulates per call. It passed output-shape, finiteness, repeatability and operator checks. The maximum output difference between one and four threads was {test['max_abs_diff_1_vs_4_threads']:.2g}.",'',
                  f"The historical benchmark fingerprint is `{small['cohort_fingerprint'][:16]}`. The expanded benchmark fingerprint is `{large['cohort_fingerprint'][:16]}`. The maximum per-player, per-horizon discrepancy when reproducing the historical 20 from the merged file was {max(agreement.values()):.2g}°. All newly scored models passed the train/validation overlap check.",'',
                  f"The single-session timing comparison used {timing['machine']['cpu']}, ONNX Runtime {timing['machine']['onnxruntime'] if 'onnxruntime' in timing['machine'] else timing['machine'].get('ort','')}, 1 and 4 CPU threads, and {timing['protocol']['timed_calls']:,} timed calls per configuration. The organisers assess latency separately on XR hardware.",'',
                  '- [All accuracy and efficiency rows, including historical variants](all_model_scores.csv)',
                  '- [Paired comparisons across players](paired_comparisons.json)',
                  '- [160-user benchmark definition](definitions/merged-800-test160.json) and [additional 140 scores](additional_140_scores.json)',
                  '- [Model and data hashes](benchmark_inputs.json), [nested-cohort verification](nested_cohort_check.json), and [latency session](latency_controlled.json)',
                  '- [Standard benchmark score records](scores/group-a-v2-test20) and [expanded score records](scores/merged-800-test160)',
                  '- [Frozen refit ONNX](frozen_models/k8_800_refit_s44.onnx), [original ONNX](frozen_models/k8_800_original_s44.onnx), and [previous edit ONNX](frozen_models/k8_800_gain105_s44.onnx)',
                  '- [Cross-check against earlier 160-user evaluations](existing_scores_check.json) and [previous edit export verification](prior_gain_export_receipt.json)',
                  '- `run_benchmark.py` runs the team scorer with separate comparison outputs. Its `--expanded-comparators` pass scores the historical models and baselines and records the final timing session. `compare.py` verifies the scores and builds this report.'])
    (OUT/'README.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    return paired


if __name__ == '__main__':
    comparison = make_report()
    for cohort in (SMALL,EXTRA,LARGE):
        print(cohort,comparison[cohort][ORIGINAL]['from_start'])
