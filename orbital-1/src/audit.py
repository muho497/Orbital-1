#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ORBITAL-1 — recompute every checkable number in the paper from source.

Run from the folder holding results/ (or verify/). Prints a ground-truth table and
writes ground_truth.json. It deliberately knows nothing about what the paper claims:
compare afterwards, so the recomputation cannot be anchored by the value it is checking.

Percentages are rounded half-up at one decimal. That matters: the paper's top-1 was
wrong because a value already rounded to four decimals (0.6955) was rounded again to
69.6 %, when 708/1018 is 69.548 % and rounds to 69.5 %.
"""
import csv, json, math, collections, sys, os
from decimal import Decimal, ROUND_HALF_UP
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results"
def pth(f): return os.path.join(D, f)
def hu(x, p=1): return float(Decimal(repr(float(x))).quantize(Decimal('1.'+'0'*p), ROUND_HALF_UP))
def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return hu((c-h)*100), hu((c+h)*100)

G = {}
sp = list(csv.DictReader(open(pth('split_v1.csv'), encoding='utf-8')))
cnt = collections.Counter(r['split'] for r in sp)
G['split'] = {'train': cnt['train'], 'val': cnt['val'], 'test': cnt['test'], 'total': len(sp),
              'seeds_unique': len({r['seed'] for r in sp}) == len(sp)}
G['class_counts'] = dict(collections.Counter(r['label'] for r in sp))

S = json.load(open(pth('screening_summary.json')))
v1 = {k: v for k, v in S['output'].items() if isinstance(v, dict) and 'agree' in v}
tot = sum(v['n'] for v in v1.values())
G['screening'] = {'n': tot,
    'agree_pct': hu(sum(v['agree'] for v in v1.values())/tot*100),
    'hard_fail_pct': hu(sum(v['hard_fail'] for v in v1.values())/tot*100),
    'drift_pct': hu(sum(v['drift'] for v in v1.values())/tot*100),
    'per_class_agree_pct': {k: hu(v['agree_frac']*100) for k, v in v1.items()}}

sc = list(csv.DictReader(open(pth('screening_scores.csv'), encoding='utf-8')))
r1 = [r for r in sc if r['run'] == 'output']
PT = lambda r: r['drift'] == '1' and r['drift_mode'] == 'plain_texture'
G['screening']['plain_texture_pct'] = hu(sum(PT(r) for r in r1)/len(r1)*100)
G['screening']['plain_texture_by_camera'] = {
    c: hu(sum(PT(r) for r in r1 if r['camera'] == c)/sum(1 for r in r1 if r['camera'] == c)*100)
    for c in sorted({r['camera'] for r in r1})}

by = {}
for r in sc: by.setdefault(r['run'], {})[(r['label'], r['seed'])] = r
pair = sorted(set(by['output']) & set(by['output_v2']))
f = lambda run, fn: hu(sum(fn(by[run][k]) for k in pair)/len(pair)*100)
G['paired'] = {'n': len(pair),
    'agree': [f('output', lambda r: int(r['agree'])), f('output_v2', lambda r: int(r['agree']))],
    'hard_fail': [f('output', lambda r: int(r['hard_fail'])), f('output_v2', lambda r: int(r['hard_fail']))],
    'drift': [f('output', lambda r: int(r['drift'])), f('output_v2', lambda r: int(r['drift']))],
    'plain_texture': [f('output', lambda r: int(PT(r))), f('output_v2', lambda r: int(PT(r)))]}
a1 = {k for k in pair if by['output'][k]['agree'] == '1'}
a2 = {k for k in pair if by['output_v2'][k]['agree'] == '1'}
G['paired']['seed_overlap'] = {'only_v1': len(a1-a2), 'only_v2': len(a2-a1), 'both': len(a1 & a2)}

pr = list(csv.DictReader(open(pth('cls_test_predictions.csv'), encoding='utf-8')))
conf = np.array([float(r['conf']) for r in pr]); ok = np.array([int(r['correct']) for r in pr])
true = np.array([r['true'] for r in pr]); pred = np.array([r['pred'] for r in pr])
n, k = len(pr), int(ok.sum())
SIX = {'crater','perforation','spallation','debris_spray','microcrater_field','melt_splash'}
fam = lambda x: 'pit' if x in SIX else x
sub = [(a, b) for a, b in zip(true, pred) if a in SIX]
d = true != 'nominal'
G['classifier'] = {
    'n_test': n, 'correct': k, 'top1_pct': hu(k/n*100), 'top1_ci95': wilson(k, n),
    'family_top1_pct': hu(float(np.mean([fam(a) == fam(b) for a, b in zip(true, pred)]))*100),
    'pit_family_n': len(sub),
    'pit_within_family_pct': hu(sum(1 for a, b in sub if b in SIX)/len(sub)*100),
    'pit_exact_pct': hu(sum(1 for a, b in sub if a == b)/len(sub)*100),
    'damage_recall_pct': hu(float((pred[d] != 'nominal').mean())*100),
    'nominal_false_alarm_pct': hu(float((pred[~d] != 'nominal').mean())*100)}
ag = {r['file'].split('/')[-1]: (r['agree'] == '1') for r in sc if r['run'] == 'output'}
m = np.array([ag.get(r['file'], False) for r in pr])
G['classifier']['proxy_agreed'] = [int(m.sum()), hu(float(ok[m].mean())*100)]
G['classifier']['proxy_rejected'] = [int((~m).sum()), hu(float(ok[~m].mean())*100)]

def gate(t):
    s = conf >= t; td, pd_ = true[s], pred[s]; dd = td != 'nominal'
    return {'frames': int(s.sum()), 'coverage_pct': hu(float(s.mean())*100),
            'top1_pct': hu(float(ok[s].mean())*100),
            'damage_recall_pct': hu(float((pd_[dd] != 'nominal').mean())*100),
            'nominal_false_alarm_pct': hu(float((pd_[~dd] != 'nominal').mean())*100)}
G['gating'] = {str(t): gate(t) for t in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)}
edges = np.linspace(0, 1, 11); ece = 0.0; rel = []
for i in range(10):
    s = (conf > edges[i]) & (conf <= edges[i+1])
    if not s.sum(): continue
    ece += s.sum()/n*abs(ok[s].mean()-conf[s].mean())
    rel.append({'bin': [round(edges[i], 2), round(edges[i+1], 2)], 'n': int(s.sum()),
                'accuracy': round(float(ok[s].mean()), 4), 'mean_conf': round(float(conf[s].mean()), 4)})
o = np.argsort(-conf)
G['calibration'] = {'ece_10bin': round(float(ece), 4), 'reliability': rel,
    'aurc': round(float(np.mean(np.cumsum(1-ok[o])/np.arange(1, n+1))), 4),
    'aurc_random_baseline': round(float(1-ok.mean()), 4),
    'mean_conf_correct': round(float(conf[ok == 1].mean()), 4),
    'mean_conf_wrong': round(float(conf[ok == 0].mean()), 4)}

R = json.load(open(pth('cls_results.json')))
G['training'] = {kk: R[kk] for kk in ('train_minutes','epochs_run','best_epoch','val_top1','val_top5','n_train','n_val','n_test')}
try:
    rows = list(csv.DictReader(open(pth('results.csv'))))
    vl = [(int(r['epoch']), float(r['val/loss'])) for r in rows]
    G['training']['val_loss_min'] = min(vl, key=lambda x: x[1])
    G['training']['val_loss_final'] = vl[-1]
    G['training']['train_loss_first_last'] = [float(rows[0]['train/loss']), float(rows[-1]['train/loss'])]
except Exception as e:
    G['training']['curve_error'] = str(e)

U = json.load(open(pth('union_results.json')))
G['union'] = {kk: {'train_images': v['train_images'],
                   'v1_top1_pct': hu(v['v1_test']['top1']*100), 'v1_macro_f1': v['v1_test']['macro_f1'],
                   'v2_top1_pct': hu(v['v2_test']['top1']*100), 'v2_macro_f1': v['v2_test']['macro_f1'],
                   'v2_n': v['v2_test']['n']} for kk, v in U['models'].items()}

json.dump(G, open('ground_truth.json', 'w'), indent=1)
print(json.dumps(G, indent=1)[:2000])
print("\n... full output in ground_truth.json")
