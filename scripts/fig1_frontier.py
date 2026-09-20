#!/usr/bin/env python3
"""Figure 1 of Efficient Thinking VIII. Reads experience/results/*.json and writes docs/figs/fig1_frontier.{png,pdf}.
Run from the repo root:  python3 scripts/fig1_frontier.py
Every number on the figure comes from an artifact except the two matched-compute limbs of §7.6, quoted from the
paper's own table and labelled with the section."""
import json, math, re, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'experience/results/'
r5c = json.load(open(R + 'r5c.json')); r1 = json.load(open(R + 'r1_intervals.json'))
r3 = json.load(open(R + 'r3_complete.json')); r4 = json.load(open(R + 'r4b_disjoint.json'))
r7 = json.load(open(R + 'r7_disjoint_stats.json')); r6 = json.load(open(R + 'r6.json'))


def wilson(p, n, z=1.96):
    p = p / 100; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def ci(s):
    m = re.search(r'([+-][\d.]+)(?: points)?\s+(?:95% CI )?\[([+-][\d.]+), ([+-][\d.]+)\]', s)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={'width_ratios': [1.3, 1.25, 0.85]})
RED, GREY, INK = '#c0392b', '#7f7f7f', '#2c3e50'

# A: the frontier
a = ax[0]
for key, lab, col, mk, dy in [('base_curve', 'base: frozen model, ordinary search', GREY, 'o', -12),
                              ('head_curve', 'head: same model + experience prior\n(controller cost charged)', RED, 's', 7)]:
    pts = sorted(r5c[key].items(), key=lambda kv: kv[1]['all_in_tokens'])
    x = [v['all_in_tokens'] / 1000 for k, v in pts]; y = [v['success_pct'] for k, v in pts]; n = [v['n'] for k, v in pts]
    lo = [yy - wilson(yy, nn)[0] for yy, nn in zip(y, n)]; hi = [wilson(yy, nn)[1] - yy for yy, nn in zip(y, n)]
    a.errorbar(x, y, yerr=[lo, hi], fmt='-' + mk, color=col, capsize=3, lw=1.8, ms=6, label=lab, alpha=.95)
    for (k, v), xx, yy in zip(pts, x, y):
        a.annotate(f'budget {k}', (xx, yy), textcoords='offset points', xytext=(5, dy), fontsize=7.5, color=col)
b24 = r5c['base_curve']['24']; h6 = r5c['head_curve']['6']
a.annotate('', xy=(h6['all_in_tokens'] / 1000, 37.5), xytext=(b24['all_in_tokens'] / 1000, 37.5),
           arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2))
a.plot([h6['all_in_tokens'] / 1000] * 2, [h6['success_pct'], 37.5], ':', color=INK, lw=.8)
a.plot([b24['all_in_tokens'] / 1000] * 2, [b24['success_pct'], 37.5], ':', color=INK, lw=.8)
a.text((h6['all_in_tokens'] + b24['all_in_tokens']) / 2000, 38.2,
       'equal quality — 24.0% both, paired — at 2.32× less compute', ha='center', fontsize=8.5, color=INK)
a.set_xlabel('total inference compute per episode (thousand tokens, controller included)')
a.set_ylabel('externally verified task success (%)')
a.set_title('A · the quality–compute frontier, debugging (n = 300 per point, Wilson 95%)', fontsize=10, loc='left')
a.grid(alpha=.3); a.legend(fontsize=8, loc='lower right'); a.set_ylim(10, 41); a.set_xlim(3.5, 15)

# B: paired differences
rows = []
for p in r1['pairs']:
    if p['n'] == 300:
        d, lo, hi = ci(p['success'])
        rows.append((f"debugging, seed-47 head → seed-21 300{' (L-BFGS)' if 'L-BFGS' in p['pair'] else ''}", d, lo, hi))
for k, v in r3['results'].items():
    d, lo, hi = ci(v['success']); rows.append((f"debugging, seed-21 head → {k.replace(' (300)', ' 300')}", d, lo, hi))
rows.append(('second model family (Llama-3.1-8B), disjoint 300', r4['success_delta_points'], *r4['success_95CI']))
rows.append(("matched compute: base given the head's tokens (§7.6)", 9.3, 3.7, 15.0))
rows.append(('compute-neutral: head at budget 8 (§7.6)', 8.3, 3.0, 14.0))
rows.append(('second search structure, SQL repair, disjoint 300', r7['success_delta_points'], *r7['success_95CI']))
b = ax[1]; ys = list(range(len(rows)))[::-1]
for y, (lab, d, lo, hi) in zip(ys, rows):
    b.plot([lo, hi], [y, y], color=RED, lw=2); b.plot(d, y, 's', color=RED, ms=6)
    b.text(-0.8, y, lab, ha='right', va='center', fontsize=8)
b.axvline(0, color='k', lw=.8); b.set_yticks([]); b.set_xlim(-24, 24); b.set_xticks([0, 5, 10, 15, 20])
b.set_xlabel('head − base, problems solved per 100 (paired 95% CI)')
b.set_title('B · the paired effect, wherever it was measured', fontsize=10, loc='left'); b.grid(axis='x', alpha=.3)
for sp in ['left', 'top', 'right']: b.spines[sp].set_visible(False)

# C: same decision, same candidates
c = ax[2]; arms = r6['arms']
keys = ['base@12', '(a) self-rerank', '(b) majority vote (5 samples)', '(c) base + 35% search = R5-E budget 16', 'G@12']
labs = ['base', "model's own\npreference", 'majority\nvote ×5', '+35%\nsearch', 'head']
vals = [arms[k]['success_pct'] for k in keys]; cols = [GREY, '#95a5a6', '#95a5a6', '#95a5a6', RED]
c.bar(range(5), vals, color=cols, width=.7)
for i, v in enumerate(vals): c.text(i, v + .6, f'{v:.1f}', ha='center', fontsize=8.5)
c.set_xticks(range(5)); c.set_xticklabels(labs, fontsize=8); c.set_ylabel('success (%)'); c.set_ylim(0, 38)
c.set_title('C · at the same decision, same candidates', fontsize=10, loc='left'); c.grid(axis='y', alpha=.3)
for sp in ['top', 'right']: c.spines[sp].set_visible(False)

fig.suptitle('Experience moves the frontier of a frozen model — Efficient Thinking VIII, Figure 1', fontsize=12, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.95)); os.makedirs('docs/figs', exist_ok=True)
fig.savefig('docs/figs/fig1_frontier.png', dpi=170); fig.savefig('docs/figs/fig1_frontier.pdf')
print('rows in B:', [(r[0], r[1]) for r in rows])
