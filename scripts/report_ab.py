#!/usr/bin/env python
"""Render the hard-vs-soft objective A/B (measured, not assumed) as HTML.

Reads runs/<ab>/summary.json (from compare_objectives.py) and shows, per width,
the two objectives side by side on the metrics that actually matter for strength
— bracketed-ladder Elo and regret — plus top-1 (noting it structurally favors
the imitation objective). A verdict line is computed from the measured deltas.
"""

from __future__ import annotations

import argparse
import html
import json
import os

HARD, SOFT = "#3D6FE3", "#E0913B"


def bar_panel(title, pairs, fmt, higher_better, width=340, height=190):
    """pairs: [(width_label, hard_val, soft_val)]. Grouped bars hard vs soft."""
    pad_l, pad_t, pad_b, pad_r = 44, 40, 34, 12
    vals = [v for _, h, s in pairs for v in (h, s)]
    vlo, vhi = min(vals + [0]), max(vals)
    if vhi == vlo:
        vhi = vlo + 1
    span = len(pairs)
    gw = (width - pad_l - pad_r) / span
    def Y(v):
        return pad_t + (1 - (v - vlo) / (vhi - vlo)) * (height - pad_t - pad_b)
    body = ""
    for i, (lab, h, s) in enumerate(pairs):
        x0 = pad_l + i * gw
        bw = gw * 0.30
        for j, (v, col, nm) in enumerate([(h, HARD, "hard"), (s, SOFT, "soft")]):
            bx = x0 + gw * 0.18 + j * (bw + 6)
            by, bh = Y(v), (height - pad_b) - Y(v)
            body += (f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                     f'height="{max(bh,0):.1f}" rx="3" fill="{col}"/>'
                     f'<text x="{bx+bw/2:.1f}" y="{by-4:.1f}" class="bl" '
                     f'text-anchor="middle">{fmt(v)}</text>')
        body += (f'<text x="{x0+gw/2:.1f}" y="{height-pad_b+16:.1f}" class="bx" '
                 f'text-anchor="middle">{html.escape(lab)}</text>')
    arrow = "higher = better" if higher_better else "lower = better"
    return f'''<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">
  <text x="{pad_l}" y="18" class="ct">{html.escape(title)}</text>
  <text x="{width-pad_r}" y="18" class="hint" text-anchor="end">{arrow}</text>
  {body}
</svg>'''


def build(summary):
    pts = summary["points"]
    widths = sorted({p["width"] for p in pts})
    by = {(p["width"], p["objective"]): p for p in pts}

    def pairs(key):
        out = []
        for w in widths:
            h, s = by.get((w, "hard")), by.get((w, "soft"))
            if h and s:
                out.append((f"W={w}", h[key], s[key]))
        return out

    elo = bar_panel("Bracketed Elo", pairs("elo"), lambda v: f"{v:.0f}", True)
    reg = bar_panel("Regret (winprob lost/move)", pairs("mean_regret"),
                    lambda v: f"{v:.3f}", False)
    top = bar_panel("Top-1 vs SF best move", pairs("val_top1"),
                    lambda v: f"{v:.3f}", True)

    # verdict from measured deltas. Quality (regret/blunder) and Elo are judged
    # SEPARATELY, and Elo deltas are compared against the measurement noise band
    # so we never over-claim from a difference that is within the error bars.
    lines = []
    reg_better = blu_better = elo_better = elo_worse = 0
    for w in widths:
        h, s = by.get((w, "hard")), by.get((w, "soft"))
        if not (h and s):
            continue
        de = s["elo"] - h["elo"]
        dr = s["mean_regret"] - h["mean_regret"]
        db = s["blunder_rate"] - h["blunder_rate"]
        margin = max(h.get("elo_margin", 0), s.get("elo_margin", 0))
        reg_better += int(dr < 0)
        blu_better += int(db < 0)
        elo_better += int(de > margin)
        elo_worse += int(de < -margin)
        lines.append(f"W={w}: Δregret {dr:+.3f}, Δblunder {db:+.3f}, "
                     f"Δelo {de:+.0f} (±{margin:.0f} noise), "
                     f"Δtop1 {s['val_top1']-h['val_top1']:+.3f}")
    nw = len(widths)
    quality = ("Soft lowers regret and blunder rate at every width — better move "
               "quality." if reg_better == nw and blu_better == nw else
               "Soft improves move quality at some widths (see deltas).")
    if elo_better == nw and nw:
        elov = "And it raises Elo beyond the noise band — a real strength win."
    elif elo_worse >= 1:
        elov = ("But Elo did not improve (some deltas below the noise band): the "
                "move-quality gain has NOT yet converted to measured Elo at this "
                "data scale / game count.")
    else:
        elov = ("Elo deltas are within the measurement noise band — inconclusive "
                "on Elo; needs more data and more games per rung to confirm.")
    verdict = quality + " " + elov

    rows = "".join(
        f"<tr><td>W={p['width']}</td><td>{p['objective']}</td>"
        f"<td>{p['params']/1e6:.2f}M</td><td>{p['val_top1']:.3f}</td>"
        f"<td>{p['mean_regret']:.3f}</td><td>{p['blunder_rate']:.3f}</td>"
        f"<td>{p['elo']:.0f} ± {p['elo_margin']:.0f}</td></tr>"
        for p in sorted(pts, key=lambda p: (p['width'], p['objective'])))

    dl = "<br>".join(html.escape(x) for x in lines)
    return f'''<title>Objective A/B — hard vs soft</title>
<style>
  :root {{ --paper:#f6f7fa; --ink:#14171f; --muted:#5b6472; --card:#fff; --line:#e4e7ee; }}
  @media (prefers-color-scheme:dark) {{ :root {{ --paper:#0e1116; --ink:#e7eaf0;
    --muted:#8b93a3; --card:#161a22; --line:#242a35; }} }}
  :root[data-theme="dark"] {{ --paper:#0e1116; --ink:#e7eaf0; --muted:#8b93a3;
    --card:#161a22; --line:#242a35; }}
  :root[data-theme="light"] {{ --paper:#f6f7fa; --ink:#14171f; --muted:#5b6472;
    --card:#fff; --line:#e4e7ee; }}
  body {{ margin:0; padding:32px 28px; background:var(--paper); color:var(--ink);
    font:14px/1.55 -apple-system,system-ui,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-0.02em; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:8px; }}
  .verdict {{ background:var(--card); border:1px solid var(--line); border-left:3px solid {SOFT};
    border-radius:10px; padding:14px 16px; margin:16px 0; font-size:14px; }}
  .verdict b {{ display:block; margin-bottom:6px; }}
  .deltas {{ color:var(--muted); font-family:ui-monospace,monospace; font-size:12px;
    margin-top:8px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:14px; margin:16px 0; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; }}
  .chart {{ width:100%; height:auto; }}
  .ct {{ font-size:12px; font-weight:600; fill:var(--ink); }}
  .hint {{ font-size:9.5px; fill:var(--muted); }}
  .bl {{ font-size:9.5px; fill:var(--ink); font-family:ui-monospace,monospace; }}
  .bx {{ font-size:10px; fill:var(--muted); font-family:ui-monospace,monospace; }}
  .leg {{ font-size:12px; color:var(--muted); }}
  .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; }}
  table {{ border-collapse:collapse; width:100%; font-family:ui-monospace,monospace;
    font-size:12.5px; background:var(--card); border-radius:10px; overflow:hidden;
    font-variant-numeric:tabular-nums; margin-top:8px; }}
  th,td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
  thead th {{ color:var(--muted); font-size:10.5px; text-transform:uppercase; }}
</style>
<div class="wrap">
  <h1>Training objective A/B — best-move vs advantage-weighted</h1>
  <div class="sub">{summary['n_positions']:,} identical soft-labeled positions · depth-{summary['depth']}
    · same architecture per pair · only the loss differs · τ={summary['tau']}</div>
  <p class="leg"><span class="sw" style="background:{HARD}"></span> hard = single best-move (imitation)
    &nbsp;&nbsp; <span class="sw" style="background:{SOFT}"></span> soft = advantage-weighted distribution</p>
  <div class="verdict"><b>Verdict</b>{html.escape(verdict)}
    <div class="deltas">{dl}</div></div>
  <div class="grid">
    <div class="card">{elo}</div>
    <div class="card">{reg}</div>
    <div class="card">{top}</div>
  </div>
  <table><thead><tr><th>width</th><th>objective</th><th>params</th><th>top-1</th>
    <th>regret</th><th>blunder</th><th>Elo</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True, help="dir with summary.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.ab, "summary.json")) as f:
        summary = json.load(f)
    out = args.out or os.path.join(args.ab, "report.html")
    with open(out, "w") as f:
        f.write(build(summary))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
