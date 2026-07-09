#!/usr/bin/env python
"""Render a sweep's summary.json into the efficiency / scaling report.

Instrument-panel dashboard, fully self-contained (inline SVG, no external
assets, theme-aware). Foregrounds *rating per coefficient*: quality-vs-params
Pareto frontier, efficiency (quality per Mparam), illegal-rate decay, and the
data-scaling curve — with points colored by which lever (depth / width /
encoding / train-size) they vary.

Example:
  PYTHONPATH=. python scripts/report.py --sweep runs/eff1 --out runs/eff1/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os

# axis -> accent color (encodes the lever, not decoration)
AXIS_COLOR = {"depth": "#3D6FE3", "width": "#E0913B",
              "encoding": "#2FA98C", "train_size": "#9B6DE3", "grid": "#5b6472"}
AXIS_LABEL = {"depth": "depth N", "width": "width W",
              "encoding": "encoding", "train_size": "train size",
              "grid": "grid (width × data)"}


# examples-per-parameter regimes (Chinchilla ~20x / classic ~10x heuristics).
# This is how we make "params plateau because data is too low" legible.
REGIMES = [
    (1.0,   "starved",  "#d9534f", "params ≫ data — capacity wasted"),
    (10.0,  "limited",  "#e0913b", "data-limited — more data would help"),
    (20.0,  "balanced", "#3D6FE3", "near compute-optimal"),
    (1e18,  "data-rich","#2FA98C", "data-rich — capacity is the limit"),
]


def regime_of(ex_per_param):
    for thr, name, color, desc in REGIMES:
        if ex_per_param < thr:
            return name, color, desc
    return REGIMES[-1][1:]


def _sc(v, lo, hi, a, b):
    return (a + b) / 2 if hi == lo else a + (v - lo) / (hi - lo) * (b - a)


def _fmt_count(n):
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}k"
    return f"{n:.0f}"


def heatmap(rows, cols, cell, title, rowlab, collab, cellfmt, subfmt,
            width=520, height=340):
    """rows/cols: axis values. cell(r,c)->(value in [0,1] or None). Sequential fill."""
    pad_l, pad_t, pad_b, pad_r = 88, 44, 30, 14
    if not rows or not cols:
        return f"<p class='empty'>No grid for {html.escape(title)}</p>"
    cw = (width - pad_l - pad_r) / len(cols)
    ch = (height - pad_t - pad_b) / len(rows)
    vals = [cell(r, c)[0] for r in rows for c in cols if cell(r, c)[0] is not None]
    vlo, vhi = (min(vals), max(vals)) if vals else (0, 1)
    body = ""
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            v, sub = cell(r, c)
            x, y = pad_l + ci * cw, pad_t + ri * ch
            if v is None:
                fill = "var(--grid)"
                txt = ""
            else:
                t = 0.10 + 0.52 * (_sc(v, vlo, vhi, 0, 1))
                fill = f"rgba(61,111,227,{t:.2f})"
                txt = (f'<text x="{x+cw/2:.0f}" y="{y+ch/2-2:.0f}" class="hm" '
                       f'text-anchor="middle">{cellfmt(v)}</text>'
                       f'<text x="{x+cw/2:.0f}" y="{y+ch/2+11:.0f}" class="hmsub" '
                       f'text-anchor="middle">{subfmt(sub)}</text>')
            body += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw-2:.1f}" '
                     f'height="{ch-2:.1f}" rx="4" fill="{fill}"/>{txt}')
        body += (f'<text x="{pad_l-8}" y="{pad_t+ri*ch+ch/2+3:.0f}" class="tick" '
                 f'text-anchor="end">{rowlab(r)}</text>')
    for ci, c in enumerate(cols):
        body += (f'<text x="{pad_l+ci*cw+cw/2:.0f}" y="{height-pad_b+18:.0f}" '
                 f'class="tick" text-anchor="middle">{collab(c)}</text>')
    return f'''<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">
  <text x="{pad_l}" y="20" class="ctitle">{html.escape(title)}</text>{body}
  <text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-6}" class="axl" text-anchor="middle">{html.escape(collab.__doc__ or "")}</text>
</svg>'''


def scatter(series, title, xlabel, ylabel, logx=False, yfmt=None,
            xfmt=None, pareto=False, width=520, height=340):
    """series: [{name,color,points:[(x,y,label)]}]. Returns inline <svg>."""
    yfmt = yfmt or (lambda v: f"{v:.0f}")
    xfmt = xfmt or (lambda v: f"{v:g}")
    pad_l, pad_b, pad_t, pad_r = 62, 46, 40, 16
    allpts = [(x, y) for s in series for (x, y, _) in s["points"]]
    if not allpts:
        return f"<p class='empty'>No data for {html.escape(title)}</p>"
    xs = [(math.log10(x) if logx else x) for x, _ in allpts]
    ys = [y for _, y in allpts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    if ylo == yhi:
        ylo -= 1; yhi += 1
    yp = (yhi - ylo) * 0.12
    ylo -= yp; yhi += yp
    xp = (xhi - xlo) * 0.06 or 0.5
    xlo -= xp; xhi += xp

    def X(v):
        return _sc(math.log10(v) if logx else v, xlo, xhi, pad_l, width - pad_r)

    def Y(v):
        return _sc(v, ylo, yhi, height - pad_b, pad_t)

    # gridlines + y ticks
    grid = ""
    for i in range(5):
        yv = ylo + (yhi - ylo) * i / 4
        yy = Y(yv)
        grid += (f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{pad_l-8}" y="{yy+3:.1f}" class="tick" text-anchor="end">'
                 f'{yfmt(yv)}</text>')

    body = ""
    # Pareto frontier (upper-left is best: high y, low x) over ALL points
    if pareto:
        pts = sorted(allpts, key=lambda p: (p[0]))
        front, best = [], -1e30
        for x, y in pts:
            if y > best:
                front.append((x, y)); best = y
        if len(front) > 1:
            poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in front)
            body += (f'<polyline points="{poly}" fill="none" '
                     f'stroke="var(--muted)" stroke-width="1.5" '
                     f'stroke-dasharray="4 3" opacity="0.7"/>')

    xticklabels = {}
    for s in series:
        pts = sorted(s["points"], key=lambda t: t[0])
        if len(pts) > 1:
            poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _ in pts)
            body += (f'<polyline points="{poly}" fill="none" stroke="{s["color"]}" '
                     f'stroke-width="2" opacity="0.85"/>')
        for x, y, lab in pts:
            body += (f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" '
                     f'fill="{s["color"]}"/>')
            if lab:
                body += (f'<text x="{X(x):.1f}" y="{Y(y)-9:.1f}" class="dot" '
                         f'text-anchor="middle">{html.escape(lab)}</text>')
            xticklabels[X(x)] = xfmt(x)

    xticks = "".join(
        f'<text x="{xx:.1f}" y="{height-pad_b+16:.1f}" class="tick" '
        f'text-anchor="middle">{lab}</text>'
        for xx, lab in sorted(xticklabels.items()))

    legend = ""
    lx = pad_l
    for s in series:
        legend += (f'<circle cx="{lx+4}" cy="{pad_t-14}" r="4" fill="{s["color"]}"/>'
                   f'<text x="{lx+12}" y="{pad_t-10}" class="leg">{html.escape(s["name"])}</text>')
        lx += 12 + 8 * len(s["name"]) + 20

    return f'''<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">
  <text x="{pad_l}" y="20" class="ctitle">{html.escape(title)}</text>
  {grid}{legend}{body}{xticks}
  <text x="{(pad_l+width-pad_r)/2:.0f}" y="{height-6}" class="axl" text-anchor="middle">{html.escape(xlabel)}</text>
  <text x="16" y="{height/2:.0f}" class="axl" text-anchor="middle" transform="rotate(-90 16 {height/2:.0f})">{html.escape(ylabel)}</text>
</svg>'''


def _series_by_axis(points, axis, xkey, ykey, labelfn):
    pts = [p for p in points if axis in p.get("axes", [])]
    return {"name": AXIS_LABEL[axis], "color": AXIS_COLOR[axis],
            "points": [(p[xkey], p[ykey], labelfn(p)) for p in pts]}


def build_html(summary):
    pts = summary["points"]
    axes_present = sorted({a for p in pts for a in p.get("axes", [])})

    # examples-per-parameter regime for every config (the "restriction" lens)
    for p in pts:
        p["ex_per_param"] = p["n_train"] / p["params"]
        p["regime"], p["regime_color"], p["regime_desc"] = regime_of(p["ex_per_param"])

    # best-by-efficiency and best-by-quality callouts
    best_q = max(pts, key=lambda p: p["val_top1"])
    best_eff = max(pts, key=lambda p: p["top1_per_mparam"])

    def dlabel(p):
        return f"d{p['depth']}·w{p['width']}"

    # --- charts ---
    # 1. Quality vs params (efficiency frontier)
    q_series = [_series_by_axis(pts, a, "params", "val_top1", dlabel)
                for a in axes_present]
    q_series = [s for s in q_series if s["points"]]
    c_quality = scatter(q_series, "Move accuracy vs parameters (Pareto frontier)",
                        "total parameters (log)", "val top-1", logx=True,
                        yfmt=lambda v: f"{v:.2f}",
                        xfmt=lambda v: f"{v/1e6:.2g}M", pareto=True)

    # 2. Efficiency: top-1 per Mparam vs params (diminishing returns)
    e_series = [_series_by_axis(pts, a, "params", "top1_per_mparam", dlabel)
                for a in axes_present]
    e_series = [s for s in e_series if s["points"]]
    c_eff = scatter(e_series, "Efficiency: accuracy per million params",
                   "total parameters (log)", "top-1 / Mparam", logx=True,
                   yfmt=lambda v: f"{v:.2f}", xfmt=lambda v: f"{v/1e6:.2g}M")

    # 3. Illegal-move rate vs params
    i_series = [_series_by_axis(pts, a, "params", "illegal_rate", dlabel)
                for a in axes_present]
    i_series = [s for s in i_series if s["points"]]
    c_illegal = scatter(i_series, "Illegal-move rate vs parameters",
                       "total parameters (log)", "illegal rate", logx=True,
                       yfmt=lambda v: f"{v:.2f}", xfmt=lambda v: f"{v/1e6:.2g}M")

    # 4. Data-scaling curve (only if train-size axis was swept)
    charts = [c_quality, c_eff, c_illegal]
    if any("train_size" in p.get("axes", []) for p in pts):
        ds_pts = [p for p in pts if "train_size" in p.get("axes", [])]
        ds_series = [{"name": "train size", "color": AXIS_COLOR["train_size"],
                      "points": [(p["n_train"], p["val_top1"],
                                  f"{int(p['train_fraction']*100)}%") for p in ds_pts]}]
        charts.append(scatter(ds_series, "Data-scaling: accuracy vs training examples",
                             "training positions", "val top-1",
                             yfmt=lambda v: f"{v:.2f}",
                             xfmt=lambda v: f"{v/1000:.0f}k"))

    # --- Data x Parameters interaction (the plateau story) ---------------------
    # Use grid points if present, else any points that vary both width & data.
    grid_pts = [p for p in pts if "grid" in p.get("axes", [])] or pts
    widths = sorted({p["width"] for p in grid_pts})
    fracs = sorted({p["train_fraction"] for p in grid_pts})
    interaction = ""
    if len(widths) >= 2 and len(fracs) >= 2:
        # plateau chart: accuracy vs params, ONE LINE PER DATA SIZE.
        # low-data lines flatten (extra params wasted); high-data lines keep rising.
        DATA_COLORS = ["#c44", "#e0913b", "#3D6FE3", "#2FA98C", "#9B6DE3", "#12141a"]
        plateau_series = []
        for i, f in enumerate(fracs):
            fp = sorted([p for p in grid_pts if p["train_fraction"] == f],
                        key=lambda p: p["params"])
            n = fp[0]["n_train"] if fp else 0
            plateau_series.append({"name": f"{_fmt_count(n)} pos",
                                   "color": DATA_COLORS[i % len(DATA_COLORS)],
                                   "points": [(p["params"], p["val_top1"], "") for p in fp]})
        c_plateau = scatter(plateau_series,
                            "Where parameters plateau (one line per data size)",
                            "total parameters (log)", "val top-1", logx=True,
                            yfmt=lambda v: f"{v:.2f}", xfmt=lambda v: f"{v/1e6:.2g}M")

        # saturation chart: accuracy vs data, one line per width.
        WIDTH_COLORS = ["#9aa", "#3D6FE3", "#E0913B", "#2FA98C", "#9B6DE3"]
        sat_series = []
        for i, w in enumerate(widths):
            wp = sorted([p for p in grid_pts if p["width"] == w],
                        key=lambda p: p["n_train"])
            sat_series.append({"name": f"W={w}",
                               "color": WIDTH_COLORS[i % len(WIDTH_COLORS)],
                               "points": [(p["n_train"], p["val_top1"], "") for p in wp]})
        c_sat = scatter(sat_series,
                       "Data-scaling per width (each width saturates)",
                       "training positions (log)", "val top-1", logx=True,
                       yfmt=lambda v: f"{v:.2f}", xfmt=lambda v: _fmt_count(v))

        # regime heatmap: width x data, colored by top-1, annotated with ex/param.
        cellmap = {(p["width"], p["train_fraction"]): p for p in grid_pts}
        def cell(w, f):
            p = cellmap.get((w, f))
            return (p["val_top1"], p["ex_per_param"]) if p else (None, None)
        collab = lambda f: _fmt_count(
            next((p["n_train"] for p in grid_pts if p["train_fraction"] == f), 0))
        collab.__doc__ = "training positions"
        c_heat = heatmap(list(reversed(widths)), fracs, cell,
                        "Accuracy over width x data (cell: top-1 / examples-per-param)",
                        rowlab=lambda w: f"W={w}", collab=collab,
                        cellfmt=lambda v: f"{v:.2f}",
                        subfmt=lambda s: f"{s:.1f}x/p")
        interaction = (c_plateau, c_sat, c_heat)

    # --- table ---
    def row(p):
        pill = (f'<span class="pill" style="background:{p["regime_color"]}22;'
                f'color:{p["regime_color"]}">{p["regime"]}</span>')
        return (f"<tr><td>{p['depth']}</td><td>{p['width']}</td>"
                f"<td>{p['encoding']}</td><td>{p['input_dim']}</td>"
                f"<td>{p['params']/1e6:.2f}M</td>"
                f"<td>{_fmt_count(p['n_train'])}</td>"
                f"<td>{p['ex_per_param']:.2f}×</td><td>{pill}</td>"
                f"<td>{p['val_top1']:.3f}</td>"
                f"<td>{p['mean_regret']:.3f}</td>"
                f"<td>{p['illegal_rate']:.3f}</td>"
                f"<td>{p['elo']:.0f}</td>"
                f"<td class='eff'>{p['top1_per_mparam']:.3f}</td></tr>")
    rows = "".join(row(p) for p in sorted(pts, key=lambda p: p["params"]))

    ld = summary.get("label_depth")
    phase = summary.get("phase_distribution", {})
    phase_str = " · ".join(f"{k} {v:,}" for k, v in phase.items()) or "—"

    kpis = f'''
    <div class="kpi"><span class="kv">{best_q['val_top1']:.3f}</span>
      <span class="kl">best top-1</span><span class="kd">{dlabel(best_q)} {best_q['encoding']} · {best_q['params']/1e6:.1f}M params</span></div>
    <div class="kpi"><span class="kv">{best_eff['top1_per_mparam']:.2f}</span>
      <span class="kl">best top-1 / Mparam</span><span class="kd">{dlabel(best_eff)} · {best_eff['params']/1e6:.2f}M params</span></div>
    <div class="kpi"><span class="kv">{summary['n_positions']:,}</span>
      <span class="kl">training positions</span><span class="kd">label depth {ld if ld else '—'}</span></div>
    <div class="kpi"><span class="kv">{len(pts)}</span>
      <span class="kl">configs swept</span><span class="kd">axes: {', '.join(AXIS_LABEL[a] for a in axes_present)}</span></div>
    '''

    cards = "".join(f'<div class="card">{c}</div>' for c in charts)

    interaction_html = ""
    if interaction:
        c_plateau, c_sat, c_heat = interaction
        interaction_html = f'''
  <h2>Data × parameters — why capacity plateaus</h2>
  <p class="cap">Each line in the first chart is a <b>fixed training-set size</b>.
    Where data is scarce the line goes <b>flat</b> — past a point, extra parameters
    stop helping because the model just memorizes what little data it has. Only with
    enough data does adding parameters keep raising accuracy. The heatmap labels every
    cell with its <b>examples-per-parameter</b> ratio; below ~10× a config is
    data-starved and its size is largely wasted.</p>
  <div class="grid">
    <div class="card">{c_plateau}</div>
    <div class="card">{c_sat}</div>
    <div class="card">{c_heat}</div>
  </div>'''

    return f'''<title>Chess Policy — Efficiency &amp; Scaling</title>
<style>
  :root {{
    --paper:#f6f7fa; --ink:#14171f; --muted:#5b6472; --grid:#e4e7ee;
    --card:#ffffff; --line:#e4e7ee; --accent:#3D6FE3;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --paper:#0e1116; --ink:#e7eaf0; --muted:#8b93a3; --grid:#242a35;
            --card:#161a22; --line:#242a35; }}
  }}
  :root[data-theme="dark"] {{ --paper:#0e1116; --ink:#e7eaf0; --muted:#8b93a3;
    --grid:#242a35; --card:#161a22; --line:#242a35; }}
  :root[data-theme="light"] {{ --paper:#f6f7fa; --ink:#14171f; --muted:#5b6472;
    --grid:#e4e7ee; --card:#ffffff; --line:#e4e7ee; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 28px 48px; background:var(--paper); color:var(--ink);
    font:14px/1.55 -apple-system, system-ui, sans-serif;
    -webkit-font-smoothing:antialiased; }}
  .mono {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; }}
  header {{ max-width:1080px; margin:0 auto 22px; }}
  h1 {{ font-size:23px; margin:0 0 4px; letter-spacing:-0.02em; text-wrap:balance; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .sub .mono {{ color:var(--ink); }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
    gap:12px; margin:0 0 22px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 16px; display:flex; flex-direction:column; gap:2px; }}
  .kv {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:24px;
    font-weight:600; letter-spacing:-0.02em; font-variant-numeric:tabular-nums; }}
  .kl {{ font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
    color:var(--muted); }}
  .kd {{ font-size:11px; color:var(--muted); font-family:ui-monospace,monospace; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
    gap:14px; margin-bottom:22px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:10px 12px 6px; }}
  .chart {{ width:100%; height:auto; display:block; }}
  .ctitle {{ font-size:12.5px; font-weight:600; fill:var(--ink); }}
  .tick {{ font-size:10px; fill:var(--muted); font-family:ui-monospace,monospace;
    font-variant-numeric:tabular-nums; }}
  .dot {{ font-size:9.5px; fill:var(--muted); font-family:ui-monospace,monospace; }}
  .leg {{ font-size:10.5px; fill:var(--muted); }}
  .axl {{ font-size:10.5px; fill:var(--muted); text-transform:uppercase;
    letter-spacing:0.06em; }}
  .empty {{ color:var(--muted); padding:40px; text-align:center; }}
  .tablewrap {{ overflow-x:auto; background:var(--card); border:1px solid var(--line);
    border-radius:12px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12.5px;
    font-family:ui-monospace,"SF Mono",Menlo,monospace;
    font-variant-numeric:tabular-nums; }}
  th, td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--line);
    white-space:nowrap; }}
  th:nth-child(3), td:nth-child(3), th:nth-child(8), td:nth-child(8) {{ text-align:left; }}
  thead th {{ color:var(--muted); font-weight:600; font-size:10.5px;
    text-transform:uppercase; letter-spacing:0.05em; position:sticky; top:0;
    background:var(--card); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.eff {{ color:var(--accent); font-weight:600; }}
  h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:0.08em;
    color:var(--muted); margin:26px 0 10px; font-weight:600; }}
  .cap {{ color:var(--muted); font-size:12.5px; max-width:920px;
    margin:-4px 0 12px; line-height:1.5; }}
  .cap b {{ color:var(--ink); font-weight:600; }}
  .pill {{ font-size:10px; padding:2px 7px; border-radius:20px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.03em; }}
  .hm {{ font-size:11px; fill:var(--ink); font-family:ui-monospace,monospace;
    font-weight:600; }}
  .hmsub {{ font-size:8.5px; fill:var(--muted); font-family:ui-monospace,monospace; }}
</style>
<header class="wrap">
  <h1>Single-Pass Chess Policy — Efficiency &amp; Scaling</h1>
  <div class="sub"><span class="mono">{summary['n_positions']:,}</span> positions
    · phases: {phase_str} · play mode: <span class="mono">{summary['mode']}</span>
    · opponent ladder: <span class="mono">{', '.join(map(str, summary['ladder']))}</span>
    · label depth: <span class="mono">{ld if ld else '—'}</span></div>
</header>
<div class="wrap">
  <div class="kpis">{kpis}</div>
  <div class="grid">{cards}</div>
  {interaction_html}
  <h2>Every configuration</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>N</th><th>W</th><th>enc</th><th>in-dim</th>
      <th>params</th><th>train</th><th>ex/param</th><th>regime</th><th>top-1</th>
      <th>regret</th><th>illegal</th><th>Elo</th><th>top1/Mp</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.sweep, "summary.json")) as f:
        summary = json.load(f)
    out = args.out or os.path.join(args.sweep, "report.html")
    with open(out, "w") as f:
        f.write(build_html(summary))
    print(f"wrote {out} ({len(summary['points'])} points, "
          f"axes {sorted({a for p in summary['points'] for a in p.get('axes',[])})})")


if __name__ == "__main__":
    main()
