#!/usr/bin/env python
"""X5 figure — the extraction's headline: search lift vs base competence (pass@1), overlaid across
the Qwen2.5 and Llama-3.x frontiers on GSM8K + MATH. If the two families lie on one curve, the
size-vs-search crossover is a function of competence, not parameter count. Dependency-free SVG.

  python reasoning/llama_frontier_x5_fig.py
"""
import json

pts = json.load(open("reasoning/llama_frontier_analysis.json"))["x5_overlay_lift_vs_pass1"]

W, H = 720, 480
L, R, T, B = 70, 24, 44, 56          # margins
pw, ph = W - L - R, H - T - B
xmax, ymax = 100.0, 24.0

def px(v): return L + pw * v / xmax
def py(v): return T + ph * (1 - v / ymax)

STYLE = {("Qwen2.5", "gsm8k"): ("#2563eb", "circle"),
         ("Qwen2.5", "math"):  ("#2563eb", "square"),
         ("Llama3.x", "gsm8k"): ("#dc2626", "circle"),
         ("Llama3.x", "math"):  ("#dc2626", "square")}

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# gridlines + axis ticks
for gx in range(0, 101, 20):
    s.append(f'<line x1="{px(gx):.1f}" y1="{T}" x2="{px(gx):.1f}" y2="{T+ph}" stroke="#eee"/>')
    s.append(f'<text x="{px(gx):.1f}" y="{T+ph+18}" font-size="12" fill="#444" text-anchor="middle">{gx}</text>')
for gy in range(0, int(ymax)+1, 4):
    s.append(f'<line x1="{L}" y1="{py(gy):.1f}" x2="{L+pw}" y2="{py(gy):.1f}" stroke="#eee"/>')
    s.append(f'<text x="{L-8}" y="{py(gy)+4:.1f}" font-size="12" fill="#444" text-anchor="end">{gy}</text>')
# axes
s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#333"/>')
s.append(f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#333"/>')
s.append(f'<text x="{L+pw/2:.0f}" y="{H-14}" font-size="14" fill="#111" text-anchor="middle">Base competence — pass@1 (%)</text>')
s.append(f'<text x="18" y="{T+ph/2:.0f}" font-size="14" fill="#111" text-anchor="middle" transform="rotate(-90 18 {T+ph/2:.0f})">Search lift — sc@N − pass@1 (pts)</text>')
s.append(f'<text x="{L}" y="26" font-size="16" font-weight="bold" fill="#111">Search lift collapses with competence — Qwen &amp; Llama on one curve</text>')
# points
for p in pts:
    color, shape = STYLE[(p["family"], p["bench"])]
    x, y = px(p["pass1"]), py(p["lift"])
    if shape == "circle":
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" fill-opacity="0.78" stroke="white" stroke-width="1"/>')
    else:
        s.append(f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{color}" fill-opacity="0.78" stroke="white" stroke-width="1"/>')
# legend
lx, ly = L + pw - 176, T + 8
s.append(f'<rect x="{lx-8}" y="{ly-6}" width="176" height="86" fill="white" stroke="#ddd" rx="4"/>')
leg = [("#2563eb", "Qwen2.5"), ("#dc2626", "Llama-3.x")]
for i, (c, name) in enumerate(leg):
    yy = ly + 12 + i*18
    s.append(f'<circle cx="{lx+6}" cy="{yy-4}" r="6" fill="{c}" fill-opacity="0.78"/>')
    s.append(f'<text x="{lx+20}" y="{yy}" font-size="12" fill="#111">{name}</text>')
s.append(f'<circle cx="{lx+6}" cy="{ly+52}" r="6" fill="#666"/>')
s.append(f'<text x="{lx+20}" y="{ly+56}" font-size="12" fill="#111">circle = GSM8K</text>')
s.append(f'<rect x="{lx+1}" y="{ly+65}" width="11" height="11" fill="#666"/>')
s.append(f'<text x="{lx+20}" y="{ly+74}" font-size="12" fill="#111">square = MATH</text>')
s.append('</svg>')
open("reasoning/llama_frontier_x5_overlay.svg", "w").write("\n".join(s))
print("[fig] wrote reasoning/llama_frontier_x5_overlay.svg")
