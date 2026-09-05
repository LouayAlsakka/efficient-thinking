#!/usr/bin/env python3
"""
et8_inject.py — Efficient Thinking VIII INJECTION module v0 (proposal §4): where the experience vector goes into a frozen
Qwen2.5 model under mlx-lm, and how it is built without gradients first.

Mechanisms implemented here (the two cheapest, gradient-free to start):
  C  steering vectors   h_l <- h_l + alpha_l * v_l  at chosen decoder layers, applied at DECISION tokens (or all tokens).
                        v_l initialised by CONTRASTIVE MEANS: mean hidden state at the decision position over productive
                        (tau+) decision prompts minus the mean over wasted (tau-) ones, unit-normalised. alpha fitted later.
  F  action-logit bias  a bias added to the logits of the first token of each candidate action/region word at the decision
                        position, from the same tau+/tau- statistics (log-odds of productive vs wasted region choice).
D (residual adapter), B (prefix) and E (LoRA) need gradients through the frozen model and are built after C/F have numbers.

The hook: mlx-lm Qwen2 exposes model.model.layers[i]; we wrap the layer's __call__ so its output residual gets the addition.
mlx arrays are functional, so the wrapper returns a new array; the mask selects positions. Nothing in the base model is
modified; the hook is removable (restore the original __call__).

Usage (v0):
  # 1. collect decision-position hidden states for tau+/tau- prompts from a run and build vectors at layers 9/18/27
  python experience/et8_inject.py build --runs experience/traj/v03_3b --tasks experience/tasks/v0 \
        --model Qwen/Qwen2.5-3B-Instruct --layers 9 18 27 --out experience/vectors/v0
  # 2. sanity: the vector's projection separates tau+ from tau- on held-out decision prompts (a linear-probe style check)
  python experience/et8_inject.py check --vectors experience/vectors/v0 --runs experience/traj/v03_7b --tasks experience/tasks/v0 --model Qwen/Qwen2.5-3B-Instruct
Status: v0 — build + check. The agent-side hook (running et8_agent with --steer) is wired in et8_agent.py once check passes.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import et8_agent as A
import et8_env as E

def decision_prompt(task: dict, history: list[str], inspected: dict[str, str], budget: int, step: int) -> list[dict]:
    """Rebuild the exact chat the agent saw at a decision step (mirrors et8_agent.run_episode's state construction)."""
    state = f"SYMPTOM: {task['symptom']}\nRegions: {', '.join(task['regions'])}\n"
    if inspected:
        state += f"Already inspected (do not inspect again): {', '.join(inspected)}\n"
        state += "\n".join(f"--- region {r} ---\n{s}" for r, s in inspected.items()) + "\n"
    if history:
        state += "Your previous actions: " + " | ".join(history[-6:]) + "\n"
    state += f"Actions left: {budget - step + 1}. Next action (one JSON object):"
    return [{"role": "system", "content": A.SYSTEM % (", ".join(E.BUG_CLASSES), budget)}, {"role": "user", "content": state}]

def hidden_at_layers(model, tok, messages: list[dict], layers: list[int]):
    """Run the frozen model once; return {layer: hidden state at the LAST prompt position} (the decision position)."""
    import mlx.core as mx
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = mx.array([tok.encode(prompt)])
    captured = {}
    inner = model.model  # Qwen2Model: embed_tokens, layers, norm
    h = inner.embed_tokens(ids)
    # build the causal mask the way mlx-lm does for a full prompt
    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, None)
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, None)
        if i in layers:
            captured[i] = h[0, -1, :]      # last position, shape (hidden,)
    mx.eval(*captured.values())
    return captured

def first_decision_states(steps: list[dict], tasks_dir: str):
    """From a run's step log, reconstruct the prompt at the FIRST HYPOTHESIS step of each episode and its label."""
    by_task = {}
    for s in steps: by_task.setdefault(s["task_id"], []).append(s)
    out = []
    for tid, st in by_task.items():
        st.sort(key=lambda s: s["step"])
        task = json.load(open(os.path.join(tasks_dir, f"{tid}.json")))
        history, inspected = [], {}
        for s in st:
            if s["action"] == "hypothesize":
                label = "tau+" if s["region_hit"] else "tau-"
                out.append({"task_id": tid, "step": s["step"], "label": label, "region": s["region"],
                            "messages": decision_prompt(task, history, inspected, 12, s["step"])})
                break
            if s["action"] == "inspect" and s["region"]:
                inspected[s["region"]] = A.region_source(task["program"], s["region"]) or ""
                history.append(f"inspect {s['region']}")
    return out

def cmd_build(a):
    import mlx.core as mx, numpy as np
    model, tok = A.load_model(a.model)
    steps = []
    for r in a.runs: steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = first_decision_states(steps, a.tasks)
    pos = [d for d in dec if d["label"] == "tau+"]; neg = [d for d in dec if d["label"] == "tau-"]
    print(f"decision prompts: {len(dec)} (tau+ {len(pos)}, tau- {len(neg)})", file=sys.stderr)
    sums = {l: [np.zeros(0), np.zeros(0)] for l in a.layers}
    acc = {l: {"pos": [], "neg": []} for l in a.layers}
    for d in dec:
        hs = hidden_at_layers(model, tok, d["messages"], a.layers)
        for l, h in hs.items():
            acc[l]["pos" if d["label"] == "tau+" else "neg"].append(np.array(h.astype(mx.float32)))
    os.makedirs(a.out, exist_ok=True)
    meta = {"model": a.model, "layers": a.layers, "n_pos": len(pos), "n_neg": len(neg), "runs": a.runs, "vectors": {}}
    for l in a.layers:
        P = np.stack(acc[l]["pos"]) if acc[l]["pos"] else None; N = np.stack(acc[l]["neg"]) if acc[l]["neg"] else None
        if P is None or N is None:
            print(f"layer {l}: not enough data (pos={len(acc[l]['pos'])}, neg={len(acc[l]['neg'])})", file=sys.stderr); continue
        v = P.mean(0) - N.mean(0); norm = float(np.linalg.norm(v)); v = v / (norm + 1e-8)
        # separation on the build set (optimistic; `check` does it held-out): projection of each state on v
        proj_p = P @ v; proj_n = N @ v
        d_prime = float((proj_p.mean() - proj_n.mean()) / (np.sqrt(0.5 * (proj_p.var() + proj_n.var())) + 1e-8))
        np.save(os.path.join(a.out, f"v_layer{l}.npy"), v.astype(np.float32))
        meta["vectors"][str(l)] = {"norm_before_unit": round(norm, 3), "d_prime_buildset": round(d_prime, 3),
                                   "mean_proj_pos": round(float(proj_p.mean()), 3), "mean_proj_neg": round(float(proj_n.mean()), 3),
                                   "hidden_mean_norm": round(float(np.linalg.norm(np.concatenate([P, N]), axis=1).mean()), 2)}
        print(f"layer {l}: |v|={norm:.2f} d'={d_prime:.2f}", file=sys.stderr)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta["vectors"], indent=1))

def cmd_check(a):
    """Held-out separation: project a DIFFERENT run's decision states on the saved vectors; report d' and a threshold accuracy."""
    import mlx.core as mx, numpy as np
    meta = json.load(open(os.path.join(a.vectors, "meta.json")))
    layers = [int(l) for l in meta["vectors"]]
    vecs = {l: np.load(os.path.join(a.vectors, f"v_layer{l}.npy")) for l in layers}
    model, tok = A.load_model(a.model)
    steps = []
    for r in a.runs: steps += [json.loads(l) for l in open(r + ".steps.jsonl")]
    dec = first_decision_states(steps, a.tasks)
    proj = {l: {"pos": [], "neg": []} for l in layers}
    for d in dec:
        hs = hidden_at_layers(model, tok, d["messages"], layers)
        for l, h in hs.items():
            proj[l]["pos" if d["label"] == "tau+" else "neg"].append(float(np.array(h.astype(mx.float32)) @ vecs[l]))
    out = {}
    for l in layers:
        p, n = np.array(proj[l]["pos"]), np.array(proj[l]["neg"])
        if len(p) == 0 or len(n) == 0: out[l] = "insufficient"; continue
        thr = 0.5 * (p.mean() + n.mean()); acc = (np.sum(p > thr) + np.sum(n <= thr)) / (len(p) + len(n))
        dp = float((p.mean() - n.mean()) / (np.sqrt(0.5 * (p.var() + n.var())) + 1e-8))
        out[l] = {"n_pos": len(p), "n_neg": len(n), "d_prime_heldout": round(dp, 3), "threshold_acc": round(float(acc), 3)}
    print(json.dumps({"vectors": a.vectors, "check_runs": a.runs, "layers": out}, indent=1))

# ---- the hook (used by et8_agent once check passes) -------------------------------------------------------------------
def install_steering(model, vectors: dict[int, "np.ndarray"], alpha: float, decision_only: bool = True):
    """Wrap chosen decoder layers so their residual output gets + alpha * v at the last position (decision token) or everywhere.
    Returns a restore() function."""
    import mlx.core as mx
    originals = {}
    for l, v in vectors.items():
        layer = model.model.layers[l]; orig = layer.__call__
        vv = mx.array(v)
        def wrapped(x, *args, _orig=orig, _v=vv, **kw):
            h = _orig(x, *args, **kw)
            if decision_only:
                add = mx.zeros_like(h); add = add.at[:, -1, :].add(alpha * _v)
                return h + add
            return h + alpha * _v
        layer.__call__ = wrapped; originals[l] = orig
    def restore():
        for l, orig in originals.items(): model.model.layers[l].__call__ = orig
    return restore

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--runs", nargs="+", required=True); b.add_argument("--tasks", required=True)
    b.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); b.add_argument("--layers", nargs="+", type=int, default=[9, 18, 27]); b.add_argument("--out", required=True); b.set_defaults(fn=cmd_build)
    c = sp.add_parser("check"); c.add_argument("--vectors", required=True); c.add_argument("--runs", nargs="+", required=True); c.add_argument("--tasks", required=True)
    c.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); c.set_defaults(fn=cmd_check)
    a = ap.parse_args(); a.fn(a)
