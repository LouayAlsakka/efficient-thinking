#!/usr/bin/env python3
"""
et8_agent.py — Efficient Thinking VIII agent harness v0: a FROZEN LLM runs the explicit action loop over et8_env tasks
and every step is logged as a trajectory record. No learning happens here; this is the DETECT stage (proposal §3.1) and
the frozen baseline for every later injection (proposal §4, mechanism A when --memory is given).

Action loop (proposal §2.1), one JSON action per model turn:
  {"action":"hypothesize","region":<r>,"bug_class":<c>,"why":"..."}      # which region / class to suspect next
  {"action":"inspect","region":<r>}                                       # reveal that region's source
  {"action":"patch","region":<r>,"source":"<full replacement source for the region>"}
  {"action":"run"}                                                        # run the external verifier
Budget: --budget actions (default 12). Episode ends green, or when the budget is exhausted.

Trajectory record (one JSONL line per step) — the schema the distill step consumes:
  task_id, family, bug_class, bug_region, symptom_region, step, action, region, hypothesis_class,
  productive (hindsight: region == bug_region or hypothesis_class == bug_class), dead_path (hypothesis in the family's
  dead paths), tokens_in, tokens_out, verifier {green, n_fail}, model, run_id
Episode summary appended to <out>.episodes.jsonl: actions, tokens, green, first_correct_hypothesis_step.

Usage:
  python experience/et8_agent.py --tasks experience/tasks/v0 --limit 3 --out experience/traj/smoke      # smoke test
  python experience/et8_agent.py --tasks experience/tasks/v0 --model Qwen/Qwen2.5-3B-Instruct --out experience/traj/base_v0
  python experience/et8_agent.py ... --memory experience/lessons/v1.txt                                   # mechanism A
Runs on mlx-lm (Apple Silicon). The same loop with a HF backend is a one-function swap (generate()).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time, uuid
sys.path.insert(0, os.path.dirname(__file__))
import et8_env as E

SYSTEM = """You are debugging a small Python program. You will be shown the failing test (the SYMPTOM), the list of code
regions, and any regions you have inspected. Act in small steps. Reply with EXACTLY ONE JSON object and nothing else, one of:
{"action":"hypothesize","region":"<region>","bug_class":"<class>","why":"<one sentence>"}
{"action":"inspect","region":"<region>"}
{"action":"patch","region":"<region>","source":"<the full corrected source of that region, as a JSON string>"}
{"action":"run"}
Regions: producer, transform, aggregate, consumer (consumer calls the others: producer -> transform -> aggregate).
Bug classes: %s.
A failing test in one region may be caused by a bug UPSTREAM of it. You have %d actions in total. Rules: inspecting a
region you have already seen is wasted (it is shown to you already); every patch runs the tests automatically and
shows you the result; hypothesize before you patch.

Shape of a good episode (placeholders, not advice about where the bug is):
  -> {"action":"hypothesize","region":"<REGION>","bug_class":"<CLASS>","why":"<one sentence>"}
  -> {"action":"inspect","region":"<REGION>"}
  -> {"action":"patch","region":"<REGION>","source":"def <name>(...):\\n    <full corrected source>\\n"}
  (tests run automatically; if not green, CHANGE the code or look elsewhere — the same source again is wasted)"""

def load_model(model_id: str):
    from mlx_lm import load
    return load(model_id)

def generate(model, tok, messages: list[dict], max_tokens: int = 400, temp: float = 0.0) -> tuple[str, int, int]:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_in = len(tok.encode(prompt))
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False)
    return text, n_in, len(tok.encode(text))

# Exploration rule (harness v0.3): the FIRST attempt at any step is greedy (temp 0, reproducible). After a WASTED step
# (repeat inspect / repeat hypothesis / identical patch / invalid) the next generation samples at EXPLORE_TEMP, and the
# temperature rises with consecutive wasted steps, capped. Greedy decoding alone cannot search: an identical prompt
# yields an identical patch, so a failed patch was re-submitted for the rest of the budget (v0.2: 136/186 steps).
EXPLORE_TEMP, EXPLORE_STEP, EXPLORE_CAP = 0.7, 0.15, 1.0
WASTED = ("repeat_inspect", "repeat_hypothesis", "repeat_patch", "noop_patch", "invalid")

def parse_action(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)   # strip code fences
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"action": "invalid", "raw": text[:200]}
    cand = m.group(0)
    a = None
    for attempt in (cand, re.sub(r",\s*}", "}", cand)):
        try:
            a = json.loads(attempt, strict=False)      # strict=False: literal newlines inside "source" are accepted
            break
        except json.JSONDecodeError:
            continue
    if a is None:
        # last resort: pull the fields by regex so a model that writes raw multi-line source is not scored invalid
        act = re.search(r'"action"\s*:\s*"(\w+)"', cand); reg = re.search(r'"region"\s*:\s*"(\w+)"', cand)
        src = re.search(r'"source"\s*:\s*"(.*)"\s*}?\s*$', cand, flags=re.S)
        if act and reg and src:
            a = {"action": act.group(1), "region": reg.group(1),
                 "source": src.group(1).encode().decode("unicode_escape") if "\\n" in src.group(1) else src.group(1)}
        else:
            return {"action": "invalid", "raw": text[:200]}
    if not isinstance(a, dict) or "action" not in a:
        return {"action": "invalid", "raw": text[:200]}
    return a

def region_source(program: str, region: str) -> str | None:
    """Extract a region's source from an assembled program (regions are delimited by '# region: <name>' headers)."""
    parts = re.split(r"^# region: (\w+)\n", program, flags=re.M)
    d = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    return d.get(region)

def replace_region(program: str, region: str, new_src: str) -> str:
    parts = re.split(r"^# region: (\w+)\n", program, flags=re.M)
    out = parts[0]
    for i in range(1, len(parts) - 1, 2):
        name, src = parts[i], parts[i + 1]
        out += f"# region: {name}\n" + (new_src.rstrip("\n") + "\n\n" if name == region else src)
    return out

def run_episode(model, tok, task: dict, budget: int, memory: str | None, run_id: str, log, model_id: str) -> dict:
    program = task["program"]
    inspected: dict[str, str] = {}
    history: list[str] = []
    messages = [{"role": "system", "content": SYSTEM % (", ".join(E.BUG_CLASSES), budget)}]
    if memory:
        messages.append({"role": "system", "content": "Lessons from previous work (may or may not apply):\n" + memory})
    green, fails = False, []
    total_in = total_out = 0
    first_correct = None
    wasted_streak = 0
    tried_patches: dict[str, set[str]] = {}
    for step in range(1, budget + 1):
        temp = 0.0 if wasted_streak == 0 else min(EXPLORE_CAP, EXPLORE_TEMP + EXPLORE_STEP * (wasted_streak - 1))
        state = f"SYMPTOM: {task['symptom']}\nRegions: {', '.join(task['regions'])}\n"
        if inspected:
            state += f"Already inspected (do not inspect again): {', '.join(inspected)}\n"
            state += "\n".join(f"--- region {r} ---\n{s}" for r, s in inspected.items()) + "\n"
        if history:
            state += "Your previous actions: " + " | ".join(history[-6:]) + "\n"
        if fails:
            state += "Last verifier result: " + "; ".join(f"{f['kind']} {f['test']} — {f.get('message','')}" for f in fails[:4]) + "\n"
        if history and history[-1].startswith("noop patch"):
            state += ("Your last 'patch' was IDENTICAL to the current code — nothing changed. Either edit the code (the bug is a "
                      "specific line), or the bug is in a DIFFERENT region: hypothesize and inspect another one.\n")
        elif history and history[-1].startswith(("patch", "repeat patch")) and fails:
            state += ("Your last patch did NOT fix it (see verifier result). Submitting the same source again is wasted: "
                      "change the code, or hypothesize/inspect a different region.\n")
        last_hyp = next((h for h in reversed(history) if h.startswith("hypothesize ")), None)
        if last_hyp:
            r = last_hyp.split()[1].split("/")[0]
            state += (f"You already suspect {r}. Do not repeat the hypothesis: "
                      f"{'inspect it' if r not in inspected else 'PATCH it now (patch runs the tests)'}.\n")
        state += f"Actions left: {budget - step + 1}. Next action (one JSON object):"
        text, n_in, n_out = generate(model, tok, messages + [{"role": "user", "content": state}], temp=temp)
        total_in += n_in; total_out += n_out
        a = parse_action(text)
        kind = a.get("action", "invalid"); region = a.get("region"); hyp = a.get("bug_class")
        verifier = None
        if kind == "inspect" and region in task["regions"] and region in inspected:
            kind = "repeat_inspect"; history.append(f"repeat inspect {region} (wasted)")
        elif kind == "inspect" and region in task["regions"]:
            inspected[region] = region_source(program, region) or ""
            history.append(f"inspect {region}")
        elif kind == "hypothesize" and f"hypothesize {region}/{hyp}" in history:
            kind = "repeat_hypothesis"; history.append(f"repeat hypothesis {region}/{hyp} (wasted — act on it)")
        elif kind == "hypothesize":
            history.append(f"hypothesize {region}/{hyp}")
            if first_correct is None and (region == task["bug_region"] or hyp == task["bug_class"]):
                first_correct = step
        elif kind == "patch" and region in task["regions"] and isinstance(a.get("source"), str) \
                and a["source"].strip() == (region_source(program, region) or "").strip():
            # the "patch" is byte-identical to the code currently in the program: nothing was changed.
            # v0.3 lumped this with re-submitting a failed patch; they are different failures. This one is the
            # model not knowing WHAT to change (it copies what it was shown) — the dominant waste in v0.1–v0.3.
            kind = "noop_patch"; history.append(f"noop patch {region} (identical to current code — nothing changed)")
        elif kind == "patch" and region in task["regions"] and isinstance(a.get("source"), str) \
                and a["source"].strip() in tried_patches.get(region, set()):
            kind = "repeat_patch"; history.append(f"repeat patch {region} (a patch you already tried — wasted)")
        elif kind == "patch" and region in task["regions"] and isinstance(a.get("source"), str):
            tried_patches.setdefault(region, set()).add(a["source"].strip())
            program = replace_region(program, region, a["source"])
            inspected[region] = a["source"]
            green, fails = E.run_tests(program, task["tests"])          # a patch always runs the verifier
            verifier = {"green": green, "n_fail": len(fails)}
            history.append(f"patch {region} -> {'green' if green else str(len(fails)) + ' failing'}")
        elif kind == "run":
            green, fails = E.run_tests(program, task["tests"])
            verifier = {"green": green, "n_fail": len(fails)}
            history.append(f"run -> {'green' if green else str(len(fails)) + ' failing'}")
        else:
            kind = "invalid"; history.append("invalid")
        wasted_streak = wasted_streak + 1 if kind in WASTED else 0
        acted = kind in ("hypothesize", "inspect", "patch")
        region_hit = (region == task["bug_region"]) if acted else None
        class_hit = (hyp == task["bug_class"]) if kind == "hypothesize" else None
        productive = (region_hit or bool(class_hit)) if acted else None   # any wasted/invalid/repeat step is None
        rec = {"run_id": run_id, "model": model_id, "task_id": task["task_id"], "family": task["family"],
               "bug_class": task["bug_class"], "bug_region": task["bug_region"], "symptom_region": task["symptom_region"],
               "step": step, "action": kind, "region": region, "hypothesis_class": hyp,
               "productive": productive, "region_hit": region_hit, "class_hit": class_hit,
               "dead_path": (hyp in task["dead_paths"]) if hyp else False,
               "raw": (text[:300] if kind == "invalid" else None), "temp": temp,
               "tokens_in": n_in, "tokens_out": n_out, "verifier": verifier}
        log.write(json.dumps(rec) + "\n")
        if green:
            break
    return {"run_id": run_id, "task_id": task["task_id"], "family": task["family"], "bug_class": task["bug_class"],
            "green": green, "actions": step, "tokens_in": total_in, "tokens_out": total_out,
            "first_correct_hypothesis_step": first_correct, "memory": bool(memory)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--memory", help="text-memory file (mechanism A)")
    ap.add_argument("--steer-vectors", help="directory from et8_inject.py build (mechanism C): installs h_l += alpha*v_l")
    ap.add_argument("--steer-layers", nargs="+", type=int, help="subset of the built layers to install (default: all in meta)")
    ap.add_argument("--steer-alpha", type=float, default=4.0, help="steering strength in hidden units (hidden norms are ~40-90)")
    ap.add_argument("--steer-all-tokens", action="store_true", help="apply at every position instead of the decision token only")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
    if a.limit: files = files[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    memory = open(a.memory).read() if a.memory else None
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    t0 = time.time(); model, tok = load_model(a.model); print(f"model loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    steer = None
    if a.steer_vectors:
        import numpy as np, et8_inject
        meta = json.load(open(os.path.join(a.steer_vectors, "meta.json")))
        layers = a.steer_layers or [int(l) for l in meta["vectors"]]
        vecs = {l: np.load(os.path.join(a.steer_vectors, f"v_layer{l}.npy")) for l in layers}
        et8_inject.install_steering(model, vecs, a.steer_alpha, decision_only=not a.steer_all_tokens)
        steer = {"vectors": a.steer_vectors, "layers": layers, "alpha": a.steer_alpha, "decision_only": not a.steer_all_tokens}
        print(f"steering installed: layers={layers} alpha={a.steer_alpha} decision_only={not a.steer_all_tokens}", file=sys.stderr)
    summaries = []
    with open(a.out + ".steps.jsonl", "a") as log, open(a.out + ".episodes.jsonl", "a") as ep:
        for i, f in enumerate(files, 1):
            task = json.load(open(f)); t1 = time.time()
            s = run_episode(model, tok, task, a.budget, memory, run_id, log, a.model)
            s["seconds"] = round(time.time() - t1, 1); ep.write(json.dumps(s) + "\n"); summaries.append(s)
            print(f"[{i}/{len(files)}] {task['task_id']} {task['family']:10s} {task['bug_class']:20s} "
                  f"green={s['green']} actions={s['actions']} first_correct={s['first_correct_hypothesis_step']} "
                  f"tok={s['tokens_in']+s['tokens_out']} {s['seconds']}s", file=sys.stderr)
    n = len(summaries); g = sum(s["green"] for s in summaries)
    print(json.dumps({"run_id": run_id, "model": a.model, "tasks": n, "green": g, "success": round(g / n, 3) if n else None,
                      "mean_actions": round(sum(s["actions"] for s in summaries) / n, 2) if n else None,
                      "mean_tokens": round(sum(s["tokens_in"] + s["tokens_out"] for s in summaries) / n) if n else None,
                      "memory": bool(memory), "steer": steer}, indent=1))

if __name__ == "__main__":
    main()
