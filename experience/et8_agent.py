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

def generate(model, tok, messages: list[dict], max_tokens: int = 400, temp: float = 0.0,
             logits_processors=None) -> tuple[str, int, int]:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_in = len(tok.encode(prompt))
    kw = {"logits_processors": logits_processors} if logits_processors else {}
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False, **kw)
    return text, n_in, len(tok.encode(text))

CAND_TEMP = 0.8   # candidates 1..k-1; candidate 0 is greedy. Greedy alone returns one action k times.

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

def run_episode(model, tok, task: dict, budget: int, memory: str | None, run_id: str, log, model_id: str,
                logits_processors=None, trivial_symptom: bool = False,
                candidate_select: str = "", cand_k: int = 8,
                redact_symptom: bool = False) -> dict:
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
        # --redact-symptom: the v1 task set has TEN distinct symptom strings and each maps to
        # exactly ONE bug region, so the test NAME in the symptom line is a lookup key that
        # localises at 100% (experience/results/v1_symptom_determines_region.json). Redaction
        # keeps the failure KIND and the assertion MESSAGE and drops the test name, which is the
        # only part carrying the key. This does not make the task unsolvable -- the agent can
        # still inspect regions and read the program -- it makes localisation something the agent
        # has to DO rather than copy. The gap between the two arms is the size of the giveaway.
        sym = task["symptom"]
        if redact_symptom:
            kind = sym.split(":", 1)[0]
            msg = sym.split("\u2014", 1)[1].strip() if "\u2014" in sym else ""
            sym = f"{kind}: a test failed" + (f" \u2014 {msg}" if msg else "")
        state = f"SYMPTOM: {sym}\nRegions: {', '.join(task['regions'])}\n"
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
        # G-TRIVIAL (理 10888): the control G must beat. On the FIRST hypothesis only, substitute the
        # one-line rule the agent's own history implies — "start where the symptom points". No model
        # call is made for that step, which is the point: it costs nothing and it reaches the ceiling.
        # Every miss in the 2,000-run is an episode where the bug WAS in the symptom region and the
        # agent looked elsewhere (239 of 239), so this rule cannot lose ground it had.
        cand_meta = None
        if trivial_symptom and not any(h.startswith("hypothesize ") for h in history):
            text = json.dumps({"action": "hypothesize", "region": task["symptom_region"],
                               "bug_class": task.get("bug_class", ""), "why": "symptom region (G-trivial)"})
            n_in = n_out = 0
        # G-TRIVIAL-PRIME (理 10895): the control that separates WHERE from HOW. G-trivial replaced
        # the model's hypothesis STEP with a canned string, so it changed how the hypothesis was
        # expressed as well as where it pointed -- and the patch that follows may depend on the
        # model's own reasoning at that step. This variant changes ONLY the where: it enumerates k
        # candidates FROM THE MODEL (step 2), then picks the model's own candidate whose region is
        # the symptom region. Same plumbing G's head will use; the symptom rule standing in for the
        # head. If it lands near G-trivial, the 239 misses were never a localisation deficit. If it
        # recovers the 45, G-trivial's loss was the canned TEXT, not the choice.
        elif candidate_select and not any(h.startswith("hypothesize ") for h in history):
            msgs = messages + [{"role": "user", "content": state}]
            cands, seen = [], set()
            for j in range(cand_k):
                # candidate 0 is GREEDY -- it is exactly the action the baseline would have taken,
                # so the fallback below is the baseline, not a third behaviour.
                t, ni, no = generate(model, tok, msgs, temp=0.0 if j == 0 else CAND_TEMP,
                                     logits_processors=logits_processors)
                total_in += ni; total_out += no
                pa = parse_action(t)
                key = (pa.get("action"), pa.get("region"), pa.get("bug_class"))
                if key in seen:
                    continue
                seen.add(key); cands.append((t, pa))
            pick = next((c for c in cands if c[1].get("action") == "hypothesize"
                         and c[1].get("region") == task["symptom_region"]), None)
            cand_meta = {"k_asked": cand_k, "k_distinct": len(cands),
                         "regions_offered": sorted({str(c[1].get("region")) for c in cands}),
                         "fell_back_to_greedy": pick is None}
            text = (pick or cands[0])[0]
            n_in = n_out = 0          # already added inside the loop
        else:
            text, n_in, n_out = generate(model, tok, messages + [{"role": "user", "content": state}], temp=temp,
                                         logits_processors=logits_processors)
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
               "candidates": cand_meta,
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
    ap.add_argument("--logit-bias", help="directory from et8_logit_bias.py build (mechanism F): "
                                         "additive bias on the EMITTABLE action tokens only")
    ap.add_argument("--trivial-symptom", action="store_true",
                    help="G-TRIVIAL (理 10888): force the first hypothesis to the SYMPTOM region. "
                         "A one-line rule learned from the agent's own history and the control G "
                         "must beat — it needs no model, no head and no activations.")
    ap.add_argument("--candidate-select", default="", choices=["", "symptom"],
                    help="G-TRIVIAL-PRIME (理 10895): enumerate k candidates FROM THE MODEL at the "
                         "first hypothesis, then pick by rule. 'symptom' picks the model's own "
                         "candidate whose region is the symptom region, falling back to the greedy "
                         "candidate — which is exactly the baseline action. Separates WHERE from "
                         "HOW: unlike --trivial-symptom it does not replace the model's text.")
    ap.add_argument("--cand-k", type=int, default=8,
                    help="candidates to draw (candidate 0 greedy, the rest at CAND_TEMP). Deduped "
                         "by (action, region, bug_class), so k_distinct is usually smaller and is "
                         "logged: a model that offers one region k times cannot be steered by "
                         "selection at all, and that is a result, not a failure of the harness.")
    ap.add_argument("--redact-symptom", action="store_true",
                    help="drop the TEST NAME from the symptom line, keeping the failure kind and "
                         "the assertion message. v1's ten symptom strings each map to exactly one "
                         "bug region, so the name is a lookup key worth 100%% localisation; this "
                         "arm measures how much of the baseline rests on it.")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.tasks, "task_*.json")))
    if a.limit: files = files[: a.limit]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    memory = open(a.memory).read() if a.memory else None
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    t0 = time.time(); model, tok = load_model(a.model); print(f"model loaded in {time.time()-t0:.1f}s", file=sys.stderr)
    bias_tbl = None; _bias_logged = {}
    if a.logit_bias:
        import et8_logit_bias
        bias_tbl = json.load(open(os.path.join(a.logit_bias, "bias.json")))["bias"]
        print(f"logit bias loaded: families={sorted(bias_tbl)}", file=sys.stderr)

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
            # Mechanism F: the bias table is PER FAMILY, so the processor is built per task.
            lp = None
            if bias_tbl:
                fam = task.get("family")
                row = bias_tbl.get(fam)
                if row:
                    proc, applied = et8_logit_bias.install_logit_bias(model, tok, row)
                    lp = [proc] if proc else None
                    if not _bias_logged.get(fam):
                        print(f"  F bias {fam}: {applied}", file=sys.stderr); _bias_logged[fam] = True
                elif not _bias_logged.get("_miss"):
                    print(f"  F: no bias row for family {fam!r} — running UNBIASED", file=sys.stderr)
                    _bias_logged["_miss"] = True
            s = run_episode(model, tok, task, a.budget, memory, run_id, log, a.model,
                            logits_processors=lp, trivial_symptom=a.trivial_symptom,
                            candidate_select=a.candidate_select, cand_k=a.cand_k,
                            redact_symptom=a.redact_symptom)
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
