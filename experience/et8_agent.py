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
             logits_processors=None, prefix: str = "") -> tuple[str, int, int]:
    """`prefix` is TEACHER-FORCED: it is appended to the prompt and prepended to the returned text,
    so the caller gets a complete action whose opening fields it chose and whose remainder the model
    wrote. That is what makes a STRUCTURAL candidate set possible -- every region becomes a
    candidate by construction instead of appearing only when the sampler happens to propose it."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
    n_in = len(tok.encode(prompt))
    kw = {"logits_processors": logits_processors} if logits_processors else {}
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temp), verbose=False, **kw)
    return prefix + text, n_in, len(tok.encode(text))

def _rank_regions_by_first_token(model, tok, messages, regions):
    """Order regions by the model's own probability on the first token of each region name, so a
    k cap keeps the model's preferred candidates rather than an arbitrary slice. Only reached when
    a task has more regions than cand_k allows; v1 and v2 both have four, so it is untaken there
    and is written for the next task set rather than exercised by this one."""
    import mlx.core as mx
    pre = '{"action": "hypothesize", "region": "'
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + pre
    ids = mx.array([tok.encode(prompt)])
    logits = model(ids)[0, -1, :]
    lp = mx.log(mx.softmax(logits.astype(mx.float32)))
    score = {}
    for r in regions:
        t = tok.encode(r, add_special_tokens=False)
        score[r] = float(lp[t[0]]) if t else -1e9
    return sorted(regions, key=lambda r: -score[r])

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
            # A TRUNCATED \uXXXX ESCAPE CRASHES unicode_escape, and a crash here kills the whole
            # run, not the step. The 7B never produced one in 3,000+ episodes; a Llama-3B produced
            # one in its third slice and took the process down with it -- so this was a latent
            # harness bug that only a second model could surface. A malformed action is an INVALID
            # action, which the loop already knows how to score; it is not a reason to stop.
            raw_src = src.group(1)
            if "\\n" in raw_src:
                try:
                    raw_src = raw_src.encode().decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    return {"action": "invalid", "raw": text[:200]}
            a = {"action": act.group(1), "region": reg.group(1), "source": raw_src}
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
                redact_symptom: bool = False, inspect_first: bool = False,
                head_state=None) -> dict:
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
        # G-TRIVIAL (理: the control G must beat. On the FIRST hypothesis only, substitute the
        # one-line rule the agent's own history implies — "start where the symptom points". No model
        # call is made for that step, which is the point: it costs nothing and it reaches the ceiling.
        # Every miss in the 2,000-run is an episode where the bug WAS in the symptom region and the
        # agent looked elsewhere (239 of 239), so this rule cannot lose ground it had.
        cand_meta = None
        # INSPECT-FIRST (理 (a)). A controller head at an observation-free decision point is a
        # lookup table on the prompt: at step 1 the prompt carries only the symptom and the region
        # names, so 80 v2 episodes produced SIX distinct prompts and any head on that state is a
        # six-row dict. This requires ONE inspect before the first hypothesis so the decision the
        # head reads is made with something observed in it.
        #
        # WHICH region is the AGENT'S OWN greedy choice, not the symptom region and not forced: the
        # inspect step stays the model's, and the head does not act there (a head at THAT step would
        # be the same six-row table). G acts at the first hypothesis AFTER the inspect.
        # G: the READ-ONLY HEAD (理). At the first hypothesis AFTER an inspect -- the only
        # decision point in this programme whose prompts are distinct (gate 1.000 on v3, 0.005-0.088
        # everywhere else) -- score one teacher-forced candidate per region and take the argmax. The
        # head never writes: it chooses among actions the model itself would emit, and the
        # hypothesis TEXT stays the model's own. Agreement with the agent's unaided pick is logged,
        # because where they agree G changes nothing and any effect must come from disagreements.
        if head_state is not None and inspected and not any(h.startswith("hypothesize ") for h in history):
            import numpy as _np
            msgs = messages + [{"role": "user", "content": state}]
            scores, pres = {}, {}
            for r in task["regions"]:
                pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
                hs = head_state["hidden"](model, tok, msgs, [head_state["layer"]], prefix=pre)
                z = _np.array(hs[head_state["layer"]].astype(head_state["mx"].float32), copy=False)
                z = (z.astype(_np.float64) - head_state["mu"]) / head_state["sd"]
                scores[r] = float(z @ head_state["w"] + head_state["b"])
                pres[r] = pre
            pick = max(scores, key=scores.get)
            text, ni, no = generate(model, tok, msgs, temp=0.0, logits_processors=logits_processors)
            total_in += ni; total_out += no
            gp = parse_action(text)
            agree = (gp.get("action") == "hypothesize" and gp.get("region") == pick)
            if not agree:
                text, ni, no = generate(model, tok, msgs, temp=0.0, prefix=pres[pick],
                                        logits_processors=logits_processors)
                total_in += ni; total_out += no
            cand_meta = {"head": True, "head_pick": pick, "agent_pick": gp.get("region"),
                         "agreed": bool(agree),
                         "scores": {k: round(v, 3) for k, v in scores.items()}}
            n_in = n_out = 0
        elif inspect_first and not inspected and not any(h.startswith("hypothesize ") for h in history):
            text, n_in, n_out = generate(model, tok, messages + [{"role": "user", "content":
                state + "\nInspect a region first: reply with an inspect action."}],
                temp=temp, logits_processors=logits_processors)
            pa = parse_action(text)
            if pa.get("action") != "inspect" or pa.get("region") not in task["regions"]:
                # the model declined to inspect; take its greedy region if it named one, else the
                # first region. Recorded on the step so a forced fallback is never invisible.
                r = pa.get("region") if pa.get("region") in task["regions"] else task["regions"][0]
                text = json.dumps({"action": "inspect", "region": r,
                                   "why": "inspect-first fallback (model did not emit an inspect)"})
                cand_meta = {"inspect_first_fallback": True, "model_action": pa.get("action")}
            else:
                cand_meta = {"inspect_first_fallback": False}
        elif trivial_symptom and not any(h.startswith("hypothesize ") for h in history):
            text = json.dumps({"action": "hypothesize", "region": task["symptom_region"],
                               "bug_class": task.get("bug_class", ""), "why": "symptom region (G-trivial)"})
            n_in = n_out = 0
        # G-TRIVIAL-PRIME (理: the control that separates WHERE from HOW. G-trivial replaced
        # the model's hypothesis STEP with a canned string, so it changed how the hypothesis was
        # expressed as well as where it pointed -- and the patch that follows may depend on the
        # model's own reasoning at that step. This variant changes ONLY the where: it enumerates k
        # candidates FROM THE MODEL (step 2), then picks the model's own candidate whose region is
        # the symptom region. Same plumbing G's head will use; the symptom rule standing in for the
        # head. If it lands near G-trivial, the 239 misses were never a localisation deficit. If it
        # recovers the 45, G-trivial's loss was the canned TEXT, not the choice.
        elif candidate_select and not any(h.startswith("hypothesize ") for h in history):
            msgs = messages + [{"role": "user", "content": state}]
            # STRUCTURAL candidate set (理, not a sampled one. One candidate per region
            # VISIBLE AT THIS DECISION POINT, each built by teacher-forcing the opening fields and
            # letting the model write the rest -- so every region is a candidate BY CONSTRUCTION.
            #
            # Sampling was the wrong instrument and the run said so before the ruling did: across
            # k=8 draws at temp 0.8, HALF the decision points offered exactly one region. A head
            # scoring a one-element set cannot move the agent, so G would have been a test of the
            # sampler. The sampler's proposal entropy is still worth knowing and is measured
            # separately (et8_candidate_diversity.py); it is not G's action space.
            cands, seen = [], set()
            greedy_text, ni, no = generate(model, tok, msgs, temp=0.0,
                                           logits_processors=logits_processors)
            total_in += ni; total_out += no
            gp = parse_action(greedy_text)
            cands.append((greedy_text, gp))          # greedy is ALWAYS in the set
            seen.add((gp.get("action"), gp.get("region"), gp.get("bug_class")))
            regions = [r for r in task["regions"]]
            if len(regions) + 1 > cand_k:            # cap k <= cand_k by the model's own preference
                regions = _rank_regions_by_first_token(model, tok, msgs, regions)[: cand_k - 1]
            for r in regions:
                pre = '{"action": "hypothesize", "region": "%s", "bug_class": "' % r
                t, ni, no = generate(model, tok, msgs, temp=0.0, prefix=pre,
                                     logits_processors=logits_processors)
                total_in += ni; total_out += no
                pa = parse_action(t)
                key = (pa.get("action"), pa.get("region"), pa.get("bug_class"))
                if key in seen:
                    continue
                seen.add(key); cands.append((t, pa))
            pick = next((c for c in cands if c[1].get("action") == "hypothesize"
                         and c[1].get("region") == task["symptom_region"]), None)
            cand_meta = {"k": len(cands), "structural": True,
                         "regions_offered": sorted({str(c[1].get("region")) for c in cands}),
                         "greedy_region": gp.get("region"),
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
                    help="G-TRIVIAL (理: force the first hypothesis to the SYMPTOM region. "
                         "A one-line rule learned from the agent's own history and the control G "
                         "must beat — it needs no model, no head and no activations.")
    ap.add_argument("--candidate-select", default="", choices=["", "symptom"],
                    help="G-TRIVIAL-PRIME (理, candidate set structural per 10909): build one "
                         "candidate per visible region by teacher-forcing the opening fields, then "
                         "pick the candidate whose region is the symptom region. Separates WHERE "
                         "from HOW: unlike --trivial-symptom the hypothesis TEXT is still the "
                         "model's own. The fallback to greedy now fires only if a region produced "
                         "no parseable hypothesis, not because the sampler never proposed it.")
    ap.add_argument("--cand-k", type=int, default=8,
                    help="cap on the STRUCTURAL candidate set: greedy plus one teacher-forced "
                         "candidate per visible region, so k = len(regions) + 1 unless that "
                         "exceeds this, in which case regions are kept by the model's own "
                         "probability on the first token of the region name. v1 and v2 have four "
                         "regions, so the cap is untaken on both.")
    ap.add_argument("--head", help="npz from et8_head_v3.py fit (w, b, mu, sd) -- mechanism G")
    ap.add_argument("--head-layer", type=int, default=18)
    ap.add_argument("--inspect-first", action="store_true",
                    help="require ONE inspect before the first hypothesis (理. The region is "
                         "the model's own greedy choice, not the symptom region; the point is that "
                         "the hypothesis the head reads is made with something OBSERVED in the "
                         "prompt. Costs one action per episode by construction -- printed, not "
                         "barred.")
    ap.add_argument("--redact-symptom", action="store_true",
                    help="drop the TEST NAME from the symptom line, keeping the failure kind and "
                         "the assertion message. v1's ten symptom strings each map to exactly one "
                         "bug region, so the name is a lookup key worth 100%% localisation; this "
                         "arm measures how much of the baseline rests on it.")
    a = ap.parse_args()
    HEAD_STATE = None
    if a.head:
        import numpy as _np, mlx.core as _mx
        import et8_head_v3 as _H
        _z = _np.load(a.head)
        HEAD_STATE = {"w": _z["w"], "b": float(_z["b"][0]), "mu": _z["mu"], "sd": _z["sd"],
                      "layer": a.head_layer, "hidden": _H.hidden_at, "mx": _mx}
        print("  G head loaded: layer %d, trained on tasks < %d"
              % (a.head_layer, int(_z["train_task_max"][0])), file=sys.stderr)
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
                            redact_symptom=a.redact_symptom,
                            inspect_first=a.inspect_first, head_state=HEAD_STATE)
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
