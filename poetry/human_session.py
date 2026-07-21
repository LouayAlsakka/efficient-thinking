#!/usr/bin/env python
"""ET-IV item 6 — human-rating session harness (BUILD, DON'T RUN; the human arm waits for G2).

Everything a real rating session needs, testable end-to-end on synthetic data:
  - blind A/B pair + ranking form generation (randomized order, model identity stripped);
  - Pareto pruning of dominated candidates BEFORE any human sees them (non-compensatory per the
    ET-IV amendment: a candidate dominated on every craft/taste dimension is never shown);
  - a session logging schema (choices, timestamps, optional per-dimension scores + rationale);
  - a week-later re-rate subset sampler (self-consistency ceiling);
  - q (selector-vs-rater agreement) and self-consistency computation.
Definition of done: a synthetic dry-run drives the real pipeline with a fake rater and produces a q
number. Timestamps are injected by the caller (kept out of the library so runs are reproducible/testable).
"""
import json, random


def strip_identity(candidates):
    """Return display copies with model identity removed (blind rating)."""
    return [{"cand_id": c["cand_id"], "text": c["text"]} for c in candidates]


def pareto_prune(candidates, dims):
    """Drop candidates dominated on EVERY dimension by some other candidate (non-compensatory).
    dims: list of keys into c['scores']; higher = better. Returns the Pareto set (what a rater sees)."""
    keep = []
    for a in candidates:
        dominated = any(b is not a and all(b["scores"][d] >= a["scores"][d] for d in dims)
                        and any(b["scores"][d] > a["scores"][d] for d in dims) for b in candidates)
        if not dominated:
            keep.append(a)
    return keep


def make_ab_pairs(candidates, rng):
    """Blind A/B pairs over the Pareto set (round-robin), order randomized, identity stripped."""
    pairs = []
    cs = strip_identity(candidates)
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = (cs[i], cs[j]) if rng.random() < 0.5 else (cs[j], cs[i])
            pairs.append({"pair_id": f"{a['cand_id']}_vs_{b['cand_id']}", "A": a, "B": b})
    rng.shuffle(pairs)
    return pairs


def log_choice(pair, winner, ts, dims=None, rationale=None):
    """Session log record schema. winner in {'A','B'}. dims/rationale optional (amendment format)."""
    return {"pair_id": pair["pair_id"], "chosen_cand": pair[winner]["cand_id"], "winner": winner,
            "ts": ts, "per_dimension": dims or {}, "rationale": rationale or ""}


def rerate_subset(pairs, frac, rng):
    """Sample a subset of pairs to re-rate a week later (self-consistency ceiling)."""
    k = max(1, int(len(pairs) * frac))
    return rng.sample(pairs, k)


def compute_q(selector_choices, rater_log):
    """q = fraction of pairs where the selector's preferred candidate matches the rater's choice."""
    rater = {r["pair_id"]: r["chosen_cand"] for r in rater_log}
    n = ok = 0
    for pid, sel_cand in selector_choices.items():
        if pid in rater:
            n += 1; ok += (sel_cand == rater[pid])
    return round(ok / n, 3) if n else None


def self_consistency(rater_log, rerate_log):
    """Agreement between a rater's first and second (week-later) choices on the re-rate subset."""
    first = {r["pair_id"]: r["winner"] for r in rater_log}
    n = ok = 0
    for r in rerate_log:
        if r["pair_id"] in first:
            n += 1; ok += (r["winner"] == first[r["pair_id"]])
    return round(ok / n, 3) if n else None


def _dry_run():
    """Synthetic end-to-end: fake candidates + a fake rater through the real pipeline -> a q number."""
    rng = random.Random(0)
    cands = [{"cand_id": f"c{i}", "text": f"poem {i}", "model": f"m{i%3}",
              "scores": {"meter": rng.random(), "rhyme": rng.random(), "taste": rng.random()}}
             for i in range(8)]
    pareto = pareto_prune(cands, ["meter", "rhyme", "taste"])
    pairs = make_ab_pairs(pareto, rng)
    # fake rater prefers higher 'taste'; selector prefers higher 'meter' (so q should be < 1)
    score = {c["cand_id"]: c["scores"] for c in pareto}
    rater_log = [log_choice(p, "A" if score[p["A"]["cand_id"]]["taste"] >= score[p["B"]["cand_id"]]["taste"]
                            else "B", ts=i) for i, p in enumerate(pairs)]
    selector = {p["pair_id"]: (p["A"]["cand_id"] if score[p["A"]["cand_id"]]["meter"] >=
                               score[p["B"]["cand_id"]]["meter"] else p["B"]["cand_id"]) for p in pairs}
    rr = rerate_subset(pairs, 0.5, rng)
    rerate_log = [log_choice(p, rng.choice(["A", "B"]), ts=100 + i) for i, p in enumerate(rr)]
    q = compute_q(selector, rater_log)
    sc = self_consistency(rater_log, rerate_log)
    assert 0.0 <= q <= 1.0 and len(pareto) <= len(cands) and pairs, "pipeline invariants failed"
    print(f"[human-session dry-run] candidates={len(cands)} -> pareto={len(pareto)} -> pairs={len(pairs)}; "
          f"q(selector vs fake rater)={q}  self-consistency={sc}")
    print("[human-session] DoD met: fake data produced a q number through the real pipeline. BUILD-ONLY.")


if __name__ == "__main__":
    _dry_run()
