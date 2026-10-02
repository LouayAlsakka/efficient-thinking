#!/usr/bin/env python
"""ET-8b §13b — the scorer. READINGS FIXED BY 理 BEFORE ANY ARM RAN (12417, form ruled 12436).

    g1  − g0           the confirmatory replication, scored under §14
    g1' − g1-matched   INDEPENDENCE AT EQUAL ROWS — the §13b question. Prediction, registered:
                       gen1' > gen1-matched, both intervals excluding zero and clearing the
                       largest within-head-arm draw
    within-arm sd for every head arm against base — "the head stabilises the run"

    g1 − g1-matched    DROPPED, not reported (12436): g1 is three per-decision heads on 2,824 rows
                       and g1-matched is one shared head on 1,437, so that difference varies FORM
                       and AMOUNT at once. This file refuses to print it even if asked.

§14 (理) IS TWO CONDITIONS AND BOTH ARE CHECKED SEPARATELY, because a pairing can pass the
first and fail the second and the difference is the whole point:
    (a) the 95% interval excludes zero
    (b) its LOWER BOUND exceeds the largest within-head-arm draw of this session
A reading that reports "significant" on (a) alone is the §11 mistake: the loop's own arm-to-arm
spread is wide, and an effect smaller than that spread is a draw from it.

⚠️ GAME 1 IS ASYMMETRIC IN TIME and this file says so in its own output. Its b/g0/g1 ran before
理's form ruling and its g1p/g1m about three hours later; games 2 and 3 have all five arms adjacent.
A game-1-only effect is confounded with that gap.

McNemar is a WITHIN-RUN DESCRIPTIVE on loop rows only (理's demotion). It is computed and printed
and it decides nothing.
"""
import argparse, itertools, json, math, os, statistics, subprocess, sys

# THE ONE PREREQUISITE, CHECKED HERE SO A READER GETS A SENTENCE INSTEAD OF A TRACEBACK.
# This is the command the challenge page advertises as "seconds, no model, no GPU" — and it shells
# out to paired_stats.py, which needs numpy for the bootstrap. A stranger with a fresh clone and a
# stock Python otherwise meets `ModuleNotFoundError: No module named 'numpy'` on their FIRST command,
# which is a worse first impression than any result in the table. The page can state the
# prerequisite too; this way it does not have to remember to.
try:
    import numpy as _numpy_probe          # noqa: F401  (imported for the check, used by paired_stats)
except ImportError:                       # pragma: no cover - the reader-facing path
    sys.exit("This scorer needs numpy (paired_stats.py uses it for the paired bootstrap).\n"
             "    pip install numpy\n"
             "Nothing else is required: no model, no GPU, no network.")

HERE = os.path.dirname(os.path.abspath(__file__))
# TWO PLANS, ONE FILE. 13c has different arms from 13b and the same readings; a copy of this
# script with four names changed is how two files drift into disagreeing about what §14 means.
# The plan names the arms, their labels, the pairings to score, and the pairings that are
# FORBIDDEN because they are confounded -- the refusal travels with the plan rather than living
# in whichever copy remembered it.
ARMS = ("b", "g0", "g1", "g1p", "g1m")
# GAME 1'S ARMS ARE ON DISK UNDER THEIR ORIGINAL NAMES and are NOT renamed here: three of them ran
# before 理's form ruling under a chain that named them b1x/g0x/g1x, and the two late ones under
# g1px1/g1mx1. Renaming files a finished chain wrote is how provenance gets lost; the alias table
# is declared, printed, and carried into the artifact instead.
ALIASES = {("b", "1"): "b1x", ("g0", "1"): "g0x", ("g1", "1"): "g1x",
           ("g1p", "1"): "g1px1", ("g1m", "1"): "g1mx1"}

ARM_LABEL = {"b": "base", "g0": "gen0 head", "g1": "gen1 head (published, 3 per-decision, 2824 rows)",
             "g1p": "gen1' (SQL, one shared head, 1437 rows)",
             "g1m": "gen1-matched (gen1 subsampled to 1437, one shared head)"}
FORBIDDEN = {("g1", "g1m")}   # 理: confounded, dropped, not reported

PLANS = {
    "13b": {"base": "b", "arms": ARMS, "labels": None, "aliases": None,
            "wanted": [("g0", "b"), ("g1", "b"), ("g1p", "b"), ("g1m", "b"),
                       ("g1", "g0"), ("g1p", "g1m")],
            "forbidden": FORBIDDEN,
            "note": ("the g1p - g1m pair is CONFOUNDED BY THE MATCHING UNIT: g1p is an intact "
                     "collection and g1m was cut to a row budget, which shreds decision groups. "
                     "理 ruled it printed as confounded rather than re-run, because the confound "
                     "worked AGAINST g1m and it still won -- so the conclusion is conservative.")},
    # VIII-c arm 1. Registered BEFORE the arms finished collecting, which is the whole point of
    # writing it here rather than after: the pairings and the note are fixed while the numbers do
    # not exist yet, so the reading cannot be shaped by what came out.
    # ─── VIII-e arm 1 ────────────────────────────────────────────────────────────
    # The baselines VIII owes: does the frozen head's gain survive against the two
    # objects a reviewer names first -- a head fitted on what the agent ALREADY
    # believed, and prompt retrieval with no head at all?
    #
    # REGISTERED BEFORE ITS ARMS FINISHED, as above. Disclosure of what had been
    # seen when this was written: the two R1 heads' OFFLINE cross-validated fits
    # (83.4% for the agent's-pick label against 60.4% for the verifier's, on
    # identical rows) and two finished base games' green counts. No pairing and no
    # arm of (b) existed. The offline gap is deliberately NOT a reading here -- see
    # the note -- and a prediction about which way it converts was published before
    # these games ran.
    #
    # R1 IS SCORED LIKE FOR LIKE on the 169 decision groups where the agent's-pick
    # label VARIES within the group. It is all-zero in the other 121 of 290 because
    # the agent's pick is recorded on only 58% of rows, and a group with no positive
    # can never be got right under pick-one-per-group scoring -- so including them
    # would depress the baseline for a reason unrelated to the agent's pick being a
    # weak signal. Both R1 heads are fitted on those same 169 groups, the same 837
    # rows in the same order, with the same 169 positives: the arms differ ONLY in
    # which candidate the label marks. The 121 are reported unscoreable by count.
    #
    # The anchor arm is NOT re-run for this plan: anchor_f132 is the same head on the
    # same task set in the same session as the coverage-or-readability arm, and its
    # three games are byte-identical copies (sha256 verified) rather than a second
    # collection. Re-running it would spend three games to produce a second copy of
    # a number the session already has.
    "viiie": {"base": "base_g",
              "arms": ("base_g", "anchor_f132_g", "retr_g",
                       "a169_verifier_g", "a169_agentpick_g"),
              "labels": {
                  "base_g": "base: no head, no retrieval",
                  "anchor_f132_g": "the published half-size forced head, in-session",
                  "retr_g": "(b) ExPEL-style prompt retrieval, k=1, ONCE PER DECISION, "
                            "no head, retrieval tokens charged in the turn's own input count",
                  "a169_verifier_g": "R1 reference: verifier-label head on the 169 groups",
                  "a169_agentpick_g": "(a) the agent's own revealed pick, same 169 groups"},
              "aliases": {},
              "wanted": [("a169_verifier_g", "a169_agentpick_g"),
                         ("anchor_f132_g", "retr_g"),
                         ("retr_g", "base_g"),
                         ("anchor_f132_g", "base_g"),
                         ("a169_verifier_g", "base_g"),
                         ("a169_agentpick_g", "base_g")],
              "forbidden": set(),
              "note": ("R1/R4 read off a169_verifier vs a169_agentpick -- like for like on the "
                       "169 varying groups. R2/R3 read off anchor_f132 vs retr. The CV column "
                       "is EXCLUDED for every arm, and for (a) that exclusion is load-bearing "
                       "rather than procedural: its label is recoverable at 67.5% from EIGHT "
                       "principal components against 33.7% for the verifier's, because the "
                       "agent chose its pick FROM the state the head reads -- so its 83.4% "
                       "offline fit is close to self-prediction and says nothing about in-loop "
                       "value. This programme has six recorded probe non-conversions; reading "
                       "the offline gap as a result would be the seventh.")},
    # ─── VIII-d arm 1 ────────────────────────────────────────────────────────────
    # COVERAGE or READABILITY: does the forced-look advantage come from the head
    # seeing more of the states the game later visits, or from post-inspection
    # states being more linearly separable?
    #
    # REGISTERED BEFORE THE ARMS FINISHED COLLECTING, for the reason this file
    # already gives above: the pairings are fixed while the numbers do not exist.
    # DISCLOSURE, so a reader can judge that for themselves: when this entry was
    # written, games 2-9 were still running and I had seen exactly one number from
    # one finished game -- S_cov game 1's green rate. No pairing, no comparison and
    # no other arm existed yet. That one rate is the whole of what could have
    # shaped this, and it is named here rather than left for someone to wonder about.
    #
    # FOUR arms, and the fourth is the reason the other three can be read at all:
    # the bar is "the session's largest within-head-arm range", a WITHIN-session
    # quantity, while the registered anchor is a rate published in a DIFFERENT
    # session. This file's own viiic7 note warns that doing so "would mix two
    # sessions in the within-arm ranges". So the anchor's own head is RE-RUN here,
    # three games, and the anchor and the bar then share one session.
    #
    # S_cov_mix is the control that makes the headline claim licit: S_cov and
    # S_read are matched in SIZE but nearly mirrored in decision point (S_cov takes
    # 57.3% of the pool's d1 groups and 25.7% of its d2; S_read 30.1% and 57.4%),
    # so a difference between THEM is partly a decision-mix difference. S_cov_mix
    # is the coverage objective constrained to S_read's exact per-decision counts,
    # so S_cov_mix vs S_read differs in the OBJECTIVE and in nothing else. The
    # coverage-versus-readability claim is read off THAT pair; S_cov vs S_read is
    # reported beside it as the unconstrained comparison and is NOT the headline.
    # ── AMENDMENT 5/6's FIFTH ARM, added to this harness AFTER its games ran ──────
    # Stated plainly because it is the one thing a reader should check here: the ARM
    # and the three pairings below were added to this file after the three rand132
    # games finished. The READING was not. It is registered verbatim in the series
    # plan at `b44fee8` (amendment 6, before any rand132 game), which fixes it as
    # amendment 5 wrote it before the draw:
    #
    #   "Reading R5: if any selected head beats `rand132` under both criteria,
    #    selection by that rule buys something inside one pool; if none does, the
    #    +3.3 to +8.0 above was the pool, not the selection."
    #
    # So the pairings are the three SELECTED heads against the random baseline, and
    # nothing else: the anchor is a published forced-look head and not a selection
    # rule, so anchor-vs-rand132 is NOT part of R5 and is not scored here. Adding it
    # would be choosing a comparison after seeing the arms, which is the whole thing
    # the registration exists to prevent.
    #
    # The arm played is the STRATIFIED draw (uniform within each decision point,
    # proportional across them, 47/46/39). The seed-0 UNIFORM draw is retained in the
    # lock with its z-scores and was deliberately not played -- it came out 2.68 sd
    # low on decision-1 groups, which would have given the baseline one selected arm's
    # decision mix and not the other's.
    #
    # NOTE A CONSEQUENCE, since §14b's bar is a WITHIN-SESSION quantity: a fifth arm
    # can RAISE "the session's largest within-head-arm range" and so make a pairing
    # that previously cleared §14b stop clearing it. That is correct behaviour and not
    # a regression, but it means the four-arm numbers and the five-arm numbers are not
    # interchangeable, and this file prints the bar it used.
    "viiid": {"base": "anchor_f132_g",
              "arms": ("anchor_f132_g", "S_cov_g", "S_read_g", "S_cov_mix_g",
                       "rand132_strat_g"),
              "labels": {
                  "anchor_f132_g": "anchor: the published half-size forced head, re-run in "
                                   "THIS session so the anchor and the bar share one",
                  "S_cov_g": "S-cov: 132 groups maximising CENTRED nearest-neighbour overlap "
                             "with a fourth, never-scored game on a disjoint task set",
                  "S_read_g": "S-read: the 132 groups with the widest CROSS-FITTED margin "
                              "under the anchor head (cross-fitted because the anchor was "
                              "fitted on an unknown third of the pool)",
                  "S_cov_mix_g": "S-cov constrained to S-read's exact per-decision counts -- "
                                 "differs from S-read in the objective alone",
                  "rand132_strat_g": "rand132 STRATIFIED: 132 groups uniform WITHIN each "
                                     "decision point and proportional across them (47/46/39), "
                                     "from the same cut290 -- the random baseline R5 reads "
                                     "selection against"},
              "aliases": {},
              "wanted": [("S_cov_mix_g", "S_read_g"),
                         ("S_cov_g", "S_read_g"),
                         ("S_cov_g", "anchor_f132_g"),
                         ("S_read_g", "anchor_f132_g"),
                         ("S_cov_mix_g", "anchor_f132_g"),
                         # R5, as registered: each SELECTED head against the random
                         # baseline drawn from the same pool. Three pairings, no more.
                         ("S_cov_g", "rand132_strat_g"),
                         ("S_read_g", "rand132_strat_g"),
                         ("S_cov_mix_g", "rand132_strat_g")],
              # S_cov vs S_read is CONFOUNDED by decision mix but is informative and is
              # reported with that said, which is this file's own convention for a
              # confounded pair -- so `forbidden` is empty rather than hiding it.
              "forbidden": set(),
              "note": ("The HEADLINE pair is S_cov_mix vs S_read -- objective alone. S_cov vs "
                       "S_read is the unconstrained comparison and is confounded by decision "
                       "mix; it is reported, never used for the attribution. The CV column of "
                       "every arm is EXCLUDED from the reading: S_read was selected for wide "
                       "margins so its CV is inflated by construction. Game scores, which no "
                       "selection touches, are the comparison.")},
    "viiic": {"base": "v_b", "arms": ("v_b", "v_g0", "v_g1m", "v_vb"),
              "labels": {"v_b": "base",
                         "v_g0": "gen0 head (per-decision -- the form gen0's own rule selected)",
                         "v_g1m": "gen1-matched, SHARED head, 290 decisions, seed 13731",
                         "v_vb": "agent B's visited places, SHARED head, 290 decisions, seed 13731"},
              "aliases": {},
              "wanted": [("v_g0", "v_b"), ("v_g1m", "v_b"), ("v_vb", "v_b"), ("v_vb", "v_g1m")],
              "forbidden": set(),
              "note": ("v_vb - v_g1m IS the arm: both are the SHARED form at the same decision "
                       "count and the same seed, and they differ in WHOSE TRAJECTORY the 7B's "
                       "states were taken at. The labels are the verifier's ground truth in both, "
                       "so no agent supplies supervision and the escape is in which states get "
                       "visited. v_g0 is per-decision because that is its own rule's form and it "
                       "is the published comparison, not a second knob. Read beside the probe "
                       "line (B 57.9%% CV, chance 20.4, permutation at chance, PCA-8 37.6) and "
                       "REMEMBER the probe has failed to convert four times in this programme.")},
    # VIII-c, THE SIX-ARM TABLE. Registered while the union arm's game 3 is at 241 of 300 --
    # before its last number exists, which is the only time a reading is worth writing down.
    #
    # WHAT IS NEW HERE AND NOTHING ELSE IS: two head arms join the four the lock already carries.
    #   v_fi  the forced-inspection agent's trajectories, SHARED head, 290 decisions, seed 13731
    #   v_un  B's and the forced agent's trajectories MERGED, then cut to 290 WHOLE groups,
    #         seed 13731 -- 1,424 rows, drawn as B 158 groups / 773 rows + forced 132 / 651.
    # Every head arm plays the games identically (--head, no forcing at play time), so v_vb, v_fi
    # and v_un differ in WHOSE TRAJECTORY the 7B's states were taken at and in nothing else. The
    # bracket (v_b, v_g0, v_g1m) is NOT re-run for them: re-running would give the late arms a
    # fresh control the early ones never had and would mix two sessions in the within-arm ranges.
    #
    # THE BAR MOVES AND THAT IS NOT A BUG. §14(b) is "the largest within-head-arm draw OF THIS
    # SESSION". The session now has five head arms instead of three, so the bar computed here can
    # be LARGER than the four-arm table's 3.67. If a §14(b) verdict the lock recorded at 3.67
    # fails at the wider bar, it is reported as FAILING -- the denominator is the loop's own
    # instability and more arms measure more of it. The lock's own numbers are not edited; this
    # plan's table is the wider reading and says so.
    #
    # THE DECISION RULE FOR v_un, written down before the head was fitted, not after the games:
    #   above BOTH single-source arms by more than the bar  -> the two experiences compound
    #   within the bar of the better single source          -> B adds nothing to a forced look
    #   below both                                          -> mixing two acquisition policies is
    #                                                          worse than either (no story for it)
    # Read beside the probes, which disagree with the ordering to be tested: B 57.9% CV, forced
    # 54.8%, union 53.8% (chance 20.4, every permutation at chance, PCA-8 above chance). This
    # programme has five probe non-conversions and one that pointed the wrong way.
    "viiic6": {"base": "v_b",
               "arms": ("v_b", "v_g0", "v_g1m", "v_vb", "v_fi", "v_un"),
               "labels": {"v_b": "base",
                          "v_g0": "gen0 head (per-decision -- the form gen0's own rule selected)",
                          "v_g1m": "gen1-matched, SHARED head, 290 decisions, seed 13731",
                          "v_vb": "agent B's visited places, SHARED head, 290 decisions, seed 13731",
                          "v_fi": "forced-inspection agent's places, SHARED head, 290 decisions, seed 13731",
                          "v_un": "B + forced MERGED, groups kept apart, SHARED head, 290 whole groups, seed 13731"},
               "aliases": {},
               "wanted": [("v_g0", "v_b"), ("v_g1m", "v_b"), ("v_vb", "v_b"),
                          ("v_fi", "v_b"), ("v_un", "v_b"),
                          ("v_vb", "v_g1m"), ("v_fi", "v_vb"),
                          ("v_un", "v_vb"), ("v_un", "v_fi")],
               "forbidden": set(),
               "note": ("v_un - v_fi and v_un - v_vb ARE the arm; v_fi - v_vb is the §16c "
                        "question (is a cheap forced look worth as much as an agent's own "
                        "selection) and v_un - v_b is only the anchor. The merge keyed decision "
                        "groups on (SOURCE, task_id, decision): both sources ran the same 300 "
                        "tasks, so an unprefixed concatenation would have collapsed two "
                        "trajectories' decision 1 for one task into a single group and the cut "
                        "would have drawn rows from both -- the §13c defect arriving through a "
                        "merge instead of a subsample. Row counts differ a few percent across "
                        "these heads (B 1,421 at 290 decisions, union 1,424) because groups vary "
                        "in length; the match is on the unit the head is SCORED over, which is "
                        "decisions, not rows.")},
    # VIII-c, THE SEVEN-ARM TABLE — the discriminator's plan, registered while arm 4 is at 149 of
    # 900 episodes and none of its three games has a number.
    #
    # v_f132 is the union cut's OWN 132 forced decision groups, alone, no B: the same groups, not a
    # re-draw at the same seed, so the only thing that differs from the union arm is the presence of
    # B's 158 groups. Head fitted before the games: CV 45.5% (n=132, chance 20.3), permutations
    # 18.2-25.8, PCA-8 30.3, sub 42.4 -- the lowest probe of the four heads, recorded here because a
    # probe that predicted the ordering would be the first probe in this programme to do so.
    #
    # THE THREE OUTCOMES, FIXED BEFORE THE NUMBER EXISTS (they are a three-way comparison against
    # arms already on disk, so no new bracket is run and the within-arm ranges stay one session):
    #   v_f132 ~ v_fi   (41.22)  -> the forced arm SATURATES at 132 groups; the union arm cannot
    #                              speak about B in either direction, and the six-arm reading
    #                              "B adds nothing" is NOT supported by it
    #   v_f132 ~ v_un   (38.56)  -> B's 158 groups neither add nor subtract: "B adds nothing" stands
    #   v_f132 ~ v_vb   (35.44)  -> the B half CARRIED the union up from 132 forced groups; B
    #                              contributes, and the six-arm reading is wrong
    # A result between two of those anchors is reported as between them, not rounded to the nearer.
    #
    # THE BAR CAN MOVE AGAIN and the same rule applies as in viiic6: it is the largest within-HEAD-arm
    # range of the session, now over six head arms. If v_f132's own range exceeds 3.67 the bar rises
    # and every (b) verdict is re-read against the wider bar, including ones the lock records as
    # passing. The denominator is the loop's instability and more arms measure more of it.
    "viiic7": {"base": "v_b",
               "arms": ("v_b", "v_g0", "v_g1m", "v_vb", "v_fi", "v_un", "v_f132"),
               "labels": {"v_b": "base",
                          "v_g0": "gen0 head (per-decision -- the form gen0's own rule selected)",
                          "v_g1m": "gen1-matched, SHARED head, 290 decisions, seed 13731",
                          "v_vb": "agent B's visited places, SHARED head, 290 decisions, seed 13731",
                          "v_fi": "forced-inspection agent's places, SHARED head, 290 decisions, seed 13731",
                          "v_un": "B + forced MERGED, groups kept apart, SHARED head, 290 whole groups, seed 13731",
                          "v_f132": "the union cut's OWN 132 forced groups, alone -- the discriminator"},
               "aliases": {},
               "wanted": [("v_g1m", "v_b"), ("v_vb", "v_b"), ("v_fi", "v_b"), ("v_un", "v_b"),
                          ("v_f132", "v_b"),
                          ("v_fi", "v_f132"), ("v_un", "v_f132"), ("v_f132", "v_vb"),
                          ("v_un", "v_fi"), ("v_un", "v_vb"), ("v_fi", "v_vb")],
               "forbidden": set(),
               "note": ("v_fi - v_f132 is THE pairing: same source, same form, same seed, 290 "
                        "decisions against 132, and it is the only thing in this programme that can "
                        "separate saturation of the forced arm from a contribution by the union's B "
                        "half. v_un - v_f132 is the same question asked from the other side: it is "
                        "exactly the 158 B groups, added to a fixed 132 forced ones, at a budget "
                        "that is NOT matched -- which is the comparison the matched-budget union "
                        "arm could not make. Read both beside the probe ordering, which is inverted "
                        "against the games everywhere it has been checked in this table.")},
    "13c": {"base": "c_b", "arms": ("c_b", "c_g0", "c_g1m", "c_g1pp"),
            "labels": {"c_b": "base", "c_g0": "gen0 head",
                       "c_g1m": "gen1-matched (290 decisions, whole groups)",
                       "c_g1pp": "gen1'' — disjoint SAME-FAMILY problems (290 decisions)"},
            "aliases": {},
            "wanted": [("c_g0", "c_b"), ("c_g1m", "c_b"), ("c_g1pp", "c_b"),
                       ("c_g1pp", "c_g1m")],
            "forbidden": set(),
            "note": ("c_g1pp - c_g1m is the arm: same family, same form, same DECISION count, "
                     "origin varied. Both heads were cut on whole decision groups, so neither is "
                     "shredded. Read beside the probe line (60.7 vs 53.4) and beside the fact that "
                     "the disjoint rows are 1.3% byte-identical to gen0's but 42% "
                     "neighbour-redundant with them.")},
}


def green_rate(path):
    n = g = 0
    for line in open(path):
        e = json.loads(line)
        n += 1
        g += bool(e.get("green"))
    return g, n


def pair(base_path, g_path, name, out_json, py=sys.executable, expect=300):
    cmd = [py, os.path.join(HERE, "paired_stats.py"), "--base", base_path, "--g", g_path,
           "--expect", str(expect), "--name", name, "--out", out_json]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("STOP: paired_stats failed on %s\n%s" % (name, (r.stderr or "")[-800:]))
    return json.load(open(out_json))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of <arm><game>.episodes.jsonl")
    ap.add_argument("--games", default="1,2,3")
    ap.add_argument("--plan", default="13b", choices=sorted(PLANS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    plan = PLANS[a.plan]
    arms_used = plan["arms"]
    aliases = plan["aliases"] if plan["aliases"] is not None else ALIASES
    labels = plan["labels"] or ARM_LABEL
    games = [g.strip() for g in a.games.split(",") if g.strip()]
    # A MEASUREMENT MUST NOT WRITE INTO THE THING IT MEASURES. This defaulted to
    # `<dir>/_pairs` -- INSIDE the published, hash-locked results directory -- so every reader who
    # ran the one-command reproduction added 33 scratch files to the artefact they were verifying,
    # and `git status` in a fresh clone showed untracked files inside it. The per-pair JSONs are
    # scratch: nothing printed depends on them. Sibling, not child.
    work = a.work or (a.dir.rstrip(os.sep) + "_pairs")
    os.makedirs(work, exist_ok=True)

    # ---- what is on disk, stated before anything is computed ------------------------------
    have, rates = {}, {}
    for g in games:
        for arm in arms_used:
            stem = aliases.get((arm, g), "%s%s" % (arm, g))
            p = os.path.join(a.dir, "%s.episodes.jsonl" % stem)
            if not os.path.exists(p):
                p = os.path.join(a.dir, "g13b_%s.episodes.jsonl" % stem)
            if os.path.exists(p):
                have[(arm, g)] = p
                rates[(arm, g)] = green_rate(p)
    print("  plan %s — arms found: %d of %d" % (a.plan, len(have), len(arms_used) * len(games)))
    for g in games:
        row = "  game %s: " % g
        for arm in arms_used:
            k = (arm, g)
            row += "%s %s  " % (arm, ("%d/%d" % rates[k]) if k in rates else "—")
        print(row)

    # ---- within-arm sd, ACROSS GAMES, per arm -------------------------------------------
    # This is the denominator §14(b) is measured against, so it is computed from the data and
    # never from a remembered figure.
    within = {}
    for arm in arms_used:
        pts = [100.0 * rates[(arm, g)][0] / rates[(arm, g)][1] for g in games if (arm, g) in rates]
        if len(pts) >= 2:
            within[arm] = {"draws": [round(p, 2) for p in pts],
                           "mean": round(statistics.mean(pts), 2),
                           "sd": round(statistics.stdev(pts), 3),
                           "range": round(max(pts) - min(pts), 2)}
    # THE BASE ARM IS NAMED, NOT GUESSED FROM ITS SUFFIX. This line used to read
    # `not x.endswith("b")`, which was true of the base arms of the two earlier plans ("b",
    # "c_b") and silently ALSO excluded VIII-c's B-experience head, `v_vb` -- the one arm under
    # test. The bar came out 2.0 (the largest range among the two arms left) instead of 3.67, and
    # a recorded verdict moved on it. A second reader caught the arithmetic; the cause was that
    # the arm had been renamed to dodge a case-insensitive filename collision and the new name
    # ended in the letter the heuristic keyed on.
    base_arm = plan.get("base")
    if base_arm is None:
        raise SystemExit("plan %r does not name its base arm; the §14b bar cannot be computed "
                         "from a suffix guess" % a.plan)
    head_arms = [x for x in arms_used if x in within and x != base_arm]
    largest_head_draw = max((within[x]["range"] for x in head_arms), default=None)
    if largest_head_draw is None:
        print("  ⚠️ §14(b) CANNOT BE EVALUATED YET: no head arm has two games on disk. The bar is "
              "the largest within-head-arm draw of THIS session and it is not measurable from one "
              "game. Every (b) verdict below reads 'not evaluable', never 'passed'.")

    # ---- the pairings ---------------------------------------------------------------------
    wanted = plan["wanted"]
    results = {}
    for hi, lo in wanted:
        if (hi, lo) in plan["forbidden"]:
            continue
        for g in games:
            if (hi, g) not in have or (lo, g) not in have:
                continue
            # the PLAN names the section, not the filename — a 13c pairing printed as "§13b"
            # would mislabel the artefact in the one field a reader uses to find it
            name = "§%s game %s: %s vs %s" % (a.plan, g, hi, lo)
            j = pair(have[(lo, g)], have[(hi, g)], name,
                     os.path.join(work, "%s%s_vs_%s%s.json" % (hi, g, lo, g)), py=a.python)
            ci = j["success_95CI"]
            excl0 = ci[0] > 0.0
            clears = None if largest_head_draw is None else ci[0] > largest_head_draw
            results["%s_vs_%s_game%s" % (hi, lo, g)] = {
                "delta_points": j["success_delta_points"], "ci": ci,
                "base_pct": j["base_success_pct"], "g_pct": j["G_success_pct"],
                "§14a_interval_excludes_zero": excl0,
                "§14b_lower_bound_clears_largest_within_head_draw": (
                    "not evaluable — needs two games of a head arm" if clears is None else clears),
                "§14b_bar_points": largest_head_draw,
                "mcnemar_DESCRIPTIVE_ONLY": j.get("McNemar"),
            }
            print("  %-34s %+5.1f  [%+.1f, %+.1f]  §14a %-5s §14b %s"
                  % (name, j["success_delta_points"], ci[0], ci[1], excl0,
                     results["%s_vs_%s_game%s" % (hi, lo, g)]
                     ["§14b_lower_bound_clears_largest_within_head_draw"]))

    out = {
        "document": "ET-8b §13b — scored under the readings 理 fixed before any arm ran",
        "readings": {
            "g1_minus_g0": "the confirmatory replication, §14 (a) AND (b)",
            "g1p_minus_g1m": ("INDEPENDENCE AT EQUAL ROWS — the §13b question. Registered "
                              "prediction: gen1' > gen1-matched, both intervals excluding zero and "
                              "clearing the largest within-head-arm draw."),
            "within_arm_sd": "the head stabilises the run",
            "g1_minus_g1m": "DROPPED (理) — confounds form with amount. Not computed here."},
        "§14": {"a": "the 95% interval excludes zero",
                "b": "its lower bound exceeds the largest within-head-arm draw of this session",
                "bar_points": largest_head_draw,
                "why_both": ("a pairing can pass (a) and fail (b); an effect smaller than the "
                             "loop's own arm-to-arm spread is a draw from that spread")},
        "⚠️_game_1_is_asymmetric_in_time": (
            "game 1's b/g0/g1 ran before 理's form ruling and its g1p/g1m about three hours later; "
            "games 2 and 3 have all five arms adjacent. A game-1-only effect is confounded with "
            "that gap."),
        "mcnemar": "within-run descriptive on loop rows only (理's demotion). Decides nothing.",
        "plan": a.plan,
        "plan_note": plan["note"],
        "arm_labels": labels,
        "file_aliases": {"%s game %s" % k: v for k, v in aliases.items()},
        "green_counts": {"%s%s" % (k[0], k[1]): {"green": v[0], "n": v[1]} for k, v in rates.items()},
        "within_arm": within,
        "pairings": results,
        "signed": "Sautee (sha-ta)"}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("\n  wrote %s" % a.out)


if __name__ == "__main__":
    main()
