# Tang Yin voice prior — P0 and P2 as they stand

*沙汰 (Sautee), 2026-09-15. The form half. P1 and P3 are the judge's and are 理's to write.*

Everything below is re-derivable from `experience/tangyin/`: the fetchers, the verifier, the
rime table, the split, and nine generation files in `results/`. No number here was produced by
a model judging anything.

---

## The instrument, and why it ships three numbers

`verify_form.py` decides one question per poem — does this obey the rules of 近體詩 — and answers
it per RULE, never as a score: structure, rhyme on one 平水韻 平 group, even-position alternation
inside a hemistich (which is what 一三五不論 actually says), 對 across a couplet, 粘 across
couplets, non-rhyme hemistichs ending 仄, and 三平調. 對 and 粘 are **derived** rather than matched
against a table of four canonical patterns — the four patterns are what those two rules generate.

Its rime table is 平水韻 (劉淵/王文鬱, 106 rhymes, 8,550 characters) built from zh.wikisource.
中華新韻 is not used: Ming poets wrote to 平水韻.

**Every form rate ships as three numbers, not one.** The rate; the fraction of *enforced positions*
the table could read; and the fraction reached only by a simplified→traditional mapping. This is
not decoration. An unreadable character is not scored neutrally — it is scored **leniently**,
because every skipped rule is a rule not failed. Unreadability therefore looks like correctness,
and it varies with exactly the things one wants to compare.

That is not hypothetical. The first P0 run reported 七律 passing at 58% against 七絕's 34% and I
published it as unexplained. The explanation was the table: 平水韻 is traditional, a model asked
for a longer poem drifts into simplified characters, and those read as UNKNOWN. 26.7% of 七律's
enforced positions were unreadable against 5.1% of 七絕's. After mapping, 七律 fell to 40% and the
24-point cross-form gap — which I had already published — became 8. **The mapping is built from the
table itself**: every traditional character converted the other way, so a simplified character maps
to the *set* of table characters collapsing onto it, and a set that disagrees on tone (发 is both
發 and 髮) is UNDECIDED, the same rule the polyphones get.

**What the instrument refuses to decide.** 874 of 8,550 characters (10.2%) sit in both a 平 and a
仄 rhyme; their tone belongs to the reading, not the glyph. At an enforced position such a
character yields UNDECIDED — never a pass, never a fail. A verifier that guessed would manufacture
whichever pass rate the guess favoured, and this number gates a training decision.

**What it cannot see at all.** The extractor strips to CJK, so a Latin token wedged inside a line
is invisible: `不 SwiftUI 也在斜` is 28 CJK characters and scores as a 七絕. It is counted
separately rather than left for whoever reads the poems.

### A strict rule, correctly written, failed a famous series

The 古體 branch exists to catch poems that are not 近體 at all. Its first test required the rhyme
characters to share a 仄 rhyme group — which is exactly what 古體 does **not** have to do: 古體
permits 通韻 across neighbouring groups, and 近體 is the form that forbids it. So the test failed
the very poems it was written for.

What it failed was not obscure. Eight of 文徵明's thirteen 五絕 are 古絕 — 其四, 瀟湘八景 and six
of the Eight Views of Xiaoxiang (瀟湘夜雨, 洞庭秋月, 平沙落鴈, 山市晴嵐, 漁村夕照, 煙寺晩鍾) —
all rhymed on 入聲 across adjacent groups. The verifier scored a famous series as defective
regulated verse, and the resulting 53.8% made 五絕 look like a form both poets were bad at.

**The cross-poet symptom is what caught it**: both the subject and the control collapsed on the
same form under the same instrument, while their 七絕 rates were 79% and 92%. Two poets, two
editions, one rule — that pattern accuses the instrument, not the poets. Corrected, 文徵明's 五絕
reads 60.0%, on a pool that falls from thirteen to five. Both columns are published; the fix
raised the rate and shrank the reference it could serve as, and that is the honest outcome rather
than an awkward one.

### The acceptance test fails, and that is recorded rather than tuned away

The study's own bar was "real Tang Yin 七絕 must pass ≥ 90%".

| pool | rate | positions read |
|---|---|---|
| 唐寅 七絕, raw | 30/38 = **78.9%** | 84.2% |
| 唐寅 七絕, filtered | 30/36 = **83.3%** | 84.2% |
| 文徵明 七絕 (control) | 146/157 = **93.0%** | 82.7% |

The bar is **not met** and is recorded as unmet. The filtered pool removes what is not 近體 — one
古絕, and one 打油詩 that appears on two Wikisource pages under different titles — each named with
its reason in `exclusions.json`. The six that remain are genuine faults of genuine 近體 poems: one
出韻, one 失對+失粘, three single 拗 hemistichs and one 三平調. Reaching 90% would mean excluding
poems that are 近體 and do break rules.

**The verifier is nevertheless usable, and the argument for it is not its acceptance test.** It
separates poet from baseline by 45 points on the same form, under the same instrument: 78.9%
against 32.0%. And the control poet is read *worse* than the subject (82.7% of positions against
84.2%) and still passes more — so his higher rate is not bought with latitude.

---

## P0 — does the untrained model write a regulated poem?

Qwen2.5-7B-Instruct bf16, 50 topics × 4 forms, temp 0.7, three seeds.

| form | pass, per seed | mean | |
|---|---|---|---|
| 七絕 | 32.0 · 32.0 · 32.0 | **32.0%** | **scored** |
| 五絕 | 34.0 · 34.0 · 34.0 | 32.0% | *observed only* |

**五絕 is reported and never used as evidence.** After the 古體 ruling its poet reference is five
scored poems for 唐寅 and five for 文徵明, and a comparison against a bar of five is not a
comparison. It gets the same treatment as the 律 forms, arrived at from the other direction: the
律 forms because their seed variance is larger than any training effect, 五絕 because its reference
is too small to be one.

**P0 predicted < 40% of 七絕 attempts. Confirmed — and narrowly.** At n=50 and p≈0.32 the binomial
SE is 6.6pp, so the threshold sits about 1.2 SE above the estimate. On the *extracted* denominator
it is 38.1%, which is the same verdict by a point and a half.

Extraction reports its own three buckets — EXTRACTED, WRONG_LENGTH (a model failure), NO_POEM (a
*parser* failure). **NO_POEM is 0 in all four forms, every seed.** A parser that extracts nothing
hands the verifier an empty string, which fails every rule and is indistinguishable from a model
that cannot write a 七絕; that number is the reassurance, and it is published rather than assumed.

### A rate can reproduce three times and mean nothing about stability

The baseline gives 五絕 34.0% on all three seeds. Between any two of those seeds, **16, 20 and 28
of the 50 poems flip verdict**, and zero of the 200 generations is identical between seeds. A rate
reproduced three times over a sample that churns half its outcomes.

Earlier, 七絕 gave 32.0% on two seeds with 24 of 50 flipping — 12 each way. "Reproduced exactly" is
never a claim about stability. **Count the flips.** A second seed landing on the same point does
not tighten the estimate.

This is also why the 律 forms are reported and never used as evidence: the baseline's own two first
seeds gave 七律 40% and 12%, a 28-point swing — larger than any plausible training effect.

---

## P2 — mechanism E (LoRA), the form half

Rank 8, 16 layers, lr 1e-5, on 63 training examples (70-poem split minus 7 held for validation),
prompted in exactly the shape P0 asks in. Three seeds per arm.

| arm | 七絕 *(scored)* | 五絕 *(observed)* | Latin intrusion / 200 |
|---|---|---|---|
| baseline | **32.0%** (spread 2.0pp) | 32.0% | 27 · 26 · 24 |
| LoRA, iter-60 *(E arm)* | **28.7%** (10.0pp) | 32.0% | 16 · 10 · 14 |
| LoRA, 3 epochs *(reported)* | **16.7%** (6.0pp) | 9.3% | 84 · 79 · 91 |

**Three epochs on 63 examples is destructive.** Form compliance halves on both scored forms, by a
margin far larger than any arm's seed spread. Three signals agree and they were available before
the generations were scored: validation loss bottoms at iteration 60 (1.388) and climbs to 1.590 by
189 while train loss falls to 0.179; form compliance halves; Latin-token intrusion triples.

**One epoch neither harms nor helps.** 28.7% against 32.0% on 七絕 — a 4-point gap well inside the
6.6pp SE, on the widest spread in the table — and identical on 五絕.

**Neither adapter improves form**, which may be the correct result rather than a disappointing one.
Mechanism E is a *voice* intervention and the form verifier is not where its benefit should appear.
What this half establishes is the **cost**: large at three epochs, nil at one.

**The one monotone effect in the study so far is Latin-token intrusion**, and it runs in the
adapter's favour: one epoch of Chinese poetry cuts English-token leakage to roughly half the base
model's rate, on every seed in the same direction.

### The final checkpoint was not the best one

`adapters.safetensors` — the weight the run leaves behind — is the 189-iteration adapter, and it is
the destructive one. The val-loss minimum at iteration 60 is the arm. This is the estate's standing
finding about restore-by-filename (窯's T3.4v) arriving in our own training run, for a measured
reason, and it is worth stating in the paper: **a training run's final weight is a default, not a
choice.**

---

## P1 — the judge (voice half), seed 0, 2026-09-16 (理)

`judge.py --provider bedrock --model us.anthropic.claude-opus-4-7 --seed 0`, no tools, `only answer A or B`,
every request in `results/judge_log.jsonl`, summary in `results/p1_judge_seed0.json`.

| arm      | pairs | judge correct | accuracy | SE    |
|----------|------:|--------------:|---------:|------:|
| baseline (base 7B, form-passing 七絕) | 42 | 42 | 1.000 | 0.000 |
| iter60 (LoRA, val-loss minimum)       | 34 | 33 | 0.971 | 0.029 |
| e3 (LoRA, 3 epochs)                   | 34 | 30 | 0.882 | 0.055 |

50% = indistinguishable from Tang Yin. Nothing is near it. e3 vs baseline is 2.1 SE, e3 vs iter60 is 1.4 SE, one seed.

**Probes, run before the pairs:** held-out 七絕 26 → judge set 22. Completion overlap ≥ 0.6 excluded four
(海棠美人圖 1.00, 題畫#7 0.95, 言志 0.90, 題畫#11 0.81) — the judge can recite those and they never reached a pair.
Attribution flagged two (開門七件事, 除夕口占) — kept, flagged. Canaries: 0 of 10 fabricated poems claimed as
known, so the judge's "I know this" is not a yes-bias and the exclusions are real memorisation.

**Reading.** The arm the form verifier called destructive (e3, 17.3% form) is the arm that fools the voice judge
most — on its form-passing survivors only. Form and voice are separate axes; a form score does not predict a
voice score, and the study needs both columns on every arm from here.

**Spend and one defect.** ~$0.72 for 172 calls (extrapolated from 81 logged calls at $0.339, 8,385 in / 2,841 out).
The log holds only the first 81 rows: a `git stash -u` on the working tree during the run moved the untracked
log file aside and the process kept writing to the unlinked inode; the last 91 rows are lost. Never stash
with `-u` under a running writer; the summary file is complete.

**Seed 1 (2026-09-16 18:3xZ), log whole (172 rows, $0.59):** judge set 24 (excluded 海棠美人圖 1.00, 言志 0.86; flagged the
same two; canaries 0/10). baseline 42/42 = 1.000; iter60 32/34 = 0.941 ± 0.040; e3 30/34 = 0.882 ± 0.055.
Pooled seeds 0+1: baseline 84/84; iter60 65/68 = 0.956 ± 0.025; e3 60/68 = 0.882 ± 0.039. e3 vs baseline 3.0 SE;
e3 vs iter60 1.6 SE. The completion probe is stochastic across seeds (4 excluded, then 2) — it runs fresh per seed.

**Next (P1 continued):** seed 2 (running), then (~$0.7 each) to put the e3 vs baseline gap past or under 2 SE; then the
reason probe — the same judge asked for one sentence on the four pairs that fooled it and four it caught, eight
calls — which is the only diagnostic of *what* gives the 7B away.


- **None of this is the judge.** Whether the *voice* moved is P1/P3, on the held-out 33, and it is
  not answerable from anything here. The form half can say the cost of the intervention and
  nothing about its benefit.
- **One base, one rank, one learning rate, one LoRA run, one split seed. No sweep.**
- **63 training examples**, and the split is form-imbalanced — 七律 41, 七絕 13, 五絕 4, 五律 1,
  11 unregulated. The 絕 forms are the ones scored and the ones least trained.
- **五絕 has no usable poet reference**: after the 古體 ruling, five scored poems for 唐寅 and five
  for 文徵明. A comparison against a bar of five poems is not a comparison.
- **The held-out set is no longer a different edition.** ctext holds four Tang Yin editions and is
  closed to us — it requires authentication and asks researchers not to scrape, which we did not.
  The Wikisource fallback is the same text: 列朝詩集's 唐寅 section transcludes the held-out pages,
  24 of 25. So the split is a random cut of one corpus, and P4's probes still mean something only
  because memorization is a property of the judge's weights rather than of our source.
- **Topics are mine**, 50 chosen in his register, sampled from nothing.
- **The verifier fails its own acceptance test**, and every number above inherits that.

---

## PRE-REGISTERED — the §4.3 gate as a rejection test (written 2026-09-16, before the first episode)

理 ruled the gate's job has changed: it is no longer meant to build a better memory, it is meant to
**reject v1_7b's lessons before injection**. Prediction P0 says the verifier refuses ≥ 20% of
candidates. Both outcomes are written here first, so the reading cannot follow the number.

**Shape, fixed now:** n = 20 × 2 conditions (in-scope, shuffled-family control) × 2 arms (with
lesson, without) × **all 12 lessons**, no subset. ~5 h of llm1 GPU. A rejection rate computed over
a confidence-chosen subset is not the rate P0 predicts, which is why every lesson runs.

**The situation that makes this sharp:** v1_7b's lessons are *true*. `[A_boundary, consumer] start
at producer` is 309/309 green at confidence 0.992 in the run it was distilled from. The arm that
injected them scored **5.0% against a 62.5% baseline**. So the gate is being asked to reject advice
that is correct and still harmful.

- **If P0 HOLDS** — the gate rejects ≥ 20% of lessons that are true and still harmful: §4.3 catches
  *carrier* harm, not only wrong advice. That is a stronger claim for the gate than rejecting
  false lessons would have been.
- **If P0 IS REFUTED** — the gate admits them: the failure lives in the carrier, no lesson-level
  gate can see it, and the paper says so. The gate would then be the wrong instrument for this
  class of harm rather than a broken one.

Neither outcome is a disappointment; the pre-registration is what makes it a finding rather than a
reading.
