# ET-8 application study: a Tang Yin (唐寅, 1470–1524) voice prior — Claude judges, Qwen trains

Louay, 2026-09-14: "can we use ET-8 to study Tang Yin's poems and make an expert skill that mimics him …
we can use Claude to judge and Qwen to train." This directory is that study, run with the ET-8 pipeline
(env → agent → distill → verify → inject) and the series rules: commit before running, no text before
numbers, scored predictions written before any run.

## The three parts (same shape as the debugging env)
| part | what | instrument |
|---|---|---|
| verifier (external, objective) | FORM: line count, chars per hemistich, rhyme class (平水韻), 平仄 against the standard 七絕/七律 patterns | `verify_form.py` (to write) — pass/fail per rule, like the failing test in `et8_env.py` |
| judge (Claude, no tools) | STYLE: blind DISCRIMINATION — one generated poem vs one held-out real poem, "which is real?" Mimicry = how far accuracy falls toward 50%. Same-circle control poet (文徵明) says whether we learned Tang Yin or Ming style | plain Messages API call, **no tools attached**; every request/response logged; usage.server_tool_use must be absent/0 |
| trainee (Qwen, local) | Qwen2.5-7B-Instruct bf16 under mlx-lm (llm1). Mechanism A = distilled two lists as a text prior; mechanism E = LoRA on corpus + Claude preference pairs | `et8_agent.py`-style loop; LoRA via mlx-lm `lora` |

## Judge contamination controls (Louay's question: search vs memorization)
- **Search** is a request property: the judge runs with no tools; the logged response has no server-tool-use blocks and
  `usage.server_tool_use.web_search_requests` is absent/0. Never run the judge from a Claude Code session (WebSearch exists there).
- **Memorization** is a weights property — three probes per held-out poem, results kept in `probes/`:
  1. completion: title + first line → continue. Verbatim/near-verbatim ⇒ EXCLUDED from the judge set (report the exclusion rate).
  2. attribution: full text, no title/author → "who wrote this?" Flags, does not exclude (style alone can answer it).
  3. canary: poems WE write in his style, posted nowhere. Confident "recognition" ⇒ the judge pattern-matches; calibrates probe 2.
- Residual: seen-but-not-recitable. Report discrimination for Tang Yin BESIDE the control poet's; if Tang Yin's is much
  higher, the judge is leaking on recognition and the number is not trusted.

## Predictions (scored, written 2026-09-14 before any run)
- P0 baseline Qwen-7B passes the form verifier on < 40% of 七絕 attempts.
- P1 Claude discriminates baseline Qwen output from real held-out Tang Yin > 90%.
- P2 mechanism A (text prior) closes < half the discrimination gap that LoRA closes.
- P3 after LoRA, discrimination stays > 70% — surface mimicry lands, the voice does not.
- P4 exclusion rate from the completion probe ≥ 20% of Wikisource held-out (his famous poems are in the weights).

## Corpus
- `fetch_corpus.py --author 唐寅` → `corpus/tangyin_wikisource.jsonl` (zh.wikisource Category:唐寅, 50 pages, <poem> blocks;
  a line of 14 chars = one couplet of two 7-char hemistichs). This is the HELD-OUT source.
- Training split: 六如居士集 / 唐伯虎全集 from the Chinese Text Project (ctext.org) — Sautee fetches (step 1b), same JSONL schema.
- Control poet: `fetch_corpus.py --author 文徵明` (same circle, same city, same forms).

## Hand-off #2 to Sautee (llm1) — build order
0. Report llm1 state first: `python3 -c "import mlx_lm; print(mlx_lm.__version__)"`, repo clone present?, free disk. (09-05: no mlx-lm, python 3.9.6.)
1. `pip install mlx-lm` (or ask tetsu for a python ≥3.10); `git pull`; `python3 experience/tangyin/fetch_corpus.py --author 文徵明 --out experience/tangyin/corpus/wenzhengming_wikisource.jsonl`.
   1b. ctext fetch for the training split — write `fetch_ctext.py` beside this one, same schema; respect their rate limit.
2. `verify_form.py`: rhyme classes from a 平水韻 table, 平仄 from a tone table (中華新韻 is NOT acceptable — Ming poets used 平水韻). Unit-test on the corpus: real Tang Yin 七絕 must pass ≥ 90% (the 10% is variant readings; list them).
3. Baseline: Qwen2.5-7B-Instruct bf16, 200 prompts (50 topics × 4 forms), temp 0.7, record form pass rate → P0.
4. Judge run: Ri writes the judge harness (`judge.py`, Messages API, no tools) and the probes; Sautee does NOT need an Anthropic key — hand the 200 generations back as JSONL.
5. LoRA (mechanism E): mlx-lm lora on the training split, 3 epochs, rank 8; regenerate the 200; hand back.
Series rule: numbers into `results/` as JSON with the sha of the run, then text.

## Rulings, 2026-09-14 (理, on Sautee's 10350–10356)
- **Control corpus:** 文徵明 from 甫田集 卷01–15 (757 poems, `fetch_skqs.py`) is the control. Larger than the subject is the right way round.
- **Training split — (b) re-cut, now.** ctext is closed (auth, and they ask not to scrape — correct to stop; a subscription is Louay's spend decision and is put to him separately). Until then: a SEEDED random split of the 103 Wikisource blocks BY POEM, ~70 train / ~33 held-out. The paper states what the split stopped meaning: held-out is no longer a different edition. P4's probes still run on the held-out 33 — memorization is a property of Claude's weights, not of our source. Run the ten-minute disjointness check on the 28 linked 集外詩 titles (c); if disjoint, they join the training split.
- **(i) the ≥90% bar:** the bar was written for a filtered pool and the category is unfiltered. RULED: keep an `exclusions.json` naming each excluded poem with its reason (古體/歌行; the one folk 打油詩 that appears as two pages; corrupt text) and apply the bar to the filtered pool. The raw-pool number (79.5% with the eight named) is PUBLISHED beside it, not replaced. Genuine 出韻 (子胥圖) stays a fail — it is true of the poem. No further tuning of the verifier; the three rule refinements (特拗 etc.) stand as documented rules.
- **Three numbers, always:** every form rate ships as (rate, fraction of enforced positions the table could read, fraction reached only by simplified→traditional mapping). Sautee's rule, adopted for the whole study.
- **(ii) the 79 `{{SKchar}}` poems:** excluded, with the printed `NOT SCORED … 79 poems` line kept in every run and report. Variant glyphs of the 四庫 edition are NOT mapped (no invented 異體字 table); reported as edition latitude.
- **(iii) `text_raw` vs `text`:** the verifier and the judge both read `text` (re-segmented, uniform presentation); `text_raw` stays in the row as provenance and is never scored.
- **P0: CONFIRMED** — 七絕 32.0% of attempts / 38.1% of extracted after the simplified mapping (`results/p0_base_7b_rescored_v2.json`, the re-score from saved generations with the 五絕 rule fix; `results/p0_base_7b_rescored.json` is the pre-fix re-score, kept for the before/after; seed 2 is `p0_base_7b_s2_v2.json`), NO_POEM 0 in all four forms. The 24-point cross-form gap was the instrument (simplified characters unread by a traditional table) and is withdrawn — the 8-point residual is reported.
- **Next:** Ri writes `judge.py` (Messages API, no tools) + the three probes and runs P1 on `results/p0_base_7b.jsonl`; Sautee builds the split, the exclusions file, then LoRA (rank 8, 3 epochs) on the train split — thin at ~70 poems, run anyway and say so; the interesting result is P2/P3 either way.

## Rulings, 2026-09-14 (second round, on Sautee's 10423–10424)
- **Seeds:** P2/P3 are scored on 七絕 (and 五絕) with ≥3 seeds per arm; the mean AND the seed spread are published; 七律/五律 are reported as observed, never as evidence (seed-to-seed swing of 28pp at n=50 exceeds any plausible LoRA effect on ~70 poems). "Reproduced exactly" is never claimed from a rate — count the flips (24/50 flipped, 12 each way, behind an identical 32.0%). P0 stands, narrowly: binomial SE 6.6pp at n=50, the < 40% line sits ~1.2 SE above the estimate.
- **The verifier bar:** filtered pool 83.3%, raw 79.5%, control poet 91.8% — the ≥90% bar was a prediction about the verifier and it is NOT met; recorded as such, not tuned toward. The verifier is ACCEPTED as the study's instrument on the strength of the 45-point separation (poet 79.5% vs baseline 32.0%, same instrument), with the eight named failures and the readability column beside every rate. Latitude left in the corpus (拗 hemistichs, corrupt text) is documented, not absorbed.
- **(c) is empty** — measured, zero disjoint pages; (b) was the only path. **Split:** 70/33 by poem, seed 7, near-duplicate clusters (bigram Jaccard ≥ 0.5) kept on one side — 桃花庵歌's four blocks and the folk-poem pair; exact matching alone would have leaked. Accepted.
- **Three open items from 10458, ruled:** (1) the mechanism-E arm IS iter-60 (the val-loss minimum); the 3-epoch adapter is reported as the overfitting result, not as an arm. (2) `--ze-neighbour` ON — same ruling as the 五絕 fix above. (3) 古體 LEAVES the scored pool: the verifier emits a third bucket, `not_regulated`, never pass/fail (宮妃夜游圖's hole closed the same way). (4) Latin-token leakage falling below baseline after one epoch is the study's only monotone effect so far — published as such, beside its seeds.

## Rulings, 2026-09-15 00:5xZ (理, on Sautee's 10453–10459)
- **五絕 verifier rule:** apply the fix (古體 通韻 across neighbouring 入聲 groups was being scored as broken 近體 — a rule Sautee wrote, cross-poet symptom proves it is the instrument). Re-score ALL pools from saved generations and publish before/after beside each other; no other rule changes.
- **P2 form half, scored:** LoRA rank 8 on 63 poems is DESTRUCTIVE at 3 epochs (七絕 17.3% vs baseline 32.7%, spread 6pp vs a 15pp gap; Latin leakage 84–91/200) and NULL at the val-loss minimum (iter-60, 28.7%, indistinguishable from baseline). Recorded as the result. "iter-60 beats baseline" was one lucky seed — the three-seed rule caught it, which is what it is for.
- **Hold further LoRA arms until P1 (the judge) runs on the same generations.** The question P2/P3 exist for is whether the VOICE moved; the form verifier says the form fell. If the judge's discrimination also fell (mimicry improved) at iter-60, the next arm is form-preserving: KL-to-base regularization (proposal §loss) or a mixed batch with instruction data; if it did not move, LoRA on 70 poems is the null result and the paper says so.
- **五絕 baseline 34.0% on three seeds with 16–28 poems flipping** — the balanced-coincidence rule again; publish the flips beside the rate. Nothing to tune.

## Sautee's lineup (2026-09-15, Louay: "make sure Sautee has a WO lineup, so she never waits")
Ordered. Each item is runnable WITHOUT waiting on the judge or on a ruling; take the next one when the current one is written up.
1. **Form-preserving LoRA arm, built now** (so the judge's answer picks an arm instead of starting one): iter-60 shape, rank 8, ONE epoch,
   plus either (a) KL-to-base regularization per the proposal's loss, or (b) a mixed batch — the 63 poems interleaved with an equal
   number of the model's own instruction-style prompts. Three seeds, all four forms observed, 七絕 scored. Publish beside the two arms.
2. **The OS-axis experiment on mini6** the moment 巳紗 opens the box (prepare the venv + adapter copy now so the run is minutes).
3. **The control poet as a discrimination reference:** build `judge.py`-shaped pairs where the real poem is 文徵明's 七絕 (not 唐寅's),
   so the judge run can report Tang-Yin-vs-generated beside Wen-vs-generated in one pass. Write the pairs file; do not call the judge.
4. **ET-8 core — the paper's own next step** (`experience/`): wire `--steer` into `et8_agent.py`, run the held-out d′ `check`, then
   mechanisms A / F / C on 20 tasks vs baseline through the verify gate, two seeds. Numbers into `experience/results/`, then text.
5. **A0 chess prior** (`chessnet/search.py`): the learned move prior in PUCT, sims-to-Elo — the anchor the series has owed since 09-05.
Rule: when an item blocks on someone, say so once on the 1:1 and take the next item; nothing on this list waits on Ri.
