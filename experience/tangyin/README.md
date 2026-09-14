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
- **P0: CONFIRMED** — 七絕 32.0% of attempts / 38.1% of extracted after the simplified mapping (`results/p0_base_7b.json`), NO_POEM 0 in all four forms. The 24-point cross-form gap was the instrument (simplified characters unread by a traditional table) and is withdrawn — the 8-point residual is reported.
- **Next:** Ri writes `judge.py` (Messages API, no tools) + the three probes and runs P1 on `results/p0_base_7b.jsonl`; Sautee builds the split, the exclusions file, then LoRA (rank 8, 3 epochs) on the train split — thin at ~70 poems, run anyway and say so; the interesting result is P2/P3 either way.
