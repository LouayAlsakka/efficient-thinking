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
