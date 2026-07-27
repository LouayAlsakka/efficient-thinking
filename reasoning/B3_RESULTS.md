# B3 — Larger-base, larger-data fine-tune: does the data lever beat a pretrained baseline? (Paper II §7 tail)

**Registered** in whitepaper2.md §7 (the shipped compendium): the two-floor 0.5B sweep showed "data sets
the destination" (both floors converge at ~26%) but explicitly disclaimed what it did *not* show —
*"neither curve ends above the instruct model's own zero-shot floor, and the larger-base, larger-data run
that would test baseline-beating is registered future work."* **B3 is that run.**

## Protocol (faithful to §7's base arm)

`reason_finetune_sweep.py --base --model mlx-community/Qwen2.5-3B-bf16 --sizes 0,64,256,1024,4096,7000
--epochs 3 --batch 4 --eval 150 --seed 0` — bf16 base (4-bit base was noted unstable in §7), plain
prompt/completion, cleaned targets, greedy eval on the same 150 held-out GSM8K. Run on llm2, 2026-07-24.
Baseline to beat = the **3B-instruct zero-shot floor, 75.3%** (same 150 held-out, computed on llm1).

## Result

| train examples | 0 | 64 | 256 | 1024 | 4096 | 7000 (full) |
|---|---:|---:|---:|---:|---:|---:|
| **3B base (bf16)** | 2.7% | 63.3 | 54.7 | 62.7 | 52.7 | **55.3%** |
| 0.5B base (bf16), §7 reference | 1.3% | 16.0 | 15.3 | 17.3 | 24.0 | 26.0% |

**Registered question — does larger-base + larger-data climb past a solid pretrained baseline? → NO.**
3B base + full GSM8K reaches 55.3%; even the curve's *best* point (63.3% at N=64) stays ~12 pts below the
3B-instruct zero-shot baseline of 75.3%, so the negative answer is robust to the 150-problem eval noise
(≈±8 pts). This **replicates the 0.5B pattern at 3B**: base+full-data 26.0 < 0.5B-instruct 35.3, exactly as
55.3 < 75.3. The §7 disclaimer holds at the larger scale — the data lever does not beat a solid pretrained
baseline in language.

## Two wrinkles (sharper than the flat prediction)

1. **The destination is capacity-dependent — "capacity is slack" does not transfer to language.** The 3B
   base plateau (~55%) sits far above the 0.5B base's (~26%). In Connect-4 §6 capacity was slack at fixed
   low data; here a larger base reaches a much higher *data-determined destination*. So the §7 headline
   sharpens: **data sets that there is a destination; base capacity sets how high it can be.** Data still
   fails to cross the pretrained baseline, but the baseline it fails to cross rises with capacity.
2. **Near-instant saturation.** The 3B base hits ~63% by just 64 examples and then wobbles flat (52–63%)
   through full data, where the 0.5B base climbed *gradually* (1.3 → 26 across the whole sweep). The big
   base needs **format alignment, not capability injection** — the reasoning is already pretrained; a few
   dozen examples align the output format and the rest is noise around the plateau. (The non-monotone
   wobble 63→55→63→53→55 is within the ±8-pt eval band; the load-bearing facts are the plateau height
   ~55–60% and that it never approaches 75.3.)

## The elicitation reading (and an E-E bridge)

Wrinkle 2 is sharper than "format alignment." Sixty-four examples buy ~85% of the entire fine-tune ceiling
at 3B (2.7 → 63.3), then thousands more add nothing; the 0.5B base instead *accumulated* information
smoothly (1.3 → 26 over the full sweep). So **the data lever at 3B is mostly elicitation, not information**:
the base already holds the capability, a handful of examples unlock the output format, and further narrow
data adds no new capability — yet it still lands 12–20 pts under the instruct floor, whose training added
something this LoRA cannot reach. That is ET-VII's **elicitation gap Δ appearing uninvited in a Paper II
experiment** — Δ made visible as the shape of a fine-tuning curve.

**Cheap registered follow-up for the E-E batch (not run here; llm1-only for now):** probe the *un-tuned* 3B
base's hidden states on GSM8K. If probe accuracy lands near 63% (the 64-example ceiling), the elicitation
reading is confirmed — the capability is present pre-tuning and the LoRA only surfaced it — and B3's miss
becomes a two-paper bridge (Paper II §7 ↔ ET-VII E-E). If probe accuracy is far below 63%, the fine-tune
added real information and the elicitation reading is wrong. Registered before running.

## Paper integration

Scored registered prediction → **Paper II v2**: one §7 note + one §10 ledger row (below). The result
*confirms* the shipped claim's stated limit and adds the capacity-scales-the-destination observation; no
existing v1.0.1 claim changes.

- **v2 §10 row:** *"the data lever cannot beat a solid pretrained baseline in language (registered §7
  future work)"* → **Hit — confirmed at 3B.** 3B base + full GSM8K = 55.3% < 3B-instruct zero-shot 75.3%,
  replicating 0.5B (26.0 < 35.3); new observation: the base-fine-tune destination scales with capacity
  (~26% at 0.5B → ~55% at 3B), so capacity is *not* slack for the destination height in language.
