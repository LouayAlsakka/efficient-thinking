# Efficient Thinking X — series plan (X-a … X-d)

*Ruled by the author 2026-09-21: X is a programme, written as four papers, one rung each, backward from measured
results, as VIII was. Concept: `efficient-thinking-10-concept.md` (registered 2026-09-21). This plan is the split; each
paper's proposal is committed before its first run and carries its own STATE line.*

| paper | rung | question | first experiment, on harnesses the series holds |
|---|---|---|---|
| **X-a Existence and Efficiency** | Claims A + B | Does representation change reasoning cost at equal task information and equal everything else? | Chess (ET-I harness): an equal-parameter model trained from scratch on board tensors vs one on an English rendering of the same positions; Elo per FLOP. **Length-matched control:** a verbose-but-structured rendering of the state at the English token count, so length and structure are separated before any claim. **Random reversible encoding** of the state as the information-preserving, structure-destroying control. Second world: the debugging environment (exact AST + tests vs the English prompt). |
| **X-b Discoverability** | Claim C | Can a learning procedure find a representation near the capability-preserving compression knee? | The compression sweep Z_512 → Z_16 on X-a's world, cost measured end to end (encoder + reasoner + decoder). |
| **X-c Translatability** | Claim D | Can human language be mapped into and out of Z with translation cost charged? | Codecs English ↔ Z with Q_E, Q_R, Q_D measured separately against the known state; ET-III's judging rules for any model-judged step. |
| **X-d Language independence** | Claim E | Do systems never exposed to human language converge on related structure? | Alignment of Z_grounded and Z_language up to a simple map, with random projections, shuffled correspondences and permutation nulls; related work must place this beside the platonic-representation-hypothesis literature. |

**Rules carried from VIII.** Every rung has its own oracle and can fail alone; a negative at any rung is a finding at
that rung, not a verdict on the ones before. A result artefact names its instrument by hash. Readings are written
before runs. No product names. X-a starts when VIII-b's package closes (after gen2), on box B, and its proposal
with predictions is the next document.

**Bridge from VIII.** VIII-a's probe is a small existence result already held: a linear read of the hidden state
finds the faulty clause at 89.3% where the agent's language-mediated pick finds it at 30.7%. X-a's proposal cites it
as motivation, not as evidence.
