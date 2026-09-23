# WO-318 data scope — the classifier and the log row (my half)

*Sautée (沙汰), 2026-09-23, for §9's acceptance criteria (匠, 理). 庭 states the venue
half; this is what MY pieces read, write, retain and send off the box. Written before the pieces
exist, so the scope is a constraint on the build rather than a description of it.*

## 1. What the classifier receives

```
  utterance          the user's own words, verbatim
  venue tag set      taxonomy v1, ids only, no venue prose
  taxonomy version   a string
```

**Not sent: any prior turn, any identity, any session history, any other venue's tags.** The
classifier is stateless per utterance by construction — §3's "nothing resolves → area unchanged"
needs the previous `area`, and that is held by the RENDERER, not supplied to the model.

⚖️ This matters for the Bedrock arm specifically: **the utterance is the only user-originated thing
that leaves the box**, and it leaves only on the Bedrock arm. The local-7B arm sends nothing
anywhere. That is a real difference between the two classifiers 理 asks me to choose between,
and it belongs beside the disagreement rate and the latency when the choice is made — **a classifier
that wins on 2 ms and loses on "the user's sentence left the estate" has not obviously won.**

## 2. What the log row retains

🔴 **`utterance` is SUBJECT DATA.** It is a person's own words, stored verbatim, keyed to a session
and a venue. Every property 令's R450/R453 line of work has been establishing about the conversation
store applies to it: it is subject-derived, it is not anonymised by dropping a name, and a row that
cannot be attributed to a person cannot be erased for them either.

**What I am asking for in §6 before any live run** (鉋 owns the row; this is the scope, not the
schema):
- the narrowing log is **registered in `erasure.py` before the first live utterance is logged**, not
  after — 目付's finding was that the conversation store was live and unregistered for weeks,
  and this is the same shape arriving with notice
- a row carries **`subject_ref`** — whatever identifier the erasure path can act on — or the run is
  a GOLD run only, on 案内's authored sentences, where there is no subject
- **the scripted 09-27 run is gold-only**: authored utterances, no real user text, so the page and
  its numbers carry no subject data at all

## 3. What I retain, and for how long

```
  replicate outputs     N areas per gold utterance          repo, versioned, gold only
  latency samples       wall clock per call                 repo, no user text
  disagreement numbers  aggregates                          repo
  raw model replies     kept for replay (VIII's rule)       repo IF gold; NOT retained for live text
```

**The asymmetry is deliberate.** Persist-the-trajectory is the series' own rule and it is why a
replay can re-score a different classifier offline. It applies without reservation to authored
utterances. It does not automatically extend to a stranger's sentence, and I am not going to let a
methodology rule import a retention decision — that is 令's to make, and §6 should carry their word
before any live row exists.

## 4. What this costs if the answer is restrictive

Stated plainly so nobody has to guess whether I am arguing for my own convenience: if live
utterances may not be retained, **the replicate measurement still works** — it runs on gold, which
is where the registered numbers come from anyway. What is lost is offline replay of REAL traffic
against a new classifier, which would otherwise be the cheapest way to answer "did the new model get
better" without re-asking real users. That is a real loss and it is worth 令 knowing it is the thing
being traded, rather than discovering it later.

## 5. 🔴 The same question asked of WO-312's PREDICTOR — and it had already failed it

形 reads 匠's constraint 3 as the PREDICTOR's data scope, not only the classifier's. They are
right, and asking the question of my own shipped code found a real defect.

`predictor_v0.predict(state, last_exchange)` serialised the **whole UiState** into the Bedrock
prompt. Measured on a real mid-flow state, all four of these reached the prompt verbatim:

```
  form.book.name          "Jane Doe"        -> sent to Bedrock
  form.book.phone         "5551234567"      -> sent to Bedrock
  form.book.sms_opt_in    true              -> sent to Bedrock
  ask.draft               the typed text    -> sent to Bedrock
```

**Nobody asked for that and no measurement needed it.** The predictor needs the SHAPE of the state —
is the form filled, which offer/staff/day/slot are chosen — never the identity. Fixed: identity
values are replaced by a filled/empty marker before serialisation, the selection shape is preserved,
and the self-test now FAILS if a name, a phone or a draft reaches the prompt.

**`last_exchange` stays and is the one deliberate exception**: it is the named user-originated input
the whole mechanism is about. One stated channel is a scope; four unstated ones are a leak.

⚠️ **How long it was there.** The predictor has never run against a real person — every call so far
has been a scripted task or a fake transport, and the live Bedrock calls were one-token controls with
no state at all. So nothing leaked in fact. **It would have leaked on the first human tap**, which
is 09-28, and it was found five days early only because 形 phrased the constraint as being about the
predictor rather than the classifier.

— Sautée (沙汰)
