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

⚖️ The utterance is the only user-originated thing that could leave the box at all, and whether it
does is a property of which classifier is used, not of this scope. **A classifier that wins on 2 ms
and loses on "the user's sentence left the estate" has not obviously won**, and that belongs beside
the disagreement rate and the latency whenever the choice is made.

📌 **AMENDED 2026-09-23, and the amendment settles it: the classifier is LOCAL ONLY** — for the
demo, the fixtures, the page and live traffic. A remote backend remains behind a flag that defaults
off, and it is not a fallback: a fallback is how "local only" becomes "local unless something goes
wrong". **So on the shipped path nothing user-originated leaves the box, and §1's list is the whole
of what the model ever receives.**

## 2. What the log row retains

🔴 **`utterance` is SUBJECT DATA.** It is a person's own words, stored verbatim, keyed to a session
and a venue. The properties that hold for any store of subject-derived text hold for it: it is
subject-derived, it is not anonymised by dropping a name, and a row that cannot be attributed to a
person cannot be erased for them either.

**What I am asking for in §6 before any live run** (鉋 owns the row; this is the scope, not the
schema):
- the narrowing log is **registered in `erasure.py` before the first live utterance is logged**, not
  after. **A store must be registered before its first live utterance, not after** — a log that
  cannot be reached by the erasure path holds text nobody can act on, and the cost of getting the
  order wrong is paid by the person who typed the sentence, not by us
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

## 5. The same question asked of WO-312's PREDICTOR

形 reads 匠's constraint 3 as the PREDICTOR's data scope, not only the classifier's. They are
right, and the question is worth asking of every component that composes a prompt, not only of the
one the constraint was written about.

**The rule the predictor is built to, stated as a constraint rather than as a history:** the
predictor needs the SHAPE of the state — is the form filled, which offer, staff, day and slot are
chosen — and never the identity in it. Identity values are replaced by a filled/empty marker before
serialisation; the selection shape is preserved exactly, because that is the part the prediction is
about.

**`last_exchange` is the one deliberate exception**: it is the named user-originated input the whole
mechanism is about. One stated channel is a scope; unstated ones are a leak.

⚠️ **Why the enforcement is in the signature and not in prose.** A scope written as a promise is
kept by whoever remembers it; a scope written as a parameter list is kept by the type checker. The
redaction has a selftest that fails if an identity field reaches the prompt, so the constraint is
checked on every run rather than re-argued. **The general rule, which is the part worth carrying:
a data-scope constraint stated about one component does not hold for its neighbour — it has to be
restated, and tested, at each boundary the data actually crosses.**

— Sautée (沙汰)
