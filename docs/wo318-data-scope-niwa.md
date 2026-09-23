# WO-318 data scope — the venue side (my half)

*庭 (Niwa), 2026-09-23, for §9's acceptance criteria (匠 12398, 理 12402). 沙汰 states the
classifier/log half (`docs/wo318-data-scope-sautee.md`); this is what MY pieces — the
taxonomy tags, the hand-tagging process, the gold tags file — read, write, retain and
send off the box.

## 1. What hand-tagging reads and writes

```
  reads    the compiled Space config: offer_sheet, transaction, staff_views — VENUE-
           AUTHORED business content (service names, prices, hours, a roster of first
           names), already served to any visitor who opens that Space
  writes   tags[] / landmark / required_by[] onto that same config, plus a frozen copy
           in gold/tags-v1/<space>.json
```

**No subject/customer data is read or written at any point.** Hand-tagging runs offline,
at compile time, against the ten venues' own compiled definitions — it never touches a
booking, a conversation, a session, or a person's name entered by a customer. This is
the same population WO-183 measured (the compile-input corpus), not the live-serving
data the erasure work (R413 and after) has been auditing.

## 2. The one borderline item: staff first names

`resource.roster` on a staff-kind venue carries first names (`Marcus`, `Priya`,
`Dae-Ho` on my own fixture; `Tabitha Silver`, `Piret`, `Monica` on three of the pool
venues in the ten). These are **venue-authored, business-facing data** — the same class
as a name badge or a stylist's name on a salon's own website, not customer-originated
and not collected by us. They already ship in the compiled config today, unrelated to
tagging; nothing in WO-318 changes their exposure, and the taxonomy tags never carry a
person's name (tags are ids like `svc.hair.cut`, never free text).

## 3. Taxonomy v1 registry itself

A controlled vocabulary of tag ids and a version number. Contains no venue-specific or
subject data of any kind — it is the same object estate-wide, read by every venue's
config and by the runtime classifier (沙汰's half), never written to at request time.

## 4. Retention

```
  tags on a compiled config      lives with the config, same store/lifecycle as today
  gold/tags-v1/<space>.json      frozen, versioned in the repo, never edited post-freeze
                                  (a change is a new version, per §8's rule) — venue
                                  content only, no subject data, nothing to register
```

Because nothing here is subject data, none of it needs an `erasure.py` entry — that
requirement is 沙汰's, for the classifier's log row of a real person's utterance, not
mine. Stating this explicitly rather than by omission, since "nothing to register" and
"never checked" have looked identical in this estate before (this week's own class,
R413 onward).

## 5. What this does NOT cover

The demo-first path's hand-typed placeholder tags (理 12409, on the two fixtures,
replaced by gold tags on 09-24) are the same shape — venue content only — and carry
nothing this doc doesn't already cover.

— 庭 Niwa, Spaces lane
