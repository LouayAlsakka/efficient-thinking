#!/usr/bin/env python
"""WO-318 — the canonical form of an `area`, and equality on it. ONE property, reused.

THE RULE IS 案内'S, NOT MINE (doc 239, 12438), and it is transcribed here rather than re-derived:

    sort + subsumption-reduce `tags` — **a tag and its already-present ancestor collapse to the
    ancestor; siblings never reduce** — single-value `commit`, closed-enum `intent`. Two statements
    are semantically equal iff their canonical forms are structurally equal.

I asked for exactly this in 12378 and gave the reason: disagreement measured by string equality on
two JSON blobs measures FORMATTING. A replicate rate is the one number the word "predictable" rests
on, and a comparator that quietly works is worse than one that fails, because it inflates or hides
the thing it is there to count.

⚠️ THE SHAPE IS CONTESTED AND THIS FILE DOES NOT PICK A SIDE. 形 showed doc 239 and the
design of record disagree on what `intent` and `commit` ARE: 239 has `intent ∈ {ask, commit}` with
`commit ::= tag_id | null`; the walkthrough, schema.py's `_INTENTS`, 理 and my own live
`/area` all have `intent` one of eight and `commit` a boolean. Those are different axes wearing the
same field name, and under 239's shape `area.intent in node.required_by` can only ever fire for the
literal "ask".

So: the SUBSUMPTION RULE is shape-independent and is implemented once. The shape is DETECTED, and
two areas of different shapes are **refused, never coerced**. A comparator that silently accepted
both would turn the estate's open design question into a number nobody could interpret.

⛔ Nothing here renders, ranks or decides. It compares.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))

DESIGN_OF_RECORD_INTENTS = ("browse", "ask", "book", "order", "hours", "directions",
                            "contact", "cancel")
DOC239_INTENTS = ("ask", "commit")


class ShapeMismatch(Exception):
    pass


def shape_of(area):
    """'design-of-record' | 'doc239' | 'unknown'. Decided on `commit`, which is unambiguous.

    `intent` cannot decide it: "ask" is legal in BOTH vocabularies, and it is the single most
    common intent, so an intent-based sniff would mislabel the majority of real areas.
    """
    if area is None:
        return "none"
    c = area.get("commit", "__missing__")
    if isinstance(c, bool):
        return "design-of-record"
    if c is None or isinstance(c, str):
        return "doc239"
    return "unknown"


def load_parents(taxonomy=None):
    tax = taxonomy or json.load(open(os.path.join(HERE, "taxonomy_demo.json")))
    return dict(tax.get("parents") or {})


def ancestors(tag, parents):
    """The declared chain upward. Uses the taxonomy's `parents` map, NEVER the dotted string.

    `info.hours`'s parent happens to be `info`, so a prefix test would agree today and would go
    wrong silently the first time a registry names a tag whose id does not spell its parent.
    """
    out, seen, cur = [], set(), parents.get(tag)
    while cur is not None and cur not in seen:
        out.append(cur); seen.add(cur); cur = parents.get(cur)
    return out


def canonical_tags(tags, parents):
    """Sort, dedupe, and drop any tag one of whose ancestors is ALSO present. Siblings survive."""
    present = set(str(t) for t in (tags or []))
    kept = [t for t in present if not (set(ancestors(t, parents)) & present)]
    return sorted(kept)


def canonical(area, parents=None):
    """A hashable canonical form, or None for a missing area."""
    if area is None:
        return None
    parents = load_parents() if parents is None else parents
    return (str(area.get("intent", "")),
            tuple(canonical_tags(area.get("tags"), parents)),
            area.get("commit", None) if not isinstance(area.get("commit"), bool)
            else bool(area["commit"]),
            str(area.get("taxonomy") or area.get("taxonomy_ref") or ""))


def make_comparator(parents=None, taxonomy=None):
    """The `comparator` replicate_runner.py asks for. Refuses across shapes; None == None is True.

    A None area is a call that did not parse. The runner already refuses to report a rate over a
    set containing one, so equality on None only has to be consistent, not meaningful.
    """
    parents = load_parents(taxonomy) if parents is None else parents

    def same(a, b):
        if a is None or b is None:
            return (a is None) and (b is None)
        sa, sb = shape_of(a), shape_of(b)
        if sa != sb or sa == "unknown":
            raise ShapeMismatch(
                "refusing to compare a %r area with a %r one (形: the two shapes disagree on "
                "what `intent` and `commit` ARE). Coercing them would make an open design question "
                "into a number nobody can interpret." % (sa, sb))
        return canonical(a, parents) == canonical(b, parents)

    return same


def _selftest():
    parents = load_parents()
    same = make_comparator(parents)
    A = lambda i, t, c: {"intent": i, "tags": t, "commit": c, "taxonomy": "demo-v0"}

    # 1 — order does not matter
    assert same(A("book", ["info.staff", "svc.hair.cut"], True),
                A("book", ["svc.hair.cut", "info.staff"], True))
    # 2 — a tag and its ancestor collapse TO THE ANCESTOR
    assert canonical_tags(["svc.hair", "svc.hair.cut"], parents) == ["svc.hair"]
    # 3 — siblings never reduce
    assert canonical_tags(["svc.hair.cut", "svc.hair.beard"], parents) == \
        ["svc.hair.beard", "svc.hair.cut"]
    # 4 — a grandparent subsumes too
    assert canonical_tags(["info", "info.hours"], parents) == ["info"]
    # 5 — commit is part of the identity
    assert not same(A("ask", ["info.hours"], False), A("ask", ["info.hours"], True))
    # 6 — so is the taxonomy the tags were drawn from
    x = A("ask", ["info.hours"], False); y = dict(x, taxonomy="v1")
    assert not same(x, y)
    # 7 — shapes are refused, not coerced
    try:
        same(A("ask", [], False), {"intent": "ask", "tags": [], "commit": None,
                                   "taxonomy_ref": "demo-v0"})
        raise AssertionError("should have refused")
    except ShapeMismatch:
        pass
    # 8 — the sniff is on commit, not intent: "ask" is legal in both vocabularies
    assert shape_of(A("ask", [], False)) == "design-of-record"
    assert shape_of({"intent": "ask", "tags": [], "commit": None}) == "doc239"
    # 9 — a dropped call compares only to another dropped call
    assert same(None, None) and not same(None, A("ask", [], False))
    # 10 — the ancestor walk uses the DECLARED map, not the dotted string
    assert ancestors("info.hours", {"info.hours": "zzz"}) == ["zzz"]
    print("  area_equal selftest: 10/10 — 案内's rule, transcribed, with the shape refusal")


if __name__ == "__main__":
    _selftest()
