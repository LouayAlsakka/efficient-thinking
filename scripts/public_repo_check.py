#!/usr/bin/env python3
"""Route candidate estate references in this repo to a HUMAN. It does not block and it does not fix.

WHY IT EXISTS, AND WHY IT IS NOT ANOTHER DENY-LIST. Five needle failures in one investigation this
week, across two people, every one of them caught by READING rather than by a better pattern:

    \\b in `git grep -E`          silently matches nothing      "names: 0", really 43 files
    ^\\| S[0-9]+ \\|              missed bolded table rows      "11 rows", really 15
    a name-match on one phrase   matched a row about another subject
    ids assumed to start 11/12   the archive starts 10xxx      21 citations invisible
    a case-sensitive alternation "per ri" survived where "per Ri" would not

A needle is written from what its author expects the text to look like, so it can only ever confirm
that expectation. This file does not try to be a better needle. It is DELIBERATELY OVER-BROAD and
its output is a reading list, so a miss costs a person thirty seconds and a false positive costs the
same. The classes below are shapes, not spellings: "a number next to a name", not a digit range.

  ⚠️ WHAT IT CANNOT DO. It cannot see a DISCLOSURE -- a sentence describing a deficiency in our own
  systems contains no name, no number and no path, and the largest one found this week would not
  trip a single rule here. Prose review is a separate control and this does not replace it.

Exit status is 0 unless --strict: routing, not gating.
"""
# filter-repo: do-not-rewrite
# This file DESCRIBES the strings a rewrite removes, so a blanket text replacement mangles its
# own rules — a dry run turned a detector pattern into its own replacement text. The marker is
# read by the rewrite driver and is the only exclusion it honours.

import argparse, os, re, subprocess, sys

# Shapes, each with the reason a human should look -- never a bare pattern.
# A name beside a 4-5 digit number reads as an internal decision id. TWO arms, on purpose.
#
# The first is the ROSTER -- our seals and romaji names -- with a bounded same-line window, so
# it finds a citation written as prose and not only one written tight. Its weakness was stated
# in this file on the day it was widened: a roster is only as complete as the day it was copied.
# That weakness then arrived. The chair's own seal had never been on it, so a real citation sat
# in a published test file and this checker returned a clean zero on the very file carrying it.
# A list cannot be the only arm of a rule whose subject is "any name".
#
# So the second arm carries NO LIST: any one or two CJK ideographs beside a number in the range
# our own numbering occupies. MEASURED over the whole tracked corpus before adopting it. A
# list-free arm with no range bound costs eight false positives against one true hit -- a poet's
# birth year, a font encoding, a LaTeX fraction, an HTML character reference -- and a rule whose
# hits a reader learns to skim is worse than one hit fewer. With the range bound and the two
# lookbehinds: nine hits, all nine real. The bound IS the discriminator, and it is a fact about
# our own numbering rather than about any name: a 4-digit number below 5000 in this corpus is a
# year, an encoding or a quantity, never one of our sequences, and the sequences only grow.
_ID_WINDOW = r'[^\n]{0,40}?'
_ID_ANY_NUM = r'\b(?!19\d\d|20\d\d)\d{4,5}\b'
_ID_ANY_SEAL = r'[㐀-䶿一-鿿]{1,2}'
_ID_OUR_RANGE = r'(?<!&#)(?<!\{)\b(?:[5-9]\d{3}|[1-9]\d{4})\b'
_ID_ROSTER = (r"(?:地図|深海|からくり|[理令匠形案内女将庭鉋目付鎖巳紗鍵雲鉄文沙汰秤信声経宝守栞瞬柱繋燕暦眸窯]{1,2}|\b(?:ri|rei|takumi|katachi|annai|okami|niwa|kanna|metsuke|kusari|"
              r"misa|kagi|kumo|tetsu|fumi|sautee|chizu|fangfei|francis|hakari|hana|hashira|hibiki|jude|kama|karakuri|keiko|koe|koyomi|mamoru|manako|nagare|nagomi|sami|shin|shinkai|shiori|shun|takara|tsubame|tsunagi|weixu)\b)")
ID_PAT = (_ID_ROSTER + _ID_WINDOW + _ID_ANY_NUM
          + '|' + _ID_ANY_SEAL + _ID_WINDOW + _ID_OUR_RANGE)


RULES = [
    ("id-next-to-a-name", ID_PAT,
     "a 4-5 digit number beside a name reads as an internal decision id"),
    # The glyph arm carries the roster's seals EXCEPT 花, 和 and 流: everyday characters that
    # flag a poetry corpus ("花枕 … GB 2312" is a font encoding beside a flower). Those three
    # lanes are reached by the romaji arm only, and that limit is stated here on purpose.
    # ✏️ 2026-10-06: the romaji arm knew 16 of the ~42 names on the roster; widened to the whole
    # roster as the channel registry lists it. A name the registry does not carry is invisible
    # to THIS arm, which is why the list-free arm above exists.
    # ✏️ 2026-10-06: the separator class was adjacency-only, so it matched a seal and a number
    # written tight and missed the same citation written as prose -- a seal, a verb, then the
    # number in brackets. A scan of the whole tracked corpus with a 40-char same-line window
    # found 22 more real ids the narrow form could not see: a possessive seal, a seal with a
    # slash, a seal with an interposed "from" or "amended". Zero false positives in those 22,
    # measured before the change, so the widening cost no signal. The window is bounded and
    # same-line on purpose: unbounded would pair a name with any number in the document.
    # Every arm here USED to require a trailing digit, and the coordination host's name has
    # none -- so the one machine that is the git server and the message store was the single
    # host shape this guard could not express, and it returned a clean zero on a corpus that
    # names it twice. Found by a second reader, validated on the whole corpus rather than on
    # chosen cases: 2 hits, both true, and none of the 109 `mlx-lm` lines. The lookarounds are
    # the file's own idiom -- `fixture-slug` already carries them for the same reason.
    ("host-shaped-token", r'\b(?:llm\d|mini\d|box[-_ ]?[A-Z]\d)\b|(?<![-\w])lm(?![-\w])',  # shape-example
     "a machine name is estate topology: how many boxes there are and what they do"),
    ("internal-path-fragment",
     # ✏️ 2026-10-06 (沙汰): the first two alternatives did NOT cover an agent session's
     # scratchpad, which is `/private/tmp/claude-<uid>/-Users-...-agents-<Lane>/<uuid>/`.
     # 13 published result JSONs carried that full path as a recorded --states/--out value
     # and NO rule here flagged it; the widened id-next-to-a-name rule caught it only by
     # accident, because "Sautee" sat 40 chars from four digits of the UUID. A blind spot
     # demonstrated by 13 files is not a bound to note, it is a gate to add.
     r'(?:reports|wo)/[a-z]+/|/Users/[a-z]+/(?:github|claude)/'
     r'|/private/tmp/claude-\d+/|-agents-[A-Za-z]+/[0-9a-f]{8}-',
     "an internal path names a private tree and often a person"),
    ("product-or-repo-name", r'nira[-_ ]?(?:net|app)|niraikanai',  # shape-example
     "the product and the estate repos are not part of the method"),
    # ✏️ 2026-10-06: `.html` joins `.json`/`.jsonl` in the lookbehinds.
    # Six of this rule's eight hits were the SITE'S OWN PAGE NAME, `discovery-chain.html`,
    # in .gitignore, index.html and stamp_site.sh -- a guard flagging the filename of the
    # page it is published beside. A rule whose hits a reader learns to skim is worse than
    # one hit fewer. The three remaining hits are a DOCUMENT ANCHOR
    # (`appendix-a.-how-this-was-found`) and are NOT excluded  # shape-example
    # the .html-basename exclusion only, and an anchor is a different shape needing its own
    # ruling rather than my widening the exemption while I am in here.
    ("fixture-slug", r'\b[a-z]+-[a-z]+\.[a-z-]{4,}\b(?<!\.json)(?<!\.jsonl)(?<!\.html)',
     "a dotted lowercase slug is usually a venue id"),
    # An internal work-order id discloses that we number work orders and roughly how many there
    # are. The class was ruled worth scrubbing (a 09-23 commit says so in its own subject) and
    # twelve instances survived that pass -- including three in the TITLES of the very pointers
    # whose bodies were rewritten to drop a lane name, a path and an incident. The body was read
    # and the line above it was not. A rule is what stops that being a matter of remembering.
    ("internal-wo-id", r'\bWO-\d{2,4}\b',
     "a work-order id is an internal reference and says how we number our own work"),
    # The PEM arm tolerates SEAMS between its words. A second reader showed that the three rules
    # carrying a literal space are defeated by an inline tag or a line break -- and measured the
    # surface as essentially zero in this corpus, so this is not a live hole. I hardened only this
    # one, on the asymmetry they named: a PEM header is the one string nobody hard-wraps, so its
    # seam risk is the lowest of the three while the cost of a miss is the highest. The other two
    # keep their literal space on purpose; letting arbitrary noise sit between `box` and `A1`
    # buys a rare true positive and a steady supply of false ones.
    ("credential-shaped",
     r'\b(?:AKIA|ASIA)[0-9A-Z]{8,}'
     r'|-----BEGIN[^A-Za-z]{0,40}(?:[A-Z]+[^A-Za-z]{0,40}){0,3}PRIVATE[^A-Za-z]{0,40}KEY',
     "a credential must never be here at all"),
]
SKIP_EXT = (".npz", ".npy", ".png", ".jpg", ".pdf", ".ico", ".woff", ".woff2", ".zip", ".gz",
            ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".mlmodel")
# Binary weights matched three rules on random bytes: a host-shaped token, and a seal beside
# four digits. A check whose output is a reading list must not fill it with tensors nobody can
# read. (The examples are described rather than quoted: this file is scanned like any other.)


_ROOT = None


def repo_root():
    """The toplevel, so every path in this file means the same thing wherever it is invoked.

    `git ls-files` lists the CURRENT DIRECTORY's subtree, not the repository. Run from
    `scripts/` this checker scanned 31 of 8,434 files and printed `TOTAL 0` -- a confident clean
    bill over 0.4% of the tree, because that subtree happens to contain nothing it detects. The
    same cwd assumption reaches the self-exclusion below: `relpath(__file__, getcwd())` matches
    the listing only from the root, so from a subdirectory the checker would also stop excluding
    itself and report its own example patterns as hits.
    """
    global _ROOT
    if _ROOT is None:
        # Cached because read() calls this once per file: the first version spawned a git
        # subprocess 8,434 times and the checker stopped finishing. A helper that is correct
        # and unusable is not a fix.
        r = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        _ROOT = r.stdout.strip() or os.getcwd()
    return _ROOT


def tracked(rev=None):
    cmd = ["git", "ls-tree", "-r", "--name-only", rev] if rev else ["git", "ls-files"]
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=repo_root()).stdout.split()


def read(path, rev=None):
    if rev:
        p = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True)
        return p.stdout.decode("utf8", "replace") if p.returncode == 0 else ""
    try:
        return open(os.path.join(repo_root(), path),
                    encoding="utf8", errors="replace").read()
    except Exception:
        return ""


SHAPE_EXAMPLE = "shape-example"   # the ONE marker that exempts a line of THIS file


def exempt_self_line(line):
    """Does this line of the checker's own source legitimately carry a shape it detects?

    The exemption used to be the whole FILE -- a `continue` on the checker's own path -- on the
    reasoning that a file quoting the shapes it looks for is noise. That reasoning is right
    about the rule literals and wrong about every other line, and under it SIX real decision
    ids accumulated in the editorial comments here, invisible to this tool by construction.
    They were found by a second reader. A guard with a blind spot over its own source is the
    one place a reader will never think to look.

    So the exemption is per LINE and must be written on the line it covers. The default is to
    SCAN: an unmarked line is read exactly like a line of any other file. Marking a line is a
    visible act in a diff, and the number of marked lines is printed with every result, so the
    exemption cannot widen quietly.
    """
    return SHAPE_EXAMPLE in line


def scan(files, rev=None, self_path=None):
    hits = []
    scan.self_exempt = 0
    for f in files:
        # THE PATH IS PUBLISHED TEXT. This guard read only the inside of files, so a file NAMED
        # after a host was invisible to it by construction -- and a skipped binary's path was
        # doubly invisible, since the extension skip took the name out with the bytes. A name
        # is listed by every clone, every tarball and every web view of the tree. Reported at
        # line 0 so a path hit can never be mistaken for a line of content.
        for name, pat, why in RULES:
            m = re.search(pat, f, re.I)
            if m:
                hits.append((f, 0, name, why + " (in the FILE PATH)", m.group(0)[:40], f))
        if f.endswith(SKIP_EXT):
            continue
        body = read(f, rev)
        for i, line in enumerate(body.split("\n"), 1):
            if f == self_path and exempt_self_line(line):
                scan.self_exempt += 1
                continue
            # ✏️ 2026-10-06: this used to `break` after the FIRST matching rule, so a
            # line carrying two classes reported only the earlier one in RULES order -- and
            # 13 result JSONs whose recorded --states path is an agent scratchpad reported
            # only `id-next-to-a-name` (RULES[0]) while `internal-path-fragment` was never
            # evaluated on that same line. The consequence is worse than a miscount: a scrub
            # becomes ITERATIVE, each pass revealing a class the previous pass hid, so a
            # zero after one pass is not a zero. Every rule is now evaluated on every line.
            for name, pat, why in RULES:
                m = re.search(pat, line, re.I)
                if m:
                    hits.append((f, i, name, why, m.group(0)[:40], line.strip()[:100]))
    return hits


def derived_staleness(files):
    """Derived artefacts that are OLDER than the source they were built from.

    This exists because a scrub fixed a markdown file and its HTML and did not rebuild the PDF,
    so the published artefact stayed a pre-scrub copy for three days and served a host name to
    anyone who fetched it. No pattern here could have caught it: the shape was expressible and
    the file was one this checker skips by extension, so the guard's zero was true of everything
    it opened and silent about the one surface that mattered.

    A date comparison catches that class WITHOUT reading the binary at all, and it keeps working
    for any artefact kind -- which a text extractor would not. It is deliberately the weaker,
    duller test: it cannot say what is inside, only that the inside is older than the outside.
    """
    import collections
    srcs = {f[:-3]: f for f in files if f.endswith(".md")}
    out = []
    for f in files:
        stem = f[:-4] if f.endswith((".pdf", ".htm")) else (f[:-5] if f.endswith(".html") else None)
        if stem is None or stem not in srcs:
            continue
        def when(path):
            r = subprocess.run(["git", "log", "-1", "--format=%at", "--", path],
                               capture_output=True, text=True)
            return int(r.stdout.strip() or 0)
        a, b = when(f), when(srcs[stem])
        if a and b and a < b:
            out.append((f, srcs[stem], a, b))
    return out


def completeness_statement(files):
    """What this searched for, printed WITH the result. A zero from an unstated needle set licenses
    "the needle fired and these are its hits" and never a total -- the completeness claim belongs to
    a reader, not to a pattern. Transplanted from the erasure register's own sentence rather than
    invented here.
    """
    return ("\n  WHAT WAS SEARCHED FOR, so the result can be read as what it is:\n"
            + "".join("    - %s\n" % why for _, _, why in RULES)
            + "    over %d files, case-insensitively, with no word boundaries.\n"
            "  An empty result is NOT a statement that this corpus carries no estate reference.\n"
            "  It is a statement that THESE SHAPES are absent. A DISCLOSURE -- a sentence describing\n"
            "  a deficiency in how we handle other people's data -- carries no name, no number and no\n"
            "  path, and would pass every rule above. That claim needs a reader." % len(files))


def self_test():
    """Controls for this checker, because it is the gate every public commit passes through.

    It shipped without any. The defect that prompted them was found by a second reader on a
    published file: a lane pseudonym beside a channel sequence number, which this tool returned
    a clean zero on, because its name predicate was a LIST and the chair's own seal had never
    been added to it.

    The FIRING fixtures keep the seal and the number in separate literals and join them at run
    time, so this file stays scannable by its own rules rather than needing an exemption for
    every fixture. The NON-FIRING fixtures are the eight real false positives measured over the
    tracked corpus while choosing the list-free arm -- they are kept as tests because a rule
    whose hits a reader learns to skim is worse than one hit fewer.
    """
    RAN = []

    def ck(name, got, want):
        ok = (got == want)
        RAN.append(ok)
        def s(v):
            r = repr(v)
            return r[:70] + u"\u2026" if len(r) > 70 else r
        print(u"  %s %-64s %s" % (u"\u2705" if ok else u"\U0001f534", name,
                                 u"" if ok else u"got %s, want %s" % (s(got), s(want))))

    BY = dict((n, p) for n, p, w in RULES)

    def fires(rule, line):
        return re.search(BY[rule], line, re.I) is not None

    # Seals and sequences, deliberately apart -- and the KEYS are roles, not names: a table
    # keyed by lane name put a romaji name within the window of a four-figure number and this
    # file flagged its own fixtures. Control 5c found that, which is the point of control 5c.
    seq = {"chair": "16481", "ruling": "15458", "second": "15683", "form": "12340",
           "studio": "10912", "ruling_b": "11104", "ruling_c": "13178", "older": "9170",
           "offroster": "15000", "below_range": "4321"}

    print(u"\n=== id-next-to-a-name: THE LEAK THAT PROMPTED THIS, as a fixture ===")
    leaked = (u'print("\\n=== CONTROL 2b \u2014 the presentation order is a property of the '
              u'CELL (\u7a76 ' + seq["chair"] + u') ===")')
    ck("1. the exact published line that leaked now fires", fires("id-next-to-a-name", leaked), True)
    ck("1b. the chair's seal beside a sequence fires on its own",
       fires("id-next-to-a-name", u"\u7a76 " + seq["chair"]), True)
    ck("1c. ...and the chair's seal is NOT on the roster arm, so arm two is what caught it",
       re.search(_ID_ROSTER + _ID_WINDOW + _ID_ANY_NUM,
                 u"\u7a76 " + seq["chair"], re.I) is not None, False)
    ck("1d. a seal no roster will ever carry fires too",
       fires("id-next-to-a-name", u"\u9df9 " + seq["offroster"]), True)

    print(u"\n=== id-next-to-a-name: the roster arm still does its own work ===")
    for who, s in (("a ruling", seq["ruling"]), ("a second read", seq["second"])):
        ck("2. a roster seal beside a sequence (%s)" % who,
           fires("id-next-to-a-name", u"\u7406 " + s), True)
    ck("2b. a possessive seal", fires("id-next-to-a-name", u"\u5f62's " + seq["form"]), True)
    ck("2c. a seal with an interposed preposition",
       fires("id-next-to-a-name", u"\u7406, from \u6c99\u6c70\u2013" + seq["studio"]), True)
    ck("2d. a seal with a slash", fires("id-next-to-a-name", u"\u7406/" + seq["ruling_b"]), True)
    ck("2e. a citation written as prose, which adjacency-only missed",
       fires("id-next-to-a-name", u"\u7406 defined it (" + seq["ruling_c"] + u" \u00a72)"), True)
    ck("2f. a romaji name", fires("id-next-to-a-name", "sautee " + seq["chair"]), True)
    ck("2g. a 4-digit id BELOW the list-free arm's range, which only the roster can catch",
       fires("id-next-to-a-name", u"\u7406 " + seq["below_range"]), True)
    ck("2h. ...and the list-free arm indeed declines that one, as designed",
       re.search(_ID_ANY_SEAL + _ID_WINDOW + _ID_OUR_RANGE,
                 u"\u7406 " + seq["below_range"], re.I) is not None, False)

    print(u"\n=== id-next-to-a-name: the eight measured false positives stay quiet ===")
    quiet = [
        ("a poet's birth year beside his name", u"a Tang Yin (\u5510\u5bc5, 1470\u20131524) voice prior"),
        ("a font encoding beside a flower", u"\u82b1\u6795 Fonts: \u6977\u9ad4 GB 2312"),
        ("a LaTeX fraction after a glyph", u"\u5f0f: \\frac{4321}{9999}"),
        ("an HTML character reference", u"\u9999 '&#26412"),
        ("a DATE citation, which the chair ruled stays", u"\u7406 2026-09-14"),
        ("a bare year", u"\u7406 1999"),
        ("an Elo ladder with no name near it", "--ladder 2400,2700,3000"),
        ("a 4-digit quantity with no name on the line", "--ckpt-every 4000 --seed 3"),
    ]
    for why, line in quiet:
        ck("3. quiet: %s" % why, fires("id-next-to-a-name", line), False)

    print(u"\n=== every other rule has a firing fixture and a quiet one ===")
    # Each row is (rule, a line that MUST fire, a line that must not). These eight lines quote
    # the shapes verbatim -- a fixture written in words tests nothing -- so they carry the
    # marker, and the count printed with every result is how a reader sees that they do.
    other = [
        ("host-shaped-token", "llm1", "mlx-lm"),  # shape-example
        ("host-shaped-token", "trained on lm", "html-lm-x"),  # shape-example
        ("internal-path-fragment", "reports/sautee/et7/x.md", "reasoning/results/x.md"),  # shape-example
        ("internal-path-fragment", "/private/tmp/claude-501/x", "/private/tmp/claude-x/y"),  # shape-example
        ("product-or-repo-name", "the niraikanai tree", "nirvana apps"),  # shape-example
        ("internal-wo-id", "WO-336 says", "WO-"),  # shape-example
        ("fixture-slug", "venue-name.some-slug", "results.json"),  # shape-example
        ("credential-shaped", "AKIAEXAMPLE1234567", "AKIA"),  # shape-example
    ]
    for rule, hot, cold in other:
        ck("4. %-22s fires" % rule, fires(rule, hot), True)
        ck("4. %-22s quiet" % rule, fires(rule, cold), False)

    print(u"\n=== the self-exemption: per LINE, and it cannot hide a new id ===")
    ck("5. an unmarked editorial citation in this file is NOT exempt",
       exempt_self_line(u"    # \u270f\ufe0f 2026-10-08 (\u7a76 " + seq["chair"] + u"): widened"), False)
    ck("5b. a line carrying the marker IS exempt",
       exempt_self_line("    a rule literal, in full, on this line  # " + SHAPE_EXAMPLE), True)
    me = os.path.relpath(os.path.abspath(__file__), repo_root())
    ck("5c. this checker is CLEAN on its own source, marker lines aside",
       scan([me], self_path=me), [])
    marked = scan.self_exempt
    ck("5d. ...and it says how many of its own lines it did not read", marked > 0, True)

    real_read = read
    try:
        globals()["read"] = lambda path, rev=None: (
            u"a rule literal naming the estate repo, in full  # " + SHAPE_EXAMPLE + u"\n"
            u"# \u270f\ufe0f 2026-10-08 (\u7a76 " + seq["chair"] + u"): a real id, unmarked\n")
        hits = scan([me], self_path=me)
    finally:
        globals()["read"] = real_read
    ck("5e. a REAL id added to this file is reported (the defect that hid six of them)",
       [(h[1], h[2]) for h in hits], [(2, "id-next-to-a-name")])
    ck("5f. ...while the marked rule literal beside it stays exempt, counted not silent",
       scan.self_exempt, 1)

    print(u"\n=== a selector that cannot testify beyond itself ===")
    ck("6. an unreadable --files path yields no content, so it must be refused not counted",
       missing_paths(["reasoning/et7_ee_probe.py", "no/such/file.py"]), ["no/such/file.py"])
    ck("6b. ...and a path that is there is not refused",
       missing_paths(["reasoning/et7_ee_probe.py"]), [])

    print(u"\n  %d/%d controls pass." % (sum(RAN), len(RAN)))
    return 0 if all(RAN) else 1


def missing_paths(files):
    """--files paths this process cannot read.

    `read()` returns "" for anything it cannot open, so a mistyped path used to produce a
    confident `0 line(s)` over a file that was never opened -- a clean bill whose scope was
    empty. The caller refuses instead of counting.
    """
    return [f for f in files
            if not os.path.isfile(os.path.join(repo_root(), f))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="", help="scan a git revision instead of the working tree")
    ap.add_argument("--files", nargs="*", help="only these paths")
    ap.add_argument("--strict", action="store_true", help="exit non-zero when anything is flagged")
    ap.add_argument("--summary", action="store_true", help="counts per rule, not the lines")
    ap.add_argument("--self-test", action="store_true", help="controls for this checker itself")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    rev = a.rev or None
    files = a.files or tracked(rev)
    if a.files and not rev:
        gone = missing_paths(a.files)
        if gone:
            print("  REFUSED: %d of the %d paths given cannot be opened, so a clean result over\n"
                  "  them would be a statement about nothing:" % (len(gone), len(a.files)))
            for f in gone:
                print("    %s" % f)
            sys.exit(2)
    me = os.path.relpath(os.path.abspath(__file__), repo_root())
    hits = scan(files, rev, self_path=me)
    if a.summary:
        import collections
        c = collections.Counter(h[2] for h in hits)
        for name, _, why in RULES:
            print("  %-24s %4d   %s" % (name, c.get(name, 0), why))
        print("  %-24s %4d   files scanned: %d" % ("TOTAL", len(hits), len(files)))
    else:
        for f, i, name, why, tok, line in hits:
            print("%s:%d  [%s] %r\n    %s\n    -> %s" % (f, i, name, tok, line, why))
        print("\n%d line(s) for a person to read, over %d files." % (len(hits), len(files)))
    print(completeness_statement(files))
    if getattr(scan, "self_exempt", 0):
        print("\n  %d line(s) of %s are marked `%s` and were NOT scanned: they define or\n"
              "  describe a shape this checker looks for. Every other line of it was read."
              % (scan.self_exempt, me, SHAPE_EXAMPLE))

    # WHAT WAS NOT OPENED, said as a number rather than left to the prose above. A guard that
    # skips a surface silently reports a zero that is true of everything it read and says
    # nothing about the rest; naming the count is what stops that zero from travelling.
    skipped = [f for f in files if f.endswith(SKIP_EXT)]
    if skipped:
        print("\n  NOT OPENED (binary by extension): %d file(s). Their PATHS were checked above;\n"
              "  their CONTENTS were not read by anything here." % len(skipped))
    # WHAT IS NOT IN THE SET AT ALL, for the same reason. The default file list is `git ls-files`,
    # so a file that is written but not yet committed is invisible here -- and that is exactly the
    # file somebody is about to push. I read a clean TOTAL over a new scorer this evening and the
    # zero was true only because the file was untracked. The count is printed so that zero cannot
    # travel again; --files scans them on demand, before the commit rather than after it.
    if not a.files and not rev:
        others = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                capture_output=True, text=True, cwd=repo_root()).stdout.split()
        if others:
            print("\n  NOT IN THIS SCAN (present, untracked): %d file(s) -- a clean result above says\n"
                  "  NOTHING about them, and an uncommitted file is the one about to be pushed:"
                  % len(others))
            for f in others[:12]:
                print("    %s" % f)
            if len(others) > 12:
                print("    ... and %d more" % (len(others) - 12))
            print("    scan them before committing:  %s --files %s"
                  % (os.path.relpath(os.path.abspath(__file__), repo_root()), " ".join(others[:3])
                     + (" ..." if len(others) > 3 else "")))
    stale = derived_staleness(files)
    if stale:
        print("\n  DERIVED ARTEFACT OLDER THAN ITS SOURCE -- rebuild before trusting a clean scan:")
        for f, src, _, _ in stale:
            print("    %s  is older than  %s" % (f, src))
    elif not a.files:
        print("  Every derived artefact with a markdown source post-dates it.")
    if (hits or stale) and a.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
