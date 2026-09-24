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
RULES = [
    ("id-next-to-a-name",
     # OUR glyphs, not any CJK: "唐寅, 1470" is a poet's birth year and flagging it teaches
     # the reader to ignore this rule, which is the only way a routing check really fails.
     r'(?:[理令匠形案内女将庭鉳目付鎖巳紗鍵雲鉄文沙汰]{1,2}|\b(?:ri|rei|takumi|katachi|annai|okami|niwa|kanna|metsuke|kusari|'
     r'misa|kagi|kumo|tetsu|fumi|sautee)\b)[\s,(]*(?!19\d\d|20\d\d)\d{4,5}\b',
     "a 4-5 digit number beside a name reads as an internal decision id"),
    # Every arm here USED to require a trailing digit, and the coordination host's name has
    # none -- so the one machine that is the git server and the message store was the single
    # host shape this guard could not express, and it returned a clean zero on a corpus that
    # names it twice. Found by a second reader, validated on the whole corpus rather than on
    # chosen cases: 2 hits, both true, and none of the 109 `mlx-lm` lines. The lookarounds are
    # the file's own idiom -- `fixture-slug` already carries them for the same reason.
    ("host-shaped-token", r'\b(?:llm\d|mini\d|box[-_ ]?[A-Z]\d)\b|(?<![-\w])lm(?![-\w])',
     "a machine name is estate topology: how many boxes there are and what they do"),
    ("internal-path-fragment", r'(?:reports|wo)/[a-z]+/|/Users/[a-z]+/(?:github|claude)/',
     "an internal path names a private tree and often a person"),
    ("product-or-repo-name", r'nira[-_ ]?(?:net|app)|niraikanai',
     "the product and the estate repos are not part of the method"),
    ("fixture-slug", r'\b[a-z]+-[a-z]+\.[a-z-]{4,}\b(?<!\.json)(?<!\.jsonl)',
     "a dotted lowercase slug is usually a venue id"),
    ("credential-shaped", r'\b(?:AKIA|ASIA)[0-9A-Z]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY',
     "a credential must never be here at all"),
]
SKIP_EXT = (".npz", ".npy", ".png", ".jpg", ".pdf", ".ico", ".woff", ".woff2", ".zip", ".gz",
            ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".mlmodel")
# Binary weights matched three rules on random bytes ("llm9", "\u7a7a8549"). A check whose output
# is a reading list must not fill it with tensors nobody can read.


def tracked(rev=None):
    cmd = ["git", "ls-tree", "-r", "--name-only", rev] if rev else ["git", "ls-files"]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.split()


def read(path, rev=None):
    if rev:
        p = subprocess.run(["git", "show", "%s:%s" % (rev, path)], capture_output=True)
        return p.stdout.decode("utf8", "replace") if p.returncode == 0 else ""
    try:
        return open(path, encoding="utf8", errors="replace").read()
    except Exception:
        return ""


def scan(files, rev=None, self_path=None):
    hits = []
    for f in files:
        if f == self_path:
            continue           # this file QUOTES the shapes it looks for; scanning it is noise
        # THE PATH IS PUBLISHED TEXT. This guard read only the inside of files, so a file NAMED
        # after a host was invisible to it by construction -- and a skipped binary's path was
        # doubly invisible, since the extension skip took the name out with the bytes. A name
        # is listed by every clone, every tarball and every web view of the tree. Reported at
        # line 0 so a path hit can never be mistaken for a line of content.
        for name, pat, why in RULES:
            m = re.search(pat, f, re.I)
            if m:
                hits.append((f, 0, name, why + " (in the FILE PATH)", m.group(0)[:40], f))
                break
        if f.endswith(SKIP_EXT):
            continue
        body = read(f, rev)
        for i, line in enumerate(body.split("\n"), 1):
            for name, pat, why in RULES:
                m = re.search(pat, line, re.I)
                if m:
                    hits.append((f, i, name, why, m.group(0)[:40], line.strip()[:100]))
                    break
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="", help="scan a git revision instead of the working tree")
    ap.add_argument("--files", nargs="*", help="only these paths")
    ap.add_argument("--strict", action="store_true", help="exit non-zero when anything is flagged")
    ap.add_argument("--summary", action="store_true", help="counts per rule, not the lines")
    a = ap.parse_args()
    rev = a.rev or None
    files = a.files or tracked(rev)
    me = os.path.relpath(os.path.abspath(__file__), os.getcwd())
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

    # WHAT WAS NOT OPENED, said as a number rather than left to the prose above. A guard that
    # skips a surface silently reports a zero that is true of everything it read and says
    # nothing about the rest; naming the count is what stops that zero from travelling.
    skipped = [f for f in files if f.endswith(SKIP_EXT)]
    if skipped:
        print("\n  NOT OPENED (binary by extension): %d file(s). Their PATHS were checked above;\n"
              "  their CONTENTS were not read by anything here." % len(skipped))
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
