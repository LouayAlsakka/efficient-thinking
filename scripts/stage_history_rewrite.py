#!/usr/bin/env python3
"""STAGE a history rewrite: build its inputs and its acceptance test. EXECUTES NOTHING.

WHAT THIS IS FOR. Removal-by-commit is not removal in a published repository -- the earlier version
of a file is one ordinary command away, and an unauthenticated fetch of the parent sha returns it.
A rewrite is the only thing that changes that, and it is a decision above this lane. So this builds
the three artefacts the decision needs and stops:

  purge-request.txt  🔴 THE FIRST STEP, NOT THE LAST. Measured on this repository: the host's
                     public events feed hands out 321 commit ids with no credentials, and ALL 321
                     serve text HEAD no longer has. A force-push moves the tip; without the purge
                     it changes nothing a stranger can reach, and nobody needs to have cloned.
  replacements.txt   every sensitive literal mapped to the replacement it ALREADY HAS at HEAD
  acceptance.sh      the check that the rewrite changed history and did NOT change HEAD

WHY TEXT REPLACEMENT RATHER THAN HUNK SURGERY. The strings were introduced across the whole
timeline, not in the commits that removed them -- a host name enters on the repository's first day
and is present in essentially every tree after it. Dropping hunks from the removing commits leaves
the content in the ~700 commits before them. Replacing each literal everywhere converges on the
state HEAD is already in, which is also why the acceptance test is cheap: HEAD contains none of
these strings today, so a correct rewrite leaves the HEAD tree BYTE-IDENTICAL.

THE MAP IS DERIVED FROM THE SCRUBS THEMSELVES, not re-guessed. Every scrub commit's diff pairs a
removed line with the line that replaced it; the literals come from those pairs. A map rebuilt from
memory would carry exactly the band and case assumptions that made two of those scrubs miss.

⚠️ WHAT A REWRITE CANNOT DO, stated here so it travels with the script: it cannot recall what an
anonymous clone already fetched; the host keeps force-pushed commits reachable BY SHA until its own
collection runs; and every sha ever quoted in a decision record dies with the rewrite, including
those quoted where no map can reach them.

🔴 AND THE SHAs ARE PUBLISHED BY THE HOST ITSELF. The usual comfort about "reachable by sha" is
"but who has the sha?" -- and the answer here is anyone who reads a public endpoint: 300 events
over seven days, 321 distinct ids, `before` on each push being the pre-push tip, which is by
construction the list a rewrite is meant to orphan. That feed is a residue store OUTSIDE the
repository, so the sha-map cannot reach it and neither can any edit we make.
"""
# filter-repo: do-not-rewrite
# This file DESCRIBES the strings a rewrite removes, so a blanket text replacement mangles its
# own rules — a dry run turned a detector pattern into its own replacement text. The marker is
# read by the rewrite driver and is the only exclusion it honours.

import argparse, collections, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*a, **kw):
    return subprocess.run(["git", "-C", HERE] + list(a), capture_output=True, text=True, **kw).stdout


def scrub_commits():
    """The commits whose subject says they removed something — the map's source."""
    out = []
    for line in git("log", "--format=%H\t%s").split("\n"):
        if not line.strip():
            continue
        h, s = line.split("\t", 1)
        if re.search(r'scrub|data scope|untrack|host name', s, re.I):
            out.append((h, s))
    return out


def literal_pairs(commits):
    """Hunks from the scrub diffs, split into what CAN be replaced automatically and what cannot.

    ⚠️ MULTI-LINE PROSE DOES NOT PAIR LINE BY LINE. A three-line paragraph rewritten into three
    lines re-wraps, so removed line 1 and added line 1 are not each other's replacement -- pairing
    them positionally and applying that across history would garble the text in every commit it
    touched. Caught by reading the generated map rather than by trusting the derivation.

    So: a hunk of exactly ONE removed and ONE added line is a safe literal. Anything wider is
    returned separately and goes to a human, who is the only thing that can pair prose.
    """
    safe, wide = [], []
    for h, _ in commits:
        d = git("show", "--format=", "--unified=0", h)
        rem, add = [], []

        def flush():
            if not rem and not add:
                return
            if len(rem) == 1 and len(add) == 1:
                safe.append((rem[0], add[0]))
            elif rem:
                wide.append((h, list(rem), list(add)))

        for l in d.split("\n"):
            if l.startswith("---") or l.startswith("+++"):
                continue
            if l.startswith("-"):
                rem.append(l[1:])
            elif l.startswith("+"):
                add.append(l[1:])
            else:
                flush(); rem, add = [], []
        flush()
    return safe, wide


# Shapes that must be replaced wherever they occur, regardless of which line they sat on.
REGEX_RULES = [
    (r'regex:\bllm1\b', 'box A'),
    (r'regex:\bllm2(\.local)?\b', 'box B'),
    (r'regex:\bmini\d+\b', 'a mini'),
    (r'regex:(理|令|匠|形|案内|女将|庭|鉋|目付|鎖|巳紗|鍵|雲|鉄|文|沙汰)[\s,]+\(?(?!19\d\d|20\d\d)\d{4,5}\)?', r'\1'),
    (r'regex:\bnirai\s+\d{4,5}\b', 'the record'),
    (r'literal:niraikanai', 'the record'),
    (r'literal:NiraNet', 'the product'),
    (r'literal:nira-app', 'the app'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "_rewrite_staging"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    # 🔴 THE OUTPUT CONTAINS THE SENSITIVE STRINGS THEMSELVES. A map of "what to remove" is a
    # complete list of what was removed and where; writing it inside the repository would publish
    # in one file exactly what the rewrite exists to take out. Refused, not defaulted away from.
    repo_real = os.path.realpath(HERE)
    if os.path.realpath(out).startswith(repo_real + os.sep) or os.path.realpath(out) == repo_real:
        sys.exit("REFUSING to write the staging inside the repository (%s).\n"
                 "  The map lists every sensitive literal and its replacement — committing it\n"
                 "  would publish, in one file, the whole of what the rewrite is meant to remove.\n"
                 "  Choose a path outside the working tree." % out)
    os.makedirs(out, exist_ok=True)

    cs = scrub_commits()
    pairs, wide = literal_pairs(cs)
    # keep only pairs whose REMOVED side holds something the HEAD tree no longer contains
    head_blob = "\n".join(
        open(os.path.join(HERE, f), encoding="utf8", errors="replace").read()
        for f in git("ls-files").split()
        if f.endswith((".md", ".py", ".json", ".sh", ".txt", ".mjs")))
    lits = []
    for rem, add in pairs:
        rem, add = rem.strip(), add.strip()
        if len(rem) < 24 or rem in head_blob or rem == add:
            continue
        lits.append((rem, add))
    seen, uniq = set(), []
    for r, a2 in lits:
        if r not in seen:
            seen.add(r); uniq.append((r, a2))

    with open(os.path.join(out, "replacements.txt"), "w", encoding="utf8") as f:
        f.write("# Derived from the scrub diffs themselves, not rebuilt from memory.\n")
        f.write("# Shape rules first; then the literal lines those scrubs actually replaced.\n")
        for pat, rep in REGEX_RULES:
            f.write("%s==>%s\n" % (pat, rep))
        for r, a2 in uniq:
            f.write("literal:%s==>%s\n" % (r, a2 if a2 else ""))
    print("  replacements.txt: %d shape rules + %d SAFE literal lines from %d scrub commits"
          % (len(REGEX_RULES), len(uniq), len(cs)))

    # The part a script must not do on its own.
    wpath = os.path.join(out, "needs-a-human.txt")
    with open(wpath, "w", encoding="utf8") as f:
        f.write("# MULTI-LINE HUNKS FROM THE SCRUBS. A paragraph rewritten into a paragraph does\n"
                "# not pair line by line -- it re-wraps -- so these cannot become literal rules\n"
                "# without garbling every commit they touch. A person pairs them, or the block is\n"
                "# replaced whole. They are the DISCLOSURE rewrites, which is to say the ones that\n"
                "# matter most.\n\n")
        for h, rem, add in wide:
            f.write("commit %s\n" % h[:12])
            for l in rem:
                f.write("  -  %s\n" % l)
            for l in add:
                f.write("  +  %s\n" % l)
            f.write("\n")
    print("  needs-a-human.txt: %d multi-line hunks a script must NOT pair automatically"
          % len(wide))

    with open(os.path.join(out, "acceptance.sh"), "w") as f:
        f.write('''#!/usr/bin/env bash
# THE ACCEPTANCE TEST. Run on the rewritten CLONE with the original as $ORIG.
# HEAD must not change: it already contains none of these strings, so a correct rewrite converges
# on the state it is in. History must change: if no earlier tree lost anything, nothing happened.
set -uo pipefail
ORIG=${ORIG:?set ORIG to the untouched clone}
NEW=${NEW:-.}
echo "== HEAD trees (must be identical except deliberately removed paths)"
diff <(git -C "$ORIG" ls-tree -r HEAD --name-only) <(git -C "$NEW" ls-tree -r HEAD --name-only) \\
  | grep -v 'ledger' || echo "   identical"
for f in $(git -C "$NEW" ls-tree -r HEAD --name-only); do
  a=$(git -C "$ORIG" rev-parse "HEAD:$f" 2>/dev/null); b=$(git -C "$NEW" rev-parse "HEAD:$f" 2>/dev/null)
  [ "$a" = "$b" ] || echo "   CHANGED AT HEAD: $f"
done
echo "== history (must have changed, or the rewrite did nothing)"
for s in llm1 llm2 niraikanai NiraNet; do
  printf "   %-12s before %4s commits   after %4s\\n" "$s" \\
    "$(git -C "$ORIG" log --format=%H -S"$s" -- . | wc -l | tr -d ' ')" \\
    "$(git -C "$NEW"  log --format=%H -S"$s" -- . | wc -l | tr -d ' ')"
done
echo "== every rewritten sha must appear in the map"
test -s sha-map.txt && echo "   sha-map.txt: $(wc -l < sha-map.txt | tr -d ' ') entries" || echo "   MISSING sha-map.txt"
''')
    os.chmod(os.path.join(out, "acceptance.sh"), 0o755)

    with open(os.path.join(out, "purge-request.txt"), "w", encoding="utf8") as f:
        f.write("""To: the hosting provider's support

Subject: request to garbage-collect unreachable objects after a history rewrite

We have rewritten the history of a repository we own and force-pushed the result. The previous
commits are no longer reachable from any ref, but we understand they remain retrievable by their
object id until your own collection runs, and that a cached view can still serve them.

Please run garbage collection on this repository so the unreachable objects are removed, and
confirm when it has completed. We are not asking for anything to be removed from anyone else's
copy; we are asking that our own repository stop serving objects that are no longer referenced.

Repository: <owner>/<name>
Approximate time of the force-push: <fill in>
""")
    print("  acceptance.sh and purge-request.txt written")
    print("\n  STAGED ONLY. Nothing was rewritten, fetched or pushed. Output: %s" % out)


if __name__ == "__main__":
    main()
