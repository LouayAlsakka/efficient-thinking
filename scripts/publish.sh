#!/usr/bin/env bash
# The control for the failure that actually happened: `git add -A` swept 653 unintended files into
# a push to a PUBLIC remote, and I checked them AFTER they were public. A pattern check cannot
# prevent that -- the files were clean; the problem was that nobody looked at the LIST.
#
# So this refuses on the two conditions that made it possible, then shows exactly what becomes
# public and asks. It is a wrapper, not a hook: a hook that can be skipped with --no-verify is a
# suggestion, and this is meant to be the only way this lane pushes.
#
#   ./scripts/publish.sh            show what would become public, then stop
#   ./scripts/publish.sh --push     the same, then push if the checks pass and you confirm
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
REMOTE=${REMOTE:-origin}; BRANCH=$(git rev-parse --abbrev-ref HEAD)
GO=0; [ "${1:-}" = "--push" ] && GO=1

fail(){ echo "REFUSING: $*" >&2; exit 1; }

# ---- 1. nothing may be in flight -------------------------------------------------------------
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the working tree is dirty. Commit or stash first; a push that races an edit publishes a state nobody reviewed."

# ---- 2. untracked files are the `git add -A` trap, so they are named BEFORE the push -----------
UNTRACKED=$(git status --porcelain --untracked-files=all | awk '$1=="??"{print $2}')
if [ -n "$UNTRACKED" ]; then
  echo "UNTRACKED and NOT ignored — these are what a stray 'git add -A' would publish:"
  printf '  %s\n' $UNTRACKED
  echo "  Add them to .gitignore, commit them on purpose, or remove them. Not deciding for you."
  fail "$(printf '%s\n' $UNTRACKED | wc -l | tr -d ' ') untracked path(s)"
fi

# ---- 3. what actually becomes public ----------------------------------------------------------
git fetch -q "$REMOTE" "$BRANCH" 2>/dev/null || true
RANGE="$REMOTE/$BRANCH..HEAD"
N=$(git rev-list --count "$RANGE" 2>/dev/null || echo 0)
[ "$N" -gt 0 ] || { echo "nothing to push."; exit 0; }
echo "== $N commit(s) about to become public on $REMOTE/$BRANCH"
git log --oneline "$RANGE" | sed 's/^/   /'
echo "== files, by what happens to them"
git diff --name-status "$REMOTE/$BRANCH...HEAD" | awk '{c[substr($1,1,1)]++} END {for (k in c) printf "   %s  %d\n", k, c[k]}'
ADDED=$(git diff --name-status "$REMOTE/$BRANCH...HEAD" | awk '$1 ~ /^A/ {print $2}')
NA=$(printf '%s\n' $ADDED | grep -c . || true)
if [ "${NA:-0}" -gt 0 ]; then
  echo "== NEWLY PUBLIC FILES ($NA) — read this list, it is the one nobody read last time:"
  printf '%s\n' $ADDED | sed 's/^/   /' | head -60
  [ "$NA" -gt 60 ] && echo "   … and $((NA-60)) more"
fi

# ---- 4. the shape check, on the new files only ------------------------------------------------
if [ "${NA:-0}" -gt 0 ]; then
  echo "== estate-reference shapes in the newly public files"
  python3 scripts/public_repo_check.py --files $ADDED --summary || true
fi

[ "$GO" -eq 1 ] || { echo; echo "(dry run — re-run with --push)"; exit 0; }
printf 'push these %s commit(s)? [y/N] ' "$N"; read -r ans
[ "$ans" = "y" ] || fail "not confirmed"
git push "$REMOTE" "$BRANCH"
