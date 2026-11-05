#!/bin/bash
# Report the share of the latest commits that change the core.
#
# Usage: scripts/core-share.sh [--csv FILE] [--window N] [ref]
#
# Capability is meant to land in files (tools, skills, templates, hooks) and
# not in the core, so the target is that at most 20% of the latest 50 commits
# of main change crates/core/. A commit here is one of the first-parent chain
# from ref (HEAD by default) in the repository of the current directory: the
# commits main is made of, one for each change that landed on it, as a squash
# or as a merge. A merge's change is its diff against its first parent, so the
# commits it brought in are part of that change and not counted on their own.
# A commit touches the core when that diff (against the empty tree, for a
# root commit) adds, removes or changes any file under crates/core/, its mode
# included. No rename is detected: a file moved out of the core is removed
# there and added elsewhere.
#
# --csv writes FILE, replacing what it held: one row for each commit of the
# chain, newest first, with its sha, committer date and subject, touches_core
# (1 or 0), the lines its diff adds and removes under crates/core/ and
# elsewhere, and the rolling share at it: how many of it and the N - 1
# commits before it touch the core, and that count as a percentage of N
# rounded to a tenth, a half up. Both are empty for the oldest N - 1 commits,
# which have fewer than N to count. N is --window, 50 by default. A file git
# takes as binary counts as touched, with no lines. Git takes a file as
# binary by its content, by its git attributes, or by its size over
# core.bigFileThreshold. Attributes files and git configuration set the last
# two, a user's or the system's as well as the repository's, so they can
# change the line counts, though not which commits touch the core.
#
# The script prints the share at ref, with its count and the target, then how
# many commits over the target it is, or that the target is met; for a chain
# shorter than the window, it prints the chain's length and the target, and
# no share. Then it prints what ref's own commit changes. In GitHub Actions it
# adds the same lines to the job's summary. The share is reported and never
# fails the script. CI's core-share job runs it on every push to main and
# every pull request. On a pull request that job checks out GitHub's test
# merge of the pull request into its base, whose first parent is the base: its
# diff against that parent is the whole pull request as a squash merge would
# land it, so the newest row is the pull request and the share is the one its
# base would have with it.
#
# Exits 0 when it measured, 1 when it cannot (no repository, a ref that names
# no commit, a shallow clone, whose oldest commit would read as adding every
# file, or a file it cannot write), and 2 on a usage error.
set -u -o pipefail

window=50
target=20
core=crates/core/

usage() {
  echo "usage: $0 [--csv FILE] [--window N] [ref]" >&2
  exit 2
}
fail() {
  if [ -n "${GITHUB_ACTIONS:-}" ]; then echo "::error::$1" >&2; else echo "error: $1" >&2; fi
  exit 1
}

csv= ref=HEAD named=
while [ $# -gt 0 ]; do
  case $1 in
    --csv) [ $# -ge 2 ] && [ -n "$2" ] || usage; csv=$2; shift 2 ;;
    --window) [ $# -ge 2 ] || usage; window=$2; shift 2 ;;
    -*) usage ;;
    *) [ -z "$named" ] || usage; ref=$1 named=1; shift ;;
  esac
done
# At most nine digits, so no sum below can overflow.
case $window in
  '' | 0* | *[!0-9]* | ??????????*)
    echo "error: --window takes a count of commits from 1 to 999999999, not '$window'" >&2
    exit 2 ;;
esac

# Whether the clone is shallow, then the commit ref names. Outside a
# repository git says so itself and exits 128.
resolved=$(git rev-parse --is-shallow-repository --verify --quiet "$ref^{commit}")
case $? in
  0) ;;
  1) fail "$ref names no commit" ;;
  *) exit 1 ;;
esac
commit=${resolved#*$'\n'}
short=${commit:0:12}
[ "${resolved%%$'\n'*}" = false ] ||
  fail "a shallow clone: its oldest commit would read as adding every file; fetch the whole history"

# The chain, newest first, each commit with its first parent, the pair that
# git diff-tree reads as the diff to take; a root commit stands alone, and
# --root diffs it against the empty tree.
chain=$(git rev-list --first-parent --parents "$commit") || fail "git rev-list failed on $ref"
pairs= n=0
while read -r sha parent _; do
  expected[n]=$sha n=$((n + 1))
  pairs+="$sha${parent:+ $parent}"$'\n'
done <<< "$chain"

# For each pair, in order: \x01, the sha, \x01, the committer date, \x01, the
# subject, then NUL, then each file it changes as added<TAB>removed<TAB>path
# and NUL, the first after a newline. These are plumbing commands, so
# settings such as diff.renames, diff.relative and log.showRoot do not change
# them, and --encoding holds the subject to UTF-8 whatever
# i18n.logOutputEncoding says. --always prints a commit that changes nothing,
# and -z prints every path as it is, unquoted. The output goes to a file in
# TMPDIR, or /tmp, removed on exit; the template names that directory because
# macOS mktemp, given none, ignores TMPDIR.
diff=$(mktemp "${TMPDIR:-/tmp}/core-share.XXXXXX") || exit 1
trap 'rm -f "$diff"' EXIT
printf '%s' "$pairs" |
  git diff-tree --stdin --root --always -r --no-renames --numstat -z --encoding=UTF-8 \
    --format='%x01%H%x01%cI%x01%s' > "$diff" ||
  fail "git diff-tree failed on $ref"

rows=0
unread() { fail "git diff-tree printed what this script does not read, after $rows commits: $1"; }
while IFS= read -r -d '' record || [ -n "$record" ]; do
  case $record in
    $'\x01'*)
      record=${record#?}
      sha=${record%%$'\x01'*} record=${record#*$'\x01'}
      [ "$rows" -lt "$n" ] && [ "$sha" = "${expected[rows]}" ] || unread "$sha"
      shas[rows]=$sha dates[rows]=${record%%$'\x01'*} subjects[rows]=${record#*$'\x01'}
      touches[rows]=0 core_added[rows]=0 core_removed[rows]=0 other_added[rows]=0 other_removed[rows]=0
      rows=$((rows + 1))
      ;;
    *)
      [ "$rows" -gt 0 ] || unread "$record"
      i=$((rows - 1)) entry=${record#$'\n'}
      case $entry in *$'\t'*$'\t'*) ;; *) unread "$record" ;; esac
      added=${entry%%$'\t'*} entry=${entry#*$'\t'}
      removed=${entry%%$'\t'*} path=${entry#*$'\t'}
      # A binary file's counts are -: no lines.
      [ "$added" != - ] || added=0
      [ "$removed" != - ] || removed=0
      case $added in '' | *[!0-9]*) unread "$record" ;; esac
      case $removed in '' | *[!0-9]*) unread "$record" ;; esac
      case $path in
        "$core"*)
          touches[i]=1
          core_added[i]=$((core_added[i] + added)) core_removed[i]=$((core_removed[i] + removed)) ;;
        *)
          other_added[i]=$((other_added[i] + added)) other_removed[i]=$((other_removed[i] + removed)) ;;
      esac
      ;;
  esac
done < "$diff"
[ "$rows" -eq "$n" ] || fail "git diff-tree gave $rows of the $n commits of $ref"

# from[i]: how many of commits i to the oldest touch the core, so the window
# at commit i, it and the window - 1 before it, holds from[i] - from[i + window].
from[rows]=0
for ((i = rows - 1; i >= 0; i--)); do from[i]=$((from[i + 1] + touches[i])); done
# Sets percent to $1 commits as a percentage of the window, rounded to a
# tenth, a half up.
percent_of() {
  local tenths=$(((2000 * $1 + window) / (2 * window)))
  percent="$((tenths / 10)).$((tenths % 10))"
}

if [ -n "$csv" ]; then
  {
    echo "sha,date,subject,touches_core,core_added,core_removed,other_added,other_removed,rolling_${window}_core,rolling_${window}_percent"
    for ((i = 0; i < rows; i++)); do
      share=,
      if [ $((i + window)) -le "$rows" ]; then
        count=$((from[i] - from[i + window]))
        percent_of "$count"
        share="$count,$percent"
      fi
      # Every subject is quoted, its quotes doubled (RFC 4180).
      subject=${subjects[i]//\"/\"\"}
      printf '%s,%s,"%s",%s,%s,%s,%s,%s,%s\n' "${shas[i]}" "${dates[i]}" "$subject" "${touches[i]}" \
        "${core_added[i]}" "${core_removed[i]}" "${other_added[i]}" "${other_removed[i]}" "$share"
    done
  } > "$csv" || fail "cannot write $csv"
fi

if [ "$rows" -ge "$window" ]; then
  count=$((from[0] - from[window]))
  allowed=$((target * window / 100))
  percent_of "$count"
  report="core share at $short ($ref): $count of the latest $window first-parent commits touch $core, $percent%; target at most $target%"
  if [ "$count" -le "$allowed" ]; then
    report+=$'\n'"the target is met"
  else
    report+=$'\n'"$((count - allowed)) over the target, which allows $allowed of $window"
  fi
else
  report="core share at $short ($ref): $rows first-parent commits, fewer than the window of $window, so no share; target at most $target%"
fi
lines="+${core_added[0]} -${core_removed[0]} there, +${other_added[0]} -${other_removed[0]} elsewhere"
if [ "${touches[0]}" = 1 ]; then
  report+=$'\n'"$short itself touches $core: $lines"
else
  report+=$'\n'"$short itself does not touch $core: $lines"
fi
echo "$report"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '## Core share\n\n```text\n%s\n```\n' "$report" >> "$GITHUB_STEP_SUMMARY" ||
    fail "cannot write the job summary"
fi
exit 0
