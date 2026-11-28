#!/usr/bin/env bash
# Say whether the built-in prompt changed between two commits, for release notes.
#
# crates/core/tests/fixtures/built-in-prompt/ records the bytes every session
# sends before anything of the project's own (the base rules of the system
# prompt and the built-in tools array), and a test holds that record to the
# code, so the record at a commit is what that commit's binary sends. This
# script reads the record at two commits from git and prints a Markdown
# section: whether each file changed, its size at both, and how many bytes
# differ (each position that holds another byte, plus the difference in
# length). It reads git objects only, in the repository of the current
# directory, so whatever is checked out there does not change what it prints.
#
# Usage:
#   scripts/prompt-record.sh [<ref>] [--since <ref>]
#
# <ref> defaults to HEAD. --since defaults to the latest v* tag before <ref>
# (an ancestor of <ref>'s parent), the previous release when <ref> is a
# release tag. The release workflow appends the output to each release's
# notes, and scripts/release.sh prints it before it asks to publish.
set -euo pipefail

# From a directory under the top, git would list only the entries under it.
top=$(git rev-parse --show-toplevel) || exit 2
cd "$top"

RECORD=crates/core/tests/fixtures/built-in-prompt
ref=HEAD
since=
while [ $# -gt 0 ]; do
  case "$1" in
    --since) [ $# -ge 2 ] || { echo "error: --since needs a ref" >&2; exit 2; }; since=$2; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; exit 2 ;;
    *) ref=$1; shift ;;
  esac
done

git rev-parse -q --verify "${ref}^{commit}" >/dev/null || { echo "error: no commit $ref" >&2; exit 2; }
if [ -z "$since" ]; then
  since=$(git describe --tags --abbrev=0 --match 'v[0-9]*' "${ref}^" 2>/dev/null) || since=
fi
[ -z "$since" ] || git rev-parse -q --verify "${since}^{commit}" >/dev/null || {
  echo "error: no commit $since" >&2; exit 2; }

label() {
  case "$1" in
    system.txt) echo "system prompt base rules" ;;
    tools.json) echo "built-in tools array" ;;
    tools-image-input.json) echo "built-in tools array with image input" ;;
    *) echo "$1" ;;
  esac
}

files=$(git ls-tree --name-only "${ref}:${RECORD}" 2>/dev/null) || {
  echo "error: $ref has no record at $RECORD" >&2; exit 1; }
[ -n "$files" ] || { echo "error: the record at $ref is empty" >&2; exit 1; }

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "### Built-in prompt"
echo
sizes=()
changes=()
for file in $files; do
  git cat-file blob "${ref}:${RECORD}/${file}" > "$tmp/now"
  now=$(wc -c < "$tmp/now" | tr -d ' ')
  sizes+=("$(label "$file") ${now} bytes")
  [ -n "$since" ] || continue
  if ! git cat-file -e "${since}:${RECORD}/${file}" 2>/dev/null; then
    changes+=("- $(label "$file"): ${now} bytes, not recorded at ${since}")
    continue
  fi
  git cat-file blob "${since}:${RECORD}/${file}" > "$tmp/then"
  cmp -s "$tmp/then" "$tmp/now" && continue
  then=$(wc -c < "$tmp/then" | tr -d ' ')
  differ=$({ cmp -l "$tmp/then" "$tmp/now" 2>/dev/null || true; } | wc -l | tr -d ' ')
  length=$(( now > then ? now - then : then - now ))
  delta=$(( now - then ))
  [ "$delta" -ge 0 ] && delta="+${delta}"
  changes+=("- $(label "$file"): ${then} -> ${now} bytes (${delta}), $(( differ + length )) bytes differ")
done
for file in $(git ls-tree --name-only "${since:-$ref}:${RECORD}" 2>/dev/null); do
  printf '%s\n' $files | grep -qx "$file" || changes+=("- $(label "$file"): removed")
done

joined=$(printf '%s, ' "${sizes[@]}")
joined=${joined%, }
if [ -z "$since" ]; then
  echo "No earlier release to compare with. Recorded here: ${joined}."
elif ! git cat-file -e "${since}:${RECORD}" 2>/dev/null; then
  echo "${since} has no record of its built-in prompt to compare with. Recorded here: ${joined}."
elif [ ${#changes[@]} -eq 0 ]; then
  echo "Unchanged since ${since}: the same bytes, ${joined}."
else
  echo "Changed since ${since}:"
  echo
  printf '%s\n' "${changes[@]}"
  echo
  echo "\`git diff ${since} ${ref} -- ${RECORD}\` shows each change."
fi
echo
echo "The system prompt's base rules write the project path as \`{root}\`; the rest of the prompt comes from the project."
