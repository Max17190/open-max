#!/usr/bin/env bash
# Cut a release without choosing a version number.
#
# Cargo, Homebrew and the git tag all require a version, and cargo-dist refuses
# any tag that does not equal the workspace version (`--tag=v2026.9.0` against a
# 0.2.0 manifest is rejected outright). A version therefore has to exist. What
# does not have to exist is a human deciding what it should be.
#
# So the date decides: YYYY.M.N, where N counts releases already cut this month.
# The month is unpadded because semver rejects a leading zero ("2026.09.0" is a
# hard cargo error), and N restarts each month so two releases on one day are
# just .0 and .1.
#
# Usage:
#   scripts/release.sh              cut and push a release
#   scripts/release.sh --dry-run    print what it would do, touch nothing
set -euo pipefail

cd "$(dirname "$0")/.."

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

# Pushing a tag publishes binaries under this project's name, so refuse to do it
# from a tree that does not match what has been reviewed.
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "main" ] || { echo "error: on '$branch'; releases are cut from main" >&2; exit 1; }
# Only tracked changes matter: untracked files never reach the commit a tag builds.
git diff --quiet && git diff --cached --quiet || {
  echo "error: uncommitted changes to tracked files" >&2; exit 1; }

git fetch --quiet --tags origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || {
  echo "error: local main differs from origin/main; pull or push first" >&2; exit 1; }

# %-m drops the zero padding that semver forbids.
month=$(date -u +%Y.%-m)
count=$(git tag -l "v${month}.*" | wc -l | tr -d ' ')
version="${month}.${count}"
tag="v${version}"

git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1 && {
  echo "error: tag ${tag} already exists" >&2; exit 1; }

echo "  version: ${version}   (was $(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2))"
echo "  tag:     ${tag}"
echo "  commit:  $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

if $DRY_RUN; then
  echo
  echo "dry run: nothing changed."
  exit 0
fi

echo
printf 'This publishes a release to GitHub, the Homebrew tap and the install URL. Continue? [y/N] '
read -r reply
[ "$reply" = "y" ] || [ "$reply" = "Y" ] || { echo "aborted."; exit 1; }

# One version lives in [workspace.package]; both crates inherit it.
perl -0pi -e "s/^version = \"[^\"]*\"/version = \"${version}\"/m" Cargo.toml
cargo update -w --quiet

git add Cargo.toml Cargo.lock
git commit -qm "release: ${version}"
git tag -a "${tag}" -m "${version}"
# --atomic or neither ref moves. A plain multi-ref push is not atomic: if the
# branch is rejected (someone pushed to main in the window since the check
# above) the tag still lands, and a tag is what starts a release, so the
# published binaries would come from a commit that never reached main.
if ! git push --quiet --atomic origin main "${tag}"; then
  # Undo the local commit and tag so a retry starts clean and does not skip a
  # number. Safe because both were created moments ago by this script, on a
  # tree it verified was clean.
  if [ "$(git log -1 --format=%s)" = "release: ${version}" ]; then
    git tag -d "${tag}" >/dev/null
    git reset --quiet --hard HEAD~1
  fi
  echo "push rejected; nothing was published and the local bump was undone." >&2
  echo "run 'git pull --rebase' and try again." >&2
  exit 1
fi

echo "pushed ${tag}. The release workflow builds and publishes it:"
echo "  https://github.com/Max17190/open-max/actions"
