#!/bin/bash
# Fail when the workspace has fewer tests than its floor.
#
# Usage: scripts/check-test-count.sh
#
# A test is a name that `cargo test --workspace -- --list` lists: one line
# ending in ": test" for each test of every test binary and of the doc tests.
# The names `cargo test --workspace -- --list --ignored` lists are taken out:
# no CI job runs an ignored test, so it checks nothing.
#
# Linux and macOS each compile a few tests the other does not (one reads
# Linux's boot id, another replaces a file the way only macOS can), so each
# counts its own tests against a floor of its own: linux_floor or
# macos_floor below, picked by `uname -s`. CI's test job runs this on both,
# once it has built the suite. Nothing sets a floor for any other platform:
# there the script reports the count and passes.
#
# Each floor is a ratchet: its platform's count when it was last raised. A
# change that adds tests raises each floor, in the same commit, to the count
# this script reports on that platform for that commit (a run here gives this
# machine's; the Check test count step of CI's test job gives each runner's,
# and names the floor it can raise), so a later change cannot delete them
# unnoticed. A change that removes tests on purpose (a retired module) lowers
# the floors and says there why. The target is 2,000 tests: under it the
# script reports how far there is to go and passes.
set -u -o pipefail

linux_floor=1427
macos_floor=1425
target=2000

[ $# -eq 0 ] || { echo "usage: $0" >&2; exit 2; }
cd "$(dirname "$0")/.."

case $(uname -s) in
  Linux) platform=Linux floor=$linux_floor name=linux_floor ;;
  Darwin) platform=macOS floor=$macos_floor name=macos_floor ;;
  *) platform=$(uname -s) floor= name= ;;
esac

errors=$(mktemp)
trap 'rm -f "$errors"' EXIT
# What one `cargo test --workspace -- --list ...` prints, cargo's own output
# shown only when it fails.
list() {
  cargo test --workspace --locked -- --list "$@" 2> "$errors" || {
    cat "$errors" >&2
    echo "${GITHUB_ACTIONS:+::error::}\`cargo test --workspace --locked -- --list${*:+ $*}\` failed" >&2
    return 1
  }
}
listed=$(list) || exit 1
ignored=$(list --ignored) || exit 1
count() { printf '%s\n' "$1" | grep -c ': test$'; }
all=$(count "$listed")
skipped=$(count "$ignored")
tests=$((all - skipped))

if [ -z "$floor" ]; then
  echo "$tests tests on $platform ($all listed, $skipped of them ignored); no floor, target $target"
  echo "scripts/check-test-count.sh has a floor for Linux and for macOS, where CI's test job counts, and none for $platform"
else
  echo "$tests tests on $platform ($all listed, $skipped of them ignored); floor $floor, target $target"
  if [ "$tests" -lt "$floor" ]; then
    echo "${GITHUB_ACTIONS:+::error::}the workspace has $tests tests on $platform, $((floor - tests)) under its floor of $floor"
    exit 1
  fi
fi
if [ "$tests" -ge "$target" ]; then
  echo "the target of $target tests is met"
else
  echo "$((target - tests)) tests under the target of $target"
fi
if [ -n "$floor" ] && [ "$tests" -gt "$floor" ]; then
  echo "$name in scripts/check-test-count.sh can be raised to $tests"
fi
exit 0
