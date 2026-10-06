#!/bin/sh
# Fail when a release binary is larger than its target's size budget.
#
# Usage: scripts/check-binary-size.sh <target-triple> <binary>
#
# CI runs this on the release-profile build of every target a tag publishes
# (dist builds with a profile that inherits release unchanged), so a
# dependency, feature or profile change that bloats a shipped binary fails the
# pull request that makes it, on the target it bloats. A warning would not:
# nothing stops a binary drifting past a limit that only warns.
#
# Each budget is that target's measured size plus 10% headroom. The sizes are
# the v2026.10.0 release binaries. The headroom absorbs ordinary growth and
# the drift between compiler releases (neither CI nor the release pins one).
# Past it, trim the binary or raise the budget deliberately: replace the
# target's size below with the size this script reports for it, in the commit
# that needs the room, and say there why the binary grew.
set -eu

[ $# -eq 2 ] || { echo "usage: $0 <target-triple> <binary>" >&2; exit 2; }
target=$1
binary=$2

case "$target" in
  aarch64-apple-darwin)       measured=7513472 ;;
  x86_64-apple-darwin)        measured=7606280 ;;
  aarch64-unknown-linux-gnu)  measured=6748552 ;;
  x86_64-unknown-linux-gnu)   measured=7570512 ;;
  aarch64-unknown-linux-musl) measured=6530928 ;;
  x86_64-unknown-linux-musl)  measured=7694768 ;;
  *)
    echo "error: no size budget for $target; add its measured size to $0" >&2
    exit 1
    ;;
esac
budget=$((measured + measured / 10))

[ -f "$binary" ] || { echo "error: $binary is not a file" >&2; exit 1; }
size=$(wc -c < "$binary" | tr -d ' ')

if [ "$size" -gt "$budget" ]; then
  # In GitHub Actions the prefix makes the line an error annotation.
  echo "${GITHUB_ACTIONS:+::error::}openmax for $target is $size bytes," \
    "$((size - budget)) over its $budget-byte budget"
  exit 1
fi
permille=$((size * 1000 / budget))
echo "openmax for $target is $size bytes," \
  "$((permille / 10)).$((permille % 10))% of its $budget-byte budget"
