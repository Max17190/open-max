#!/usr/bin/env bash
# Sweep a single f64 constant in recall.rs, rebuild, and evaluate on the DEV
# corpus only. Tuning must never see the held-out domain.
#
#   CONST_NAME=BM25_K1 ./sweep.sh 1.2 0.9 0.6 0.3
#
# Restores the source on exit, including on failure or interrupt.
set -euo pipefail

REPO=${REPO:-$(git rev-parse --show-toplevel)}
SRC=$REPO/crates/core/src/recall.rs
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
BIN=$TARGET/release/openmax
CONST_NAME=${CONST_NAME:-EPISODE_COV_P}
CORPUS=${CORPUS:-dev}
RIG=$(cd "$(dirname "$0")" && pwd)

[ -f "$SRC" ] || { echo "no such source: $SRC" >&2; exit 1; }

# A unique backup, so two sweeps cannot restore each other's source and a
# stale file from a crashed run cannot overwrite live edits.
BACKUP=$(mktemp "${TMPDIR:-/tmp}/sweep-recall-XXXXXX")
cp "$SRC" "$BACKUP"
cleanup() {
  cp "$BACKUP" "$SRC"
  rm -f "$BACKUP"
}
trap cleanup EXIT INT TERM

for v in "$@"; do
  python3 - "$SRC" "$CONST_NAME" "$v" <<'PY'
import re, sys
path, name, val = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
pat = rf'(const {name}: f64 = )[0-9.]+;'
assert re.search(pat, s), f"constant {name} not found in {path}"
open(path, 'w').write(re.sub(pat, rf'\g<1>{val};', s))
PY
  # A failed build has to stop the sweep. Continuing would measure whichever
  # binary happened to be on disk and print it under this value, which is the
  # one way a measurement tool can actively mislead.
  if ! (cd "$REPO" && cargo build --release --quiet); then
    echo "build failed at $CONST_NAME=$v; stopping rather than measuring a stale binary" >&2
    exit 1
  fi
  [ -x "$BIN" ] || { echo "no binary at $BIN" >&2; exit 1; }
  printf '%s=%s  ' "$CONST_NAME" "$v"
  (cd "$RIG" && python3 eval2.py "$BIN" --corpus "$CORPUS" | sed -n '2p')
done
