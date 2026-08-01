#!/usr/bin/env bash
# Sweep a single constant in recall.rs, rebuild, and evaluate on the DEV
# corpus only. Tuning must never see the held-out domain.
set -euo pipefail
REPO=${REPO:-$(git rev-parse --show-toplevel)}
SRC=$REPO/crates/core/src/recall.rs
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
CONST_NAME=${CONST_NAME:-EPISODE_COV_P}
CORPUS=${CORPUS:-dev}
cp "$SRC" /tmp/sweep_orig.rs
trap 'cp /tmp/sweep_orig.rs "$SRC"' EXIT

for v in "$@"; do
  python3 - "$SRC" "$CONST_NAME" "$v" <<'PY'
import re, sys
path, name, val = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path).read()
pat = rf'(const {name}: f64 = )[0-9.]+;'
assert re.search(pat, s), f"constant {name} not found"
open(path, 'w').write(re.sub(pat, rf'\g<1>{val};', s))
PY
  (cd "$REPO" && cargo build --release 2>&1 | grep -E '^error' && exit 1 || true)
  printf '%s=%s  ' "$CONST_NAME" "$v"
  (cd "$(dirname "$0")" && python3 eval2.py "$TARGET/release/openmax" --corpus "$CORPUS" 2>/dev/null | sed -n '2p')
done
