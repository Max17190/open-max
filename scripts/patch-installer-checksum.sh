#!/bin/sh
# Make dist's shell installer verify the downloaded archive with `shasum -a 256`
# when there is no `sha256sum`.
#
# Usage: scripts/patch-installer-checksum.sh <installer.sh>
#
# The installer dist generates checks the archive's SHA-256 only with
# `sha256sum` and skips the check, printing a message, without it. macOS 14 and
# earlier ship `shasum` but no `sha256sum`, so on those machines the installer
# would install whatever archive it downloaded. The release workflow runs this
# on the installer before it is uploaded: `sha256sum` is still used where it
# exists, `shasum -a 256` where it does not, and the check is skipped, with a
# message, only on a machine with neither.
#
# The installer comes from a template inside the dist binary, so this replaces
# the exact text of its sha256 check and fails, leaving the file as it was, if
# that text is not there exactly once. A dist bump that changes the check fails
# the release rather than publishing an installer that skips it; re-read the
# new check and update both blocks below in the commit that bumps
# cargo-dist-version.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 <installer.sh>" >&2; exit 2; }
installer=$1
[ -f "$installer" ] || { echo "error: $installer is not a file" >&2; exit 1; }

# Each block is printed by a function rather than written as a heredoc inside
# $(...): the shell macOS runs as sh (bash 3.2) ends the substitution at the
# first unbalanced ")" in such a heredoc.
old_check() {
  cat <<'SH'
        sha256)
            if ! check_cmd sha256sum; then
                say "skipping sha256 checksum verification (it requires the 'sha256sum' command)"
                return 0
            fi
            _calculated_checksum="$(sha256sum -b "$_file" | awk '{printf $1}')"
            ;;
SH
}
new_check() {
  cat <<'SH'
        sha256)
            # Patched by scripts/patch-installer-checksum.sh in the release
            # workflow: macOS 14 and earlier have `shasum` but no `sha256sum`.
            if check_cmd sha256sum; then
                _calculated_checksum="$(sha256sum -b "$_file" | awk '{printf $1}')"
            elif check_cmd shasum; then
                _calculated_checksum="$(shasum -a 256 -b "$_file" | awk '{printf $1}')"
            else
                say "skipping sha256 checksum verification (it requires the 'sha256sum' or 'shasum' command)"
                return 0
            fi
            ;;
SH
}
# awk finds the block as literal text (index, not a pattern) in one pass, and
# reads both blocks from the environment, where -v would expand backslashes.
tmp="$installer.patch-tmp"
code=0
OLD=$(old_check) NEW=$(new_check) awk '
  { text = text $0 "\n" }
  END {
    old = ENVIRON["OLD"]
    at = index(text, old)
    if (at == 0) exit 3
    rest = substr(text, at + length(old))
    if (index(rest, old) != 0) exit 4
    printf "%s%s%s", substr(text, 1, at - 1), ENVIRON["NEW"], rest
  }
' "$installer" > "$tmp" || code=$?
case $code in
  0) ;;
  3) echo "error: $installer has no sha256 check in the shape this script patches; dist changed it, so re-read the new one and update $0" >&2 ;;
  4) echo "error: $installer has more than one sha256 check in the shape this script patches" >&2 ;;
  *) echo "error: awk failed ($code) on $installer" >&2 ;;
esac
# Copied over rather than moved, so the installer keeps its own mode.
[ "$code" -eq 0 ] && cat "$tmp" > "$installer"
rm -f "$tmp"
exit "$code"
