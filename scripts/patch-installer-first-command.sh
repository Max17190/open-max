#!/bin/sh
# Make dist's shell installer end by printing the first command to run, with
# the binary's absolute path.
#
# Usage: scripts/patch-installer-first-command.sh <installer.sh>
#
# The installer dist generates ends with "everything's installed!" and, when it
# adds ~/.openmax/bin to PATH through the shell profiles, a note to restart the
# shell or source an env script: the PATH change reaches only a new shell, so a
# bare `openmax` typed next fails in the shell that ran the install. The
# release workflow runs this on the installer before it is uploaded, so its
# last line is the whole first run, one command that grants trust, writes the
# endpoint to settings.json and answers a first task, naming the binary by the
# absolute path it was installed to. The capitals are placeholders the flag
# parser refuses, '$MY_API_KEY' too while that variable is unset, so the line
# pasted with any of them left unchanged writes nothing.
#
# The installer comes from a template inside the dist binary, so this replaces
# the exact text of the end of its install function and fails, leaving the file
# as it was, if that text is not there exactly once. A dist bump that changes it
# fails the release rather than publishing an installer without the line;
# re-read the new end of install() and update both blocks below in the commit
# that bumps cargo-dist-version.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 <installer.sh>" >&2; exit 2; }
installer=$1
[ -f "$installer" ] || { echo "error: $installer is not a file" >&2; exit 1; }

# Each block is printed by a function rather than written as a heredoc inside
# $(...): the shell macOS runs as sh (bash 3.2) ends the substitution at the
# first unbalanced ")" in such a heredoc.
old_end() {
  cat <<'SH'
    _shadowed_bins="$(check_for_shadowed_bins "$_install_dir" "$_bins")"
    if [ -n "$_shadowed_bins" ]; then
        warn "The following commands are shadowed by other commands in your PATH:$_shadowed_bins"
    fi
}
SH
}
new_end() {
  cat <<'SH'
    _shadowed_bins="$(check_for_shadowed_bins "$_install_dir" "$_bins")"
    if [ -n "$_shadowed_bins" ]; then
        warn "The following commands are shadowed by other commands in your PATH:$_shadowed_bins"
    fi

    # Patched by scripts/patch-installer-first-command.sh in the release
    # workflow: a PATH change reaches only a new shell, so the first command
    # names the binary by its absolute path (a relative OPENMAX_INSTALL_DIR is
    # resolved against the directory the installer ran in), single-quoted when
    # it holds a character a shell would split or expand, so it pastes as one
    # word.
    _first_bin="$_install_dir/openmax"
    case "$_first_bin" in
        /*) ;;
        *) _first_bin="$(pwd -P)/$_first_bin" ;;
    esac
    case "$_first_bin" in
        *[!A-Za-z0-9_./-]*) _first_bin="'$(printf '%s' "$_first_bin" | sed "s/'/'\\\\''/g")'" ;;
    esac
    say ""
    say "run a first task in your project, with your model server's values for the capitals:"
    # printf rather than say: the echo in say turns a backslash in the path
    # into an escape in dash and macOS sh.
    if [ "0" = "$PRINT_QUIET" ]; then
        printf '    %s\n' "$_first_bin --trust-project --base-url URL --model MODEL --api-key '\$MY_API_KEY' --context-tokens TOKENS -p 'first task'"
    fi
}
SH
}
# awk finds the block as literal text (index, not a pattern) in one pass, and
# reads both blocks from the environment, where -v would expand backslashes.
tmp="$installer.patch-tmp"
code=0
OLD=$(old_end) NEW=$(new_end) awk '
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
  3) echo "error: $installer has no end of install() in the shape this script patches; dist changed it, so re-read the new one and update $0" >&2 ;;
  4) echo "error: $installer has more than one end of install() in the shape this script patches" >&2 ;;
  *) echo "error: awk failed ($code) on $installer" >&2 ;;
esac
# Copied over rather than moved, so the installer keeps its own mode.
[ "$code" -eq 0 ] && cat "$tmp" > "$installer"
rm -f "$tmp"
exit "$code"
