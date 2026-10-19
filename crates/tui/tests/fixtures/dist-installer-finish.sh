# The end of install() from the shell installer cargo-dist 0.32.0 generated for
# v2026.10.1 (openmax-installer.sh on that release), with the helpers it
# calls, extracted verbatim with
#
#   sed -n '/^say() {/,/^}/p; /^warn() {/,/^}/p; /^check_for_shadowed_bins() {/,/^}/p' openmax-installer.sh
#   sed -n '/^    _shadowed_bins=/,/^}/p' openmax-installer.sh
#
# with only the `finish_install() {` line before the second part added, so a
# test can run it. The installer is MIT licensed by its header.
# crates/tui/tests/release.rs patches this copy with
# scripts/patch-installer-first-command.sh and runs it, and fails while
# cargo-dist-version differs from the version named on the first line, so
# when it changes, re-extract this from an installer the new version generates
# and name that version there.

check_for_shadowed_bins() {
    local _install_dir="$1"
    local _bins="$2"
    local _shadow

    for _bin_name in $_bins; do
        _shadow="$(command -v "$_bin_name")"
        if [ -n "$_shadow" ] && [ "$_shadow" != "$_install_dir/$_bin_name" ]; then
            _shadowed_bins="$_shadowed_bins $_bin_name"
        fi
    done

    echo "$_shadowed_bins"
}
say() {
    if [ "0" = "$PRINT_QUIET" ]; then
        echo "$1"
    fi
}
warn() {
    if [ "0" = "$PRINT_QUIET" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        say "${red}WARN${reset}: $1" >&2
    fi
}
finish_install() {
    _shadowed_bins="$(check_for_shadowed_bins "$_install_dir" "$_bins")"
    if [ -n "$_shadowed_bins" ]; then
        warn "The following commands are shadowed by other commands in your PATH:$_shadowed_bins"
    fi
}
