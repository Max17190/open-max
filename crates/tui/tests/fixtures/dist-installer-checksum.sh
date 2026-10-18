# The archive check from the shell installer cargo-dist 0.32.0 generated for
# v2026.10.1 (openmax-installer.sh on that release), with the helpers it
# calls, extracted verbatim with
#
#   sed -n '/^say() {/,/^}/p; /^err() {/,/^}/p; /^check_cmd() {/,/^}/p; /^verify_checksum() {/,/^}/p' openmax-installer.sh
#
# The installer is MIT licensed by its header. crates/tui/tests/release.rs
# patches this copy with scripts/patch-installer-checksum.sh and runs it, and
# fails while cargo-dist-version differs from the version named on the first
# line, so when it changes, re-extract this from an installer the new version
# generates and name that version there.

say() {
    if [ "0" = "$PRINT_QUIET" ]; then
        echo "$1"
    fi
}
err() {
    if [ "0" = "$PRINT_QUIET" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        say "${red}ERROR${reset}: $1" >&2
    fi
    exit 1
}
check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}
verify_checksum() {
    local _file="$1"
    local _checksum_style="$2"
    local _checksum_value="$3"
    local _calculated_checksum

    if [ -z "$_checksum_value" ]; then
        return 0
    fi
    case "$_checksum_style" in
        sha256)
            if ! check_cmd sha256sum; then
                say "skipping sha256 checksum verification (it requires the 'sha256sum' command)"
                return 0
            fi
            _calculated_checksum="$(sha256sum -b "$_file" | awk '{printf $1}')"
            ;;
        sha512)
            if ! check_cmd sha512sum; then
                say "skipping sha512 checksum verification (it requires the 'sha512sum' command)"
                return 0
            fi
            _calculated_checksum="$(sha512sum -b "$_file" | awk '{printf $1}')"
            ;;
        sha3-256)
            if ! check_cmd openssl; then
                say "skipping sha3-256 checksum verification (it requires the 'openssl' command)"
                return 0
            fi
            _calculated_checksum="$(openssl dgst -sha3-256 "$_file" | awk '{printf $NF}')"
            ;;
        sha3-512)
            if ! check_cmd openssl; then
                say "skipping sha3-512 checksum verification (it requires the 'openssl' command)"
                return 0
            fi
            _calculated_checksum="$(openssl dgst -sha3-512 "$_file" | awk '{printf $NF}')"
            ;;
        blake2s)
            if ! check_cmd b2sum; then
                say "skipping blake2s checksum verification (it requires the 'b2sum' command)"
                return 0
            fi
            # Test if we have official b2sum with blake2s support
            local _well_known_blake2s_checksum="93314a61f470985a40f8da62df10ba0546dc5216e1d45847bf1dbaa42a0e97af"
            local _test_blake2s
            _test_blake2s="$(printf "can do blake2s" | b2sum -a blake2s | awk '{printf $1}')" || _test_blake2s=""

            if [ "X$_test_blake2s" = "X$_well_known_blake2s_checksum" ]; then
                _calculated_checksum="$(b2sum -a blake2s "$_file" | awk '{printf $1}')" || _calculated_checksum=""
            else
                say "skipping blake2s checksum verification (installed b2sum doesn't support blake2s)"
                return 0
            fi
            ;;
        blake2b)
            if ! check_cmd b2sum; then
                say "skipping blake2b checksum verification (it requires the 'b2sum' command)"
                return 0
            fi
            _calculated_checksum="$(b2sum "$_file" | awk '{printf $1}')"
            ;;
        false)
            ;;
        *)
            say "skipping unknown checksum style: $_checksum_style"
            return 0
            ;;
    esac

    if [ "$_calculated_checksum" != "$_checksum_value" ]; then
        err "checksum mismatch
            want: $_checksum_value
            got:  $_calculated_checksum"
    fi
}
