#!/bin/bash
# Fail when the workspace's test suite takes longer than its time budget.
#
# Usage: scripts/check-suite-time.sh [multiplier]
#
# The target is the whole suite in 5 seconds of wall time after compilation,
# on the 10-core machine it is measured on. Cargo runs test binaries one
# after another, so `cargo test` takes the sum of their times. Most of that
# time is tests waiting, not computing: on a child process, a socket, or a
# deadline the binary keeps in production (the 3-second stop deadline, the
# 2-second keyboard probe, the 1-second retry backoff). This script runs
# every test binary at once, each from its package directory with its own
# libtest thread pool, so the suite takes about as long as its slowest
# binary, and fails when that is over budget or any test fails (a binary
# that fails beside the others is run again alone, below). The pools
# are wider than the machine for the same reason: a pool of one thread per
# core, libtest's default, leaves the tests that wait queued behind each
# other on a 4-core runner while the cores sit idle.
#
# The budget is 5 seconds times the multiplier, set for each CI runner from
# its runs. CI passes a multiplier of 3 on ubuntu-latest (4 cores) and
# a multiplier of 4 on macos-latest (3 cores, M1), where the target is
# measured on 10 cores. The suite's processes spend about 22 CPU-seconds,
# so on those runners the CPU work alone takes 5.5 and 7.3 seconds beside
# the tests that wait. Their first two runs took 6.9 and 8.5 seconds on
# ubuntu-latest, and 13.1 and 10.4 seconds on macos-latest, where the TUI's
# unit tests took 13.1 seconds beside the other binaries against 4.7
# seconds run alone. A shared runner's time moved by a fifth to a quarter
# between those two runs, so each budget is at least half again its
# runner's slowest run.
#
# Run it after `cargo test --workspace --locked` (or with `--no-run`): it
# asks cargo for the test executables before the clock starts, which builds
# nothing when they are fresh. Doc tests are not binaries and are not run.
# Each executable cargo built is started once before the clock starts, too:
# macOS checks a new binary the first time it runs, and every test that
# starts the same new binary meanwhile waits on that check.
set -u -o pipefail

multiplier=${1:-1}
if ! printf '%s' "$multiplier" | grep -Eq '^[0-9]+([.][0-9]+)?$'; then
  echo "usage: $0 [multiplier]" >&2
  exit 2
fi
budget_ms=$(awk -v m="$multiplier" 'BEGIN { printf "%d", 5000 * m }')
# Two of perl's core modules: JSON::PP reads cargo's list of what it built,
# Time::HiRes reads the clock. A minimal Linux image can have a perl without
# them: the perl-base package of Debian and Ubuntu has neither, perl has both.
perl -MJSON::PP -MTime::HiRes -e 1 2> /dev/null || {
  echo "error: $0 needs perl with its core modules JSON::PP and Time::HiRes" >&2
  exit 2
}
# Tests per binary in flight at once (see above).
threads=32

cd "$(dirname "$0")/.."
now_ms() { perl -MTime::HiRes=time -e 'printf "%d\n", time * 1000'; }
secs() { printf '%d.%02ds' "$(($1 / 1000))" "$(($1 % 1000 / 10))"; }

# One line per executable: whether it is a test binary, its package
# directory, its name, its path.
built=$(cargo test --workspace --locked --no-run --message-format=json |
  perl -CO -MJSON::PP -ne '
    my $m = decode_json($_);
    next unless $m->{reason} eq "compiler-artifact" && defined $m->{executable};
    (my $dir = $m->{manifest_path}) =~ s{/Cargo\.toml$}{};
    print join("\t", $m->{profile}{test} ? "true" : "false", $dir,
      "$m->{target}{name} ($m->{target}{kind}[0])", $m->{executable}), "\n";
  ') || {
  echo "error: cargo could not build the test binaries" >&2
  exit 1
}
tab=$(printf '\t')
binaries=$(printf '%s\n' "$built" | sed -n "s/^true$tab//p")
[ -n "$binaries" ] || { echo "error: cargo listed no test binaries" >&2; exit 1; }
# A test binary lists its tests; any other binary prints its version.
printf '%s\n' "$built" | {
  while IFS="$tab" read -r test _ _ exe; do
    case "$test" in true) "$exe" --list ;; *) "$exe" --version ;; esac > /dev/null 2>&1 &
  done
  wait
}

# A shell starts a background job with SIGINT and SIGQUIT ignored, and the
# binary under test keeps a signal it inherited as ignored: put the defaults
# back, as `cargo test` runs it, or every Ctrl+C test fails.
run_test() {
  perl -e '$SIG{INT} = $SIG{QUIT} = "DEFAULT"; exec @ARGV or die "cannot run $ARGV[0]: $!\n"' \
    "$1" --test-threads "$threads"
}

logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT
start=$(now_ms)
count=0
while IFS="$tab" read -r dir name exe; do
  count=$((count + 1))
  (
    cd "$dir" || exit
    began=$(now_ms)
    run_test "$exe" > "$logs/$count.log" 2>&1
    status=$?
    printf '%s\t%s\t%s\t%s\t%s\n' "$(($(now_ms) - began))" "$status" "$name" "$dir" "$exe" > "$logs/$count.status"
  ) &
done <<EOF
$binaries
EOF
wait
elapsed_ms=$(($(now_ms) - start))

reported=$(find "$logs" -name '*.status' | wc -l | tr -d ' ')
[ "$reported" -eq "$count" ] || { echo "error: $((count - reported)) test binaries did not report" >&2; exit 1; }
cat "$logs"/*.status | sort -rn | while IFS="$tab" read -r ms status name _; do
  printf '%9s  %s%s\n' "$(secs "$ms")" "$name" "$([ "$status" = 0 ] || echo " FAILED (exit $status)")"
done
# libtest's account of its failures, or the end of whatever it printed.
failures() { if grep -q '^failures:$' "$1"; then sed -n '/^failures:$/,$p' "$1"; else tail -n 40 "$1"; fi; }
failed=0
for report in "$logs"/*.status; do
  IFS="$tab" read -r _ status name dir exe < "$report"
  [ "$status" = 0 ] && continue
  log=${report%.status}.log
  echo
  # A test with a deadline can miss it beside every other binary on a small
  # runner, and pass alone as `cargo test` runs it: such a binary is run
  # again alone, after the clock, and fails the gate only if it fails again.
  if (cd "$dir" && run_test "$exe") > "$log.alone" 2>&1; then
    echo "${GITHUB_ACTIONS:+::warning::}$name failed beside the other binaries and passed alone:"
    failures "$log"
    continue
  fi
  failed=1
  echo "${GITHUB_ACTIONS:+::error::}$name failed:"
  failures "$log.alone"
done
[ "$failed" = 0 ] || exit 1

if [ "$elapsed_ms" -gt "$budget_ms" ]; then
  echo "${GITHUB_ACTIONS:+::error::}the test suite took $(secs "$elapsed_ms")," \
    "$(secs $((elapsed_ms - budget_ms))) over its $(secs "$budget_ms") budget"
  exit 1
fi
echo "the test suite took $(secs "$elapsed_ms") of its $(secs "$budget_ms") budget"
