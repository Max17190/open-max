#!/bin/bash
# Fail when the workspace's test suite takes longer than its time budget.
#
# Usage: scripts/check-suite-time.sh [--runs N] [multiplier]
#
# The target is the whole suite in 5 seconds of wall time after compilation,
# on the 10-core machine it is measured on. Cargo runs test binaries one
# after another, so `cargo test` takes the sum of their times. Most of that
# time is tests waiting, not computing: on a child process, a socket, or a
# deadline the binary keeps in production (the 2-second keyboard probe, the
# MCP bridge's 2-second stop grace, the 1-second retry backoff). This script
# runs every test binary at once, each from its package directory with its
# own libtest thread pool, so the suite takes about as long as its slowest
# binary, and fails when that is over budget or any test fails (a binary
# that fails beside the others is run again alone, below). The pools
# are wider than the machine for the same reason: a pool of one thread per
# core, libtest's default, leaves the tests that wait queued behind each
# other on a 4-core runner while the cores sit idle.
#
# The budget is 5 seconds times the multiplier, set for each CI runner from
# its runs. CI passes a multiplier of 3 on ubuntu-latest (4 cores) and
# a multiplier of 4 on macos-latest (3 cores, M1), where the target is
# measured on 10 cores. The suite's processes spend about 20 CPU-seconds,
# so on those runners the CPU work alone takes 5.0 and 6.7 seconds beside
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
#
# A binary still running at three times the budget is stopped, with every
# process still under it, and fails, naming the tests it had not finished: a
# test that hangs fails its run in that time instead of holding the run, and
# CI's step, until a timeout that prints nothing, and what it started does
# not run on into the runs after it. In the run the clock times, a binary
# still running then has put the run over its budget anyway. The run again
# alone, below, is not timed but has the same bound, so a binary that needs
# longer than that even alone fails there.
#
# One run says little about the next: the slowest binary's time moves with
# whatever else the machine is doing. With `--runs N` the suite runs N times,
# one run after another, after the one build and the one start of each
# executable. Each run is held to the budget as a single run is, and prints
# one line: its time, its slowest binary, the machine's 1-minute load average
# as it began, and how many binaries passed only when run again alone, if
# any; a run that failed or ran a binary again alone shows its report first.
# Then come the median and the slowest run, and the script fails when any
# run failed.
set -u -o pipefail

runs=1
if [ "${1:-}" = --runs ]; then
  runs=${2:-}
  shift $(($# < 2 ? $# : 2))
fi
multiplier=${1:-1}
if [ $# -gt 1 ] || ! printf '%s' "$multiplier" | grep -Eq '^[0-9]+([.][0-9]+)?$' ||
  ! printf '%s' "$runs" | grep -Eq '^[1-9][0-9]*$'; then
  echo "usage: $0 [--runs N] [multiplier]" >&2
  exit 2
fi
budget_ms=$(awk -v m="$multiplier" 'BEGIN { printf "%d", 5000 * m }')
bound_ms=$((3 * budget_ms))
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
#
# run_test runs executable $1, its output in the file $2, and sets `ended_as`
# to its exit status (128 plus the signal for a binary a signal ended), or to
# `stopped` for a binary still running at the bound. Its launcher stays the
# binary's parent, and at the bound stops the binary and every process still
# under it, in its session or another, and leaves a mark, since any exit
# status could be the binary's own (142 is that of a binary SIGALRM ended):
# it stops each process as it finds it, by its parent in /proc or else ps, so
# none starts another past the sweep, then kills them.
#
# The test pool is $3: unset, this script's wide one; empty, libtest's own,
# one thread per core, which is the pool `cargo test` gives it.
run_test() {
  local pools=${3-$threads} pool=()
  [ -z "$pools" ] || pool=(--test-threads "$pools")
  perl -MTime::HiRes=alarm -e '
    sub parents {
      opendir my $proc, "/proc" or return map { /(\d+)\s+(\d+)/ ? [$1, $2] : () } `ps -A -o pid= -o ppid=`;
      map { my $stat; open($stat, "<", "/proc/$_/stat") && <$stat> =~ /.*\)\s+\S+\s+(\d+)/s ? [$_, $1] : () }
        grep /^\d+$/, readdir $proc;
    }
    my ($bound, $mark) = (shift() / 1000, shift);
    my $pid = fork // die "cannot fork: $!\n";
    if (!$pid) { $SIG{INT} = $SIG{QUIT} = "DEFAULT"; exec @ARGV or die "cannot run $ARGV[0]: $!\n" }
    $SIG{ALRM} = sub {
      my %under = ($pid => kill STOP => $pid);
      for (1 .. 100) {
        my @found = grep { !exists $under{$_->[0]} && exists $under{$_->[1]} } parents() or last;
        $under{$_->[0]} = kill STOP => $_->[0] for @found;
      }
      kill KILL => keys %under;
      open my $made, ">", $mark or die "cannot create $mark: $!\n";
    };
    alarm $bound if $bound > 0;
    waitpid $pid, 0;
    exit($? & 127 ? 128 + ($? & 127) : $? >> 8);
  ' "$bound_ms" "$2.stopped" "$1" ${pool[@]+"${pool[@]}"} > "$2" 2>&1
  ended_as=$?
  [ ! -e "$2.stopped" ] || ended_as=stopped
}
# How a binary ended, after its name, from the `ended_as` ($1) of its run.
ended() { case $1 in 0) ;; stopped) echo " FAILED (stopped at $(secs "$bound_ms"))" ;; *) echo " FAILED (exit $1)" ;; esac; }
# libtest's account of the failures in log $1 of executable $3, whose run
# ended as $2, or the end of whatever it printed; for a binary stopped at
# the bound, the tests it lists that the log does not report finished.
failures() {
  if [ "$2" = stopped ]; then
    echo "still running at $(secs "$bound_ms"), when it was stopped:"
    "$3" --list 2> /dev/null | sed -n 's/: test$//p' | LC_ALL=C sort > "$1.listed"
    sed -n 's/^test \(.*\) \.\.\. .*$/\1/p' "$1" | LC_ALL=C sort | LC_ALL=C comm -23 "$1.listed" -
  elif grep -q '^failures:$' "$1"; then
    sed -n '/^failures:$/,$p' "$1"
  else
    tail -n 40 "$1"
  fi
}

logs=$(mktemp -d)
trap 'rm -rf "$logs"' EXIT

# Run number $1 of the suite: every test binary at once, then each one's
# time and the failures of any that failed. Sets `elapsed_ms`, `slowest`
# (the slowest binary's time and name, or `unknown`) and `retried` (how
# many binaries passed only when run again alone), and returns 1 when a
# binary did not report, a test failed or the run took longer than its
# budget.
run_suite() {
  local out="$logs/$1" count=0 start failed=0
  mkdir "$out"
  retried=0 slowest=unknown
  start=$(now_ms)
  while IFS="$tab" read -r dir name exe; do
    count=$((count + 1))
    (
      cd "$dir" || exit
      began=$(now_ms)
      run_test "$exe" "$out/$count.log"
      printf '%s\t%s\t%s\t%s\t%s\n' "$(($(now_ms) - began))" "$ended_as" "$name" "$dir" "$exe" > "$out/$count.status"
    ) &
  done <<EOF
$binaries
EOF
  wait
  elapsed_ms=$(($(now_ms) - start))

  reported=$(find "$out" -name '*.status' | wc -l | tr -d ' ')
  [ "$reported" -eq "$count" ] || { echo "error: $((count - reported)) of $count test binaries did not report" >&2; return 1; }
  slowest=$(cat "$out"/*.status | sort -rn | head -n 1 | { IFS="$tab" read -r ms _ name _; echo "$(secs "$ms") $name"; })
  cat "$out"/*.status | sort -rn | while IFS="$tab" read -r ms status name _; do
    printf '%9s  %s%s\n' "$(secs "$ms")" "$name" "$(ended "$status")"
  done
  for report in "$out"/*.status; do
    IFS="$tab" read -r _ status name dir exe < "$report"
    [ "$status" = 0 ] && continue
    log=${report%.status}.log
    echo
    # A test with a deadline can miss it beside every other binary on a small
    # runner, and pass alone as `cargo test` runs it: such a binary is run
    # again alone, after the clock, and fails the gate only if it fails again.
    # It runs with libtest's own pool, not this script's wide one, or the
    # retry would starve the same waits the wide pool starved beside the
    # other binaries, and the comparison would prove nothing.
    alone=$(cd "$dir" && run_test "$exe" "$log.alone" "" && echo "$ended_as")
    if [ "$alone" = 0 ]; then
      echo "${GITHUB_ACTIONS:+::warning::}$name failed beside the other binaries and passed alone:"
      failures "$log" "$status" "$exe"
      retried=$((retried + 1))
      continue
    fi
    failed=1
    echo "${GITHUB_ACTIONS:+::error::}$name failed:"
    failures "$log.alone" "$alone" "$exe"
  done
  [ "$failed" = 0 ] || return 1

  if [ "$elapsed_ms" -gt "$budget_ms" ]; then
    echo "${GITHUB_ACTIONS:+::error::}the test suite took $(secs "$elapsed_ms")," \
      "$(secs $((elapsed_ms - budget_ms))) over its $(secs "$budget_ms") budget"
    return 1
  fi
  echo "the test suite took $(secs "$elapsed_ms") of its $(secs "$budget_ms") budget"
}

if [ "$runs" = 1 ]; then
  run_suite 1
  exit
fi

# `--runs N` (see above).
# The 1-minute load average, from /proc/loadavg on Linux and the vm.loadavg
# sysctl elsewhere (in /usr/sbin, which a PATH can leave out): `uptime` is
# a package of its own (procps) on a minimal Debian or Ubuntu image.
load_average() {
  local one=
  if [ -r /proc/loadavg ]; then
    read -r one _ < /proc/loadavg
  else
    one=$(PATH=$PATH:/usr/sbin:/sbin sysctl -n vm.loadavg 2> /dev/null | tr -d '{' | awk '{ print $1 }')
  fi
  echo "${one:-unknown}"
}
times="" failed_runs=0
for run in $(seq "$runs"); do
  load=$(load_average)
  passed=1
  run_suite "$run" > "$logs/$run.report" 2>&1 || passed=0
  if [ "$passed" = 0 ] || [ "$retried" != 0 ]; then
    cat "$logs/$run.report"
    echo
  fi
  [ "$passed" = 1 ] || failed_runs=$((failed_runs + 1))
  echo "run $run of $runs: $(secs "$elapsed_ms"), slowest binary $slowest, load $load$(
    [ "$retried" = 0 ] || echo ", $retried passed only when run again alone")$([ "$passed" = 1 ] || echo ", failed")"
  times="$times $elapsed_ms"
done
sorted=$(printf '%s\n' $times | sort -n)
median=$(printf '%s\n' "$sorted" | awk '{ t[NR] = $1 } END { print NR % 2 ? t[(NR + 1) / 2] : int((t[NR / 2] + t[NR / 2 + 1]) / 2) }')
echo "median $(secs "$median"), slowest $(secs "$(printf '%s\n' "$sorted" | tail -n 1)") over $runs runs," \
  "against the $(secs "$budget_ms") budget of each"
[ "$failed_runs" = 0 ]
