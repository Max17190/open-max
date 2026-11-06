#!/bin/bash
# Report how long the pull-request gate takes to turn green.
#
# Usage: scripts/gate-time.sh [--runs N] [--repo OWNER/NAME]
#
# The gate is the CI workflow, .github/workflows/ci.yml, on a pull request:
# every one of its jobs has to pass before the pull request merges. A run's
# gate time is the wall clock from the start of its first job to the end of
# its last, in whole seconds, as GitHub records the jobs of the run's first
# attempt: how long a push waits for its verdict once a runner picks it up.
# A job the run skipped counts at neither end.
#
# The sample is the latest N pull-request runs of the workflow (20 by
# default) whose first attempt passed: every job succeeded or was skipped. A
# run still in progress is not read. A run whose first attempt failed (a job
# failed, timed out or never ran) is not a green gate, so it is counted
# instead, with how many of those passed when run again; so is a run whose
# first attempt was cancelled with no job failed, which a newer push to its
# pull request replaced or someone stopped. A failed job outranks a
# cancelled one, since a matrix that fails fast cancels the jobs beside the
# one that failed. Both counts cover the runs as recent as the oldest in the
# sample.
#
# It prints a line for each run in the sample, newest first: its id, its
# head commit, its gate time and the job that finished last. Then the p50,
# p90 and p99 of the gate times, each the time at rank ceil(q * n / 100) of
# the n in the sample in ascending order (the nearest rank), so it is one of
# them, and the slowest. Then the counts. Then the target: a median under 90
# seconds and a p99 under 120 over N consecutive runs that passed at their
# first attempt, met only when the sample holds all N and no run as recent
# failed at its first attempt (a cancelled one gave no verdict, so it breaks
# no run of them). In GitHub Actions it adds the same lines to the job's
# summary.
#
# A report, not a gate, until the target is met: it exits 0 whatever it
# measures; 1 when it cannot measure (gh fails or prints what this script
# does not read, or no run in the latest 1,000 passed at its first attempt);
# and 2 on a usage error, or when perl lacks JSON::PP or Time::Local. Needs
# bash, gh (logged in, or with GH_TOKEN set) and perl with its core modules
# JSON::PP and Time::Local. The repository is --repo, else
# GITHUB_REPOSITORY, which GitHub Actions sets. CI's gate-time workflow runs
# it on every push to main and every night; crates/tui/tests/thesis.rs holds
# the method to runs it makes up.
set -u -o pipefail

runs=20
repo=${GITHUB_REPOSITORY:-}

usage() {
  echo "usage: $0 [--runs N] [--repo OWNER/NAME]" >&2
  exit 2
}
while [ $# -gt 0 ]; do
  case $1 in
    --runs) [ $# -ge 2 ] || usage; runs=$2; shift 2 ;;
    --repo) [ $# -ge 2 ] || usage; repo=$2; shift 2 ;;
    *) usage ;;
  esac
done
case $runs in
  '' | 0* | *[!0-9]* | ????*)
    echo "error: --runs takes a count of runs from 1 to 999, not '$runs'" >&2
    exit 2 ;;
esac
[[ $repo =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || {
  echo "error: name the repository as OWNER/NAME with --repo (or GITHUB_REPOSITORY), not '$repo'" >&2
  exit 2
}
perl -MJSON::PP -MTime::Local -e 1 2> /dev/null || {
  echo "error: $0 needs perl with its core modules JSON::PP and Time::Local" >&2
  exit 2
}

report=$(perl -MJSON::PP -MTime::Local=timegm -e '
use strict;
use warnings;

my ($repo, $want) = @ARGV;
my $workflow = "ci.yml";
my $json = JSON::PP->new->utf8;
sub fail {
  print STDERR ($ENV{GITHUB_ACTIONS} ? "::error::" : "error: "), @_, "\n";
  exit 1;
}

# One GET of the REST API, decoded; gh prints its own error when it fails.
sub api {
  my ($path) = @_;
  open(my $gh, "-|", "gh", "api", $path) or fail("cannot run gh: $!");
  my $body = do { local $/; <$gh> };
  close $gh or fail("gh api $path failed");
  my $data = eval { $json->decode($body) };
  ref $data eq "HASH" or fail("gh api $path printed what this script does not read");
  return $data;
}

# A time as GitHub writes one, in UTC to the second, as seconds since the
# epoch; undef for anything else.
sub seconds {
  my ($time) = @_;
  return undef unless defined $time && !ref $time
    && $time =~ /\A(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)Z\z/;
  return timegm($6, $5, $4, $3, $2 - 1, $1);
}

my @sample;
my ($red, $passed_again, $cancelled) = (0, 0, 0);
PAGES: for my $page (1 .. 10) {
  my $path = "repos/$repo/actions/workflows/$workflow/runs?event=pull_request&status=completed&per_page=100&page=$page";
  my $listed = api($path)->{workflow_runs};
  ref $listed eq "ARRAY" or fail("gh api $path printed no list of runs");
  for my $run (@$listed) {
    ref $run eq "HASH" && defined $run->{id} && !ref $run->{id} && $run->{id} =~ /\A[1-9][0-9]*\z/
      && defined $run->{head_sha} && !ref $run->{head_sha} && $run->{head_sha} =~ /\A[0-9a-f]{40}\z/
      or fail("gh api $path listed a run this script does not read");
    my $jobs_path = "repos/$repo/actions/runs/$run->{id}/attempts/1/jobs?per_page=100";
    my $first = api($jobs_path);
    my $jobs = $first->{jobs};
    ref $jobs eq "ARRAY" && defined $first->{total_count} && !ref $first->{total_count}
      && $first->{total_count} =~ /\A[0-9]+\z/ && $first->{total_count} == @$jobs
      or fail("gh api $jobs_path printed no whole list of jobs");
    my @verdicts;
    for my $job (@$jobs) {
      ref $job eq "HASH" && defined $job->{conclusion} && !ref $job->{conclusion}
        && defined $job->{name} && !ref $job->{name}
        or fail("gh api $jobs_path listed a job this script does not read");
      push @verdicts, $job->{conclusion};
    }
    # A failed job outranks a cancelled one: a matrix that fails fast
    # cancels the jobs beside the one that failed.
    my @ran = grep { $_->{conclusion} ne "skipped" } @$jobs;
    if (!@ran || grep { !/\A(?:success|skipped|cancelled)\z/ } @verdicts) {
      $red++;
      $passed_again++ if ($run->{conclusion} // "") eq "success";
      next;
    }
    if (grep { $_ eq "cancelled" } @verdicts) {
      $cancelled++;
      next;
    }
    my ($start, $end, $last);
    for my $job (@ran) {
      my ($began, $ended) = (seconds($job->{started_at}), seconds($job->{completed_at}));
      defined $began && defined $ended && $ended >= $began
        or fail("gh api $jobs_path gave $job->{name} no start and end this script reads");
      $start = $began if !defined $start || $began < $start;
      ($end, $last) = ($ended, $job->{name}) if !defined $end || $ended > $end;
    }
    push @sample, { id => $run->{id}, sha => substr($run->{head_sha}, 0, 8), time => $end - $start, last => $last };
    last PAGES if @sample == $want;
  }
  last if @$listed < 100;
}
@sample or fail("no pull-request run of $workflow in $repo passed at its first attempt");

my $n = @sample;
my @times = sort { $a <=> $b } map { $_->{time} } @sample;
# The nearest rank, ceil(percent * n / 100), in integers.
my $rank = sub { my ($percent) = @_; return $times[int(($percent * $n + 99) / 100) - 1] };
my ($p50, $p90, $p99) = map { $rank->($_) } 50, 90, 99;
my $runs = $n == 1 ? "1 run" : "$n runs";
my $fewer = $n < $want ? ", fewer than the $want asked for" : "";
print "gate time of the latest $runs of $workflow in $repo on a pull request that passed at the first attempt$fewer, from the start of its first job to the end of its last:\n";
printf "run %s at %s: %d s, %s last\n", $_->{id}, $_->{sha}, $_->{time}, $_->{last} for @sample;
printf "p50 %d s, p90 %d s, p99 %d s, slowest %d s over %s\n", $p50, $p90, $p99, $times[-1], $runs;
printf "since the oldest of them, %d %s at the first attempt (%d of them passed when run again) and %d %s cancelled\n",
  $red, $red == 1 ? "run failed" : "runs failed", $passed_again, $cancelled, $cancelled == 1 ? "was" : "were";
# Met over the whole sample asked for, and only when no run as recent failed
# at its first attempt, so the runs are consecutive; a cancelled run gave no
# verdict, so it breaks no run of them.
printf "target p50 under 90 s and p99 under 120 s over %d consecutive %s that passed at the first attempt: %s\n",
  $want, $want == 1 ? "run" : "runs", $n == $want && $red == 0 && $p50 < 90 && $p99 < 120 ? "met" : "not met";
' "$repo" "$runs") || exit $?

echo "$report"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  printf '## Gate time\n\n```text\n%s\n```\n' "$report" >> "$GITHUB_STEP_SUMMARY" || {
    if [ -n "${GITHUB_ACTIONS:-}" ]; then
      echo "::error::cannot write the job summary" >&2
    else
      echo "error: cannot write the job summary" >&2
    fi
    exit 1
  }
fi
exit 0
