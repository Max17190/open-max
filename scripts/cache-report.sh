#!/bin/bash
# Report how much of the prompt the provider served from its cache, and the
# gaps between requests, over a directory of recorded sessions.
#
# Usage: scripts/cache-report.sh <sessions dir>
#
# The harness appends one line to a session's <id>.usage.jsonl in the data
# dir's sessions/ (~/.openmax/sessions by default) for every model request
# whose reply reported its token usage: a JSON object with ts, the Unix time
# in seconds, prompt_tokens, completion_tokens and, when the endpoint
# reported it, cached_tokens, the prompt tokens the provider served from its
# cache. Any other request (one to an endpoint that reports no usage, or one
# that failed, timed out or was cancelled before its usage came) leaves no
# line and counts in no number here: a session none of whose requests
# reported usage has no file, so the zero-cache share leaves it out, and a
# gap can span requests that left no line. This reads every *.usage.jsonl
# file in the directory and nothing else, and writes nothing.
#
# A line is a request when it is a JSON object whose ts, prompt_tokens and
# completion_tokens are non-negative integers and whose cached_tokens is
# absent, null or one; a blank line is passed over, and any other line is
# skipped and counted. A session is a file with at least one request; a file
# with none is counted and left out. A request that reported no
# cached_tokens counts as 0 cached, its prompt tokens counted all the same,
# so a protocol whose cache use the client does not read shows as uncached;
# each row says how many of its requests reported none. The harness's own
# hit rate, sessions::cache_hit_totals, leaves such a request out instead;
# both shares here count it, so an endpoint that reports no cached tokens
# cannot make either look better by dropping out.
#
# It prints a row for each session, the most prompt tokens first and ties by
# name: its requests, prompt tokens, cached tokens, their share, and its
# requests that reported no cached tokens. Then, when any, what it skipped,
# and three lines:
#   cache: cached tokens over prompt tokens, each summed over the sessions
#     with more than 100,000 prompt tokens, against the 75% target.
#   zero-cache: the sessions whose cached tokens sum to 0, of every session,
#     against the 5% target, and how many of them, if any, reported none at
#     all.
#   gaps: p50 and p90 of the gaps between the ts of each request and the ts
#     of the request before it in the same file, a gap over 600 s left out
#     as idle, with how many gaps are in the sample, against the p90 target
#     of 90 s. A percentile is the gap at rank ceil(q * n) of the n in the
#     sample in ascending order (the nearest rank), so it is one of them. A
#     negative gap (the clock moved back) stays in the sample and is counted.
#
# A report, not a gate: it exits 0 whatever it measures; 1 when it cannot
# read the directory or a usage file in it, or the directory holds no
# *.usage.jsonl file; and 2 on a usage error or when perl lacks JSON::PP or
# B. Needs bash and perl with its core modules JSON::PP and B.
# crates/tui/tests/thesis.rs holds the method to the synthetic sessions in
# crates/tui/tests/fixtures/sessions.
set -u -o pipefail

[ $# -eq 1 ] && [ -d "$1" ] || { echo "usage: $0 <sessions dir>" >&2; exit 2; }
perl -MJSON::PP -MB -e 1 2> /dev/null || {
  echo "error: $0 needs perl with its core modules JSON::PP and B" >&2
  exit 2
}

perl -MJSON::PP -MB -e '
use strict;
use warnings;

my $dir = shift;
sub fail { print STDERR "error: @_\n"; exit 1 }
opendir(my $listing, $dir) or fail("cannot read $dir: $!");
my @files = sort grep { /\.usage\.jsonl\z/ && -f "$dir/$_" } readdir $listing;
closedir $listing;
@files or fail("$dir holds no *.usage.jsonl file");

# A JSON integer at or above zero, as decoded: no fraction, no string.
sub count {
  my ($value) = @_;
  return 0 unless defined $value && !ref $value;
  my $flags = B::svref_2object(\$value)->FLAGS;
  return ($flags & B::SVf_IOK) && !($flags & (B::SVf_NOK | B::SVf_POK)) && $value >= 0;
}

sub grouped {
  my $digits = reverse shift;
  $digits =~ s/(\d{3})(?=\d)/$1,/g;
  return scalar reverse $digits;
}

sub share {
  my ($part, $whole) = @_;
  return $whole ? sprintf("%.1f%%", 100 * $part / $whole) : "-";
}

my $json = JSON::PP->new->utf8;
my (@sessions, @gaps);
my ($skipped, $empty, $idle) = (0, 0, 0);
for my $file (@files) {
  open(my $in, "<", "$dir/$file") or fail("cannot read $dir/$file: $!");
  my %session = (name => $file =~ s/\.usage\.jsonl\z//r, requests => 0, prompt => 0, cached => 0, unreported => 0);
  my $last;
  while (my $line = <$in>) {
    next unless $line =~ /\S/;
    my $request = eval { $json->decode($line) };
    unless (ref $request eq "HASH" && count($request->{ts}) && count($request->{prompt_tokens})
      && count($request->{completion_tokens})
      && (!defined $request->{cached_tokens} || count($request->{cached_tokens})))
    {
      $skipped++;
      next;
    }
    $session{requests}++;
    $session{prompt} += $request->{prompt_tokens};
    if (defined $request->{cached_tokens}) {
      $session{cached} += $request->{cached_tokens};
    } else {
      $session{unreported}++;
    }
    if (defined $last) {
      my $gap = $request->{ts} - $last;
      if ($gap > 600) { $idle++ } else { push @gaps, $gap }
    }
    $last = $request->{ts};
  }
  close $in;
  if ($session{requests}) { push @sessions, \%session } else { $empty++ }
}

@sessions = sort { $b->{prompt} <=> $a->{prompt} || $a->{name} cmp $b->{name} } @sessions;
my $width = 7;
for (@sessions) { $width = length $_->{name} if length $_->{name} > $width }
printf "%-*s %8s %14s %14s %7s %10s\n", $width, "session", "requests", "prompt tokens", "cached tokens", "cached", "unreported";
for my $s (@sessions) {
  printf "%-*s %8s %14s %14s %7s %10s\n", $width, $s->{name}, grouped($s->{requests}), grouped($s->{prompt}),
    grouped($s->{cached}), share($s->{cached}, $s->{prompt}), grouped($s->{unreported});
}
if ($skipped || $empty) {
  printf "skipped: %s, %s %s with no request\n", $skipped == 1 ? "1 line that is not a usage record"
    : grouped($skipped) . " lines that are not usage records", grouped($empty), $empty == 1 ? "file" : "files";
}

my @large = grep { $_->{prompt} > 100_000 } @sessions;
my ($cached, $prompt) = (0, 0);
for (@large) { $cached += $_->{cached}; $prompt += $_->{prompt} }
if (@large) {
  printf "cache: %s of prompt tokens cached, %s of %s, over %s %s above 100,000 prompt tokens; target 75%%\n",
    share($cached, $prompt), grouped($cached), grouped($prompt), grouped(scalar @large),
    @large == 1 ? "session" : "sessions";
} else {
  print "cache: no session above 100,000 prompt tokens; target 75%\n";
}

my @cold = grep { $_->{cached} == 0 } @sessions;
my $silent = grep { $_->{unreported} == $_->{requests} } @cold;
if (@sessions) {
  printf "zero-cache: %s of %s %s (%s) cached no prompt token%s; target 5%%\n",
    grouped(scalar @cold), grouped(scalar @sessions), @sessions == 1 ? "session" : "sessions",
    share(scalar @cold, scalar @sessions), $silent ? sprintf(", %s of them reporting no cached tokens at all", grouped($silent)) : "";
} else {
  print "zero-cache: no session; target 5%\n";
}

@gaps = sort { $a <=> $b } @gaps;
my $n = @gaps;
my $negative = grep { $_ < 0 } @gaps;
# The nearest rank, ceil(percent * n / 100), in integers.
my $rank = sub { my ($percent) = @_; return $gaps[int(($percent * $n + 99) / 100) - 1] };
my $left = sprintf "%s longer %s left out as idle", grouped($idle), $idle == 1 ? "gap" : "gaps";
my $back = $negative ? sprintf(", %s of them negative", grouped($negative)) : "";
if ($n) {
  printf "gaps: p50 %s s, p90 %s s over %s %s of at most 600 s between consecutive requests of a session%s; %s; target p90 90 s\n",
    $rank->(50), $rank->(90), grouped($n), $n == 1 ? "gap" : "gaps", $back, $left;
} else {
  printf "gaps: no gap of at most 600 s between consecutive requests of a session; %s; target p90 90 s\n", $left;
}
' "$1"
