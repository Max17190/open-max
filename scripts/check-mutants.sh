#!/bin/bash
# Judge one cargo-mutants run: the mutation score of the workspace, or the
# mutants in one pull request's change.
#
# Usage: scripts/check-mutants.sh [--diff] [--mutants N] [--missed FILE] DIR...
#
# Each DIR is the output directory (mutants.out) of one shard of one run: the
# same commit, arguments and shard count. Run this from the root of the tree
# the mutants were made from: it reads the source files they are in, and
# .cargo/mutants-allowed.txt.
#
# A mutant ends one of six ways:
#   caught     a test failed;
#   missed     every test passed;
#   timed out  the tests ran past the timeout in .cargo/mutants.toml;
#   signalled  a signal ended cargo while it built the mutant or ran its
#              tests (cargo-mutants 27.1.0 sums that up as Failure), as when
#              the kernel kills cargo at the shard's memory cap;
#   unviable   it does not build, so there is no program to test;
#   not built  it is in a function, or a statement, under a #[cfg(...)] that
#              is false for the target the run built (cfg.txt in DIR, which
#              CI writes with `rustc --print cfg`, else this machine's) both
#              with and without cfg(test), such as macOS-only code in a run
#              on Linux: it changed nothing the tests ran, so it is neither
#              caught nor missed.
# A mutant listed in .cargo/mutants-allowed.txt that timed out or was
# signalled counts as caught: the list gives the reason a hang, or the
# signal, is how it fails. Unlisted, either counts as missed: neither shows a
# test that failed. A listed mutant that was missed still counts as missed:
# the list lets a change through, it does not catch anything. Any other
# outcome is refused: a mutant that was only checked, never built or tested
# (cargo-mutants' Success, from a run with --check), a Failure no signal
# ended, or a summary cargo-mutants 27.1.0 does not write.
#
# Without --diff, the score is caught / (caught + missed), unviable and unbuilt
# mutants left out, and the run fails when the score is under the floor below.
# The floor is a ratchet that sits a margin under the score of the run that
# last raised it, because two runs of the same code do not decide every mutant
# alike: a test that fails by chance, such as a timing bound a loaded runner
# misses, catches whichever mutant it ran under. The run that set the floor
# (Actions run 41444364783, on 2d139e8) caught 6,783 of the 8,024 mutants it
# scored, 84.53%. For 191 of them the only test to fail was
# first_frame::provider_setup_hides_keys_and_returns_input_to_the_same_tui in
# crates/tui/tests/frame_io.rs, at its deadline for the terminal's keyboard
# probe. It failed that way for one of them that the allowlist gives as
# equivalent, and for a mutant in code Linux does not compile, so a catch that
# failure alone made cannot be told from chance. The floor's basis leaves those
# 191 out, 6,592 of the 8,024, 82.15%: a fix of that test takes them away, and
# the floor does not count on them. Two full runs on main of the same code
# (Actions runs 39644754820 and 40304868989, of the trees of fa18395a and
# c47ddc13 on main, which differ only in README.md) scored 7,332 and 7,331
# mutants and decided 44 of them differently; of the 19 the second caught and
# the first missed, a timing bound alone caught 16. Each mutant that differs
# moves the caught count by one, either way, so the caught counts of two runs
# of unchanged code differ by a net whose standard deviation is the square root
# of how many differ. At the highest rate that pair allows with 95% confidence,
# 56.6 in 7,331, 62 of the 8,024 mutants the run that set the floor scored
# would differ: a standard deviation of 7.9. The margin, 0.26 points, is 21 of
# them, so a run of unchanged code, those 191 aside, fails only when more than
# 21 turn from caught to missed: about 3 runs in 1,000 at that pair's rate. The
# pair predates the keyboard-probe test: derive the margin again from the next
# full run on main, set beside that run on the files that did not change
# between them. Raise the floor, in the commit that makes the tests catch more,
# to the score a run of that commit reports less the margin, which the report
# gives (the workflow runs on any branch by hand); lower it only in a commit
# that removes code and its tests on purpose, and say there why. The target is
# 95%, and 90% in each file: under them the run reports how far there is to go
# and passes.
#
# With --diff (a pull request's `cargo mutants --in-diff`), the run fails when
# any mutant was missed, timed out or was signalled and is not listed.
#
# --mutants N refuses a run whose shards do not hold N mutants between them,
# the number `cargo mutants --list` gave for the run: a shard that is missing,
# or that tested another set, would leave its mutants out of the judgement.
#
# --missed FILE writes the missed mutants, the timeouts and signals counted
# missed among them, one name per line. In GitHub Actions the report also
# goes to the job summary.
set -u -o pipefail

floor=81.89
margin=0.26
target=95
file_target=90

usage() {
  echo "usage: $0 [--diff] [--mutants N] [--missed FILE] DIR..." >&2
  exit 2
}
diff=0
missed=
expected=
while [ $# -gt 0 ]; do
  case "$1" in
    --diff) diff=1; shift ;;
    --mutants) [ $# -ge 2 ] && [ -n "$2" ] && [ -z "${2//[0-9]/}" ] || usage; expected=$2; shift 2 ;;
    --missed) [ $# -ge 2 ] || usage; missed=$2; shift 2 ;;
    --) shift; break ;;
    -*) usage ;;
    *) break ;;
  esac
done
[ $# -ge 1 ] || usage
# JSON::PP is one of perl's core modules, but a minimal Linux image can have a
# perl without it: the perl-base package of Debian and Ubuntu.
perl -MJSON::PP -e 1 2> /dev/null || {
  echo "error: $0 needs perl with its core module JSON::PP" >&2
  exit 2
}

DIFF=$diff EXPECTED=$expected MISSED=$missed FLOOR=$floor MARGIN=$margin TARGET=$target FILE_TARGET=$file_target exec perl -e '
use strict;
use warnings;
use JSON::PP;

my ($diff, $expected, $missed_out, $floor, $margin, $target, $file_target) =
    @ENV{qw(DIFF EXPECTED MISSED FLOOR MARGIN TARGET FILE_TARGET)};
my @dirs = @ARGV;
sub refuse { print STDERR "error: $_[0]\n"; exit 2 }

# The cfg of the target the run built, as `rustc --print cfg` prints it.
my %cfg;
{
    my $printed;
    for my $dir (grep { -e "$_/cfg.txt" } @dirs) {
        open my $fh, "<", "$dir/cfg.txt" or refuse("$dir/cfg.txt: $!");
        my $text = do { local $/; <$fh> };
        refuse("$dir/cfg.txt differs from another shard'"'"'s: the shards built different targets")
            if defined $printed && $text ne $printed;
        $printed = $text;
    }
    $printed //= `rustc --print cfg`;
    refuse("no cfg.txt in the run, and `rustc --print cfg` failed") if !defined $printed || $printed eq "";
    for my $line (split /\n/, $printed) {
        $line =~ s/\s+//g;
        $cfg{$line} = 1 if length $line;
    }
    # `test` is left out: the tests build each crate twice, with cfg(test) for
    # its unit tests and without it for its integration tests and the binary
    # (see built_somewhere).
}

my %allowed;
if (open my $fh, "<", ".cargo/mutants-allowed.txt") {
    while (my $line = <$fh>) {
        chomp $line;
        next if $line =~ /^\s*(?:#|$)/;
        my ($name, $reason) = $line =~ /^(\S.*?)\s+#\s+(\S.*)$/
            or refuse(".cargo/mutants-allowed.txt:$.: not a mutant'"'"'s name, then \" # \" and the reason it may survive");
        $allowed{$name} = $reason;
    }
}

my (@mutants, %dir_of);
for my $dir (@dirs) {
    my %here;
    my $path = "$dir/outcomes.json";
    open my $fh, "<", $path or refuse("$path: $!");
    my $run = eval { decode_json(do { local $/; <$fh> }) } or refuse("$path: not a cargo-mutants outcomes file");
    # cargo-mutants writes end_time once it has tested every mutant of the
    # shard, or found that the unmutated tests fail (refused next); a run that
    # was stopped has none. A shard that left no outcomes at all is caught by
    # --mutants.
    defined $run->{end_time} or refuse("$path: the run did not finish");
    my @outcomes = @{ $run->{outcomes} || [] };
    my @baseline = grep { !ref $_->{scenario} && $_->{scenario} eq "Baseline" } @outcomes;
    @baseline == 1 && $baseline[0]{summary} eq "Success"
        or refuse("$path: the unmutated tests did not pass, so no mutant was tested");
    for my $outcome (grep { ref $_->{scenario} } @outcomes) {
        my $mutant = $outcome->{scenario}{Mutant} or refuse("$path: an outcome that is neither the baseline nor a mutant");
        my ($file, $line, $column, $what) = $mutant->{name} =~ /^(.+?):(\d+):(\d+): (.+)\z/s
            or refuse("$path: a mutant name with no file, line and column: $mutant->{name}");
        refuse("$mutant->{name} is in $dir_of{$mutant->{name}} and $dir: the shards overlap")
            if exists $dir_of{$mutant->{name}};
        $here{$mutant->{name}} = $dir;
        # cargo-mutants runs a mutant'"'"'s phases (build, then test) until one
        # does not succeed, so the last phase is the one that ended its run.
        my $last = ref $outcome->{phase_results} eq "ARRAY" ? $outcome->{phase_results}[-1] : undef;
        my $status = ref $last eq "HASH" ? $last->{process_status} : undef;
        push @mutants, {
            name => $mutant->{name},
            listed_as => "$file: $what",
            file => $file,
            line => $line,
            column => $column,
            what => $what,
            function => $mutant->{function} ? $mutant->{function}{span}{start}{line} : undef,
            summary => $outcome->{summary} // "",
            phase => ref $last eq "HASH" ? lc($last->{phase} // "") : "",
            status => $status,
        };
    }
    %dir_of = (%dir_of, %here);
}
refuse("the shards hold " . @mutants . " mutants, and the run listed $expected: a shard is missing, or tested another set")
    if length $expected && @mutants != $expected;

# Whether a #[cfg(...)] predicate holds for the run'"'"'s target.
our $test_build = 0;
sub holds {
    my ($text) = @_;
    my @tokens;
    push @tokens, $1 while $text =~ /\G\s*("[^"]*"|\w+|[(),=])/gc;
    $text =~ /\G\s*\z/gc or refuse("cannot read #[cfg($text)]");
    my $at = 0;
    my $holds = predicate(\@tokens, \$at, $text);
    $at == @tokens or refuse("cannot read #[cfg($text)]");
    return $holds;
}
sub predicate {
    my ($tokens, $at, $text) = @_;
    my $name = $tokens->[$$at++] // "";
    $name =~ /^\w+$/ or refuse("cannot read #[cfg($text)]");
    my $next = $tokens->[$$at] // "";
    if ($next eq "=") {
        my $value = $tokens->[++$$at] // refuse("cannot read #[cfg($text)]");
        $$at++;
        return exists $cfg{"$name=$value"} ? 1 : 0;
    }
    if ($next ne "(") {
        # test is the build'"'"'s own (see built_somewhere), and true and false
        # are the literals cfg(true) and cfg(false); any other name holds when
        # the target sets it.
        return $test_build if $name eq "test";
        return $name eq "true" ? 1 : 0 if $name eq "true" || $name eq "false";
        return exists $cfg{$name} ? 1 : 0;
    }
    $$at++;
    my @args;
    until (($tokens->[$$at] // refuse("cannot read #[cfg($text)]")) eq ")") {
        push @args, predicate($tokens, $at, $text);
        $$at++ if ($tokens->[$$at] // "") eq ",";
    }
    $$at++;
    return $args[0] ? 0 : 1 if $name eq "not" && @args == 1;
    return (grep { $_ } @args) ? 1 : 0 if $name eq "any";
    return (grep { !$_ } @args) ? 0 : 1 if $name eq "all";
    refuse("cannot read #[cfg($text)]");
}

my %source;
sub source_of {
    my ($file) = @_;
    $source{$file} //= do {
        open my $fh, "<", $file or refuse("$file: $! (run this from the root of the tree the mutants were made from)");
        [ map { s/\r?\n\z//r } <$fh> ];
    };
}
sub indent_of { $_[0] =~ /^(\s*)/; length $1 }

# Whether a predicate holds in either build the tests make of a crate: with
# cfg(test) for its unit tests, or without it for its integration tests and
# the binary.
sub built_somewhere {
    my ($text) = @_;
    for my $test (1, 0) {
        local $test_build = $test;
        return 1 if holds($text);
    }
    return 0;
}

# Whether the run built the code a mutant changed: false when the function it
# is in, or a statement around it in that function, is under a #[cfg(...)]
# that is false for the target in both builds of the crate (built_somewhere).
# A function'"'"'s span starts at its first attribute or doc comment; a
# statement is read as rustfmt lays it out, one line ending in ";" or ",", or
# lines up to the one that closes it at the attribute'"'"'s indentation.
sub built {
    my ($mutant) = @_;
    return 1 unless defined $mutant->{function};
    my $src = source_of($mutant->{file});
    my $at = $mutant->{function} - 1;
    while ($at < @$src && $src->[$at] =~ m{^\s*(?:#|//|$)}) {
        return 0 if $src->[$at] =~ /^\s*#\[cfg\((.*)\)\]\s*$/ && !built_somewhere($1);
        $at++;
    }
    for my $i ($at + 1 .. $mutant->{line} - 2) {
        last if $i >= @$src;
        my ($indent, $predicate) = $src->[$i] =~ /^(\s*)#\[cfg\((.*)\)\]\s*$/ or next;
        my $first = $i + 1;
        $first++ while $first < @$src && $src->[$first] =~ /^\s*#\[/;
        my $last = $first;
        if ($first < @$src && $src->[$first] !~ /[;,]\s*$/) {
            my $n = $first + 1;
            {
                $n++ while $n < @$src && ($src->[$n] =~ /^\s*$/ || indent_of($src->[$n]) > length $indent);
                $last = $n < @$src && $src->[$n] =~ /^\Q$indent\E[\}\)\]]/ ? $n : $n - 1;
                # "} else {" closes one block of the statement and opens the
                # next.
                if ($last == $n && $src->[$n] =~ /\{\s*$/) {
                    $n++;
                    redo;
                }
            }
        }
        return 0 if $mutant->{line} - 1 >= $first && $mutant->{line} - 1 <= $last && !built_somewhere($predicate);
    }
    return 1;
}

# cargo-mutants sums up a mutant whose last phase a signal ended as Failure,
# that phase'"'"'s status written {"Signalled": N}. A Failure, or any summary,
# that this does not describe is refused below, unbuilt mutants included: the
# run is not one this script knows how to read.
my %is = map { $_ => [] } qw(caught missed listed_timeout timeout listed_signalled signalled unviable unbuilt);
for my $mutant (@mutants) {
    my ($summary, $status, $listed) = ($mutant->{summary}, $mutant->{status}, exists $allowed{ $mutant->{listed_as} });
    $mutant->{signal} = $status->{Signalled} if ref $status eq "HASH" && ($status->{Signalled} // "") =~ /^\d+\z/;
    my $state = $summary eq "CaughtMutant" ? "caught"
        : $summary eq "MissedMutant" ? "missed"
        : $summary eq "Unviable" ? "unviable"
        : $summary eq "Timeout" ? ($listed ? "listed_timeout" : "timeout")
        : $summary eq "Failure" && defined $mutant->{signal} ? ($listed ? "listed_signalled" : "signalled")
        : $summary eq "Success" ? refuse("$mutant->{name}: it was only checked, never built or tested (a run with --check)")
        : $summary eq "Failure" ? refuse("$mutant->{name}: an outcome this script does not know: Failure, "
            . ($mutant->{phase} ? "its $mutant->{phase} phase ending " . JSON::PP->new->canonical->allow_nonref->encode($status) : "with no phase"))
        : refuse("$mutant->{name}: an outcome this script does not know: $summary");
    push @{ $is{ built($mutant) ? $state : "unbuilt" } }, $mutant;
}
my $by_place = sub { $a->{file} cmp $b->{file} || $a->{line} <=> $b->{line} || $a->{column} <=> $b->{column} || $a->{name} cmp $b->{name} };
my @missed = sort $by_place @{ $is{missed} }, @{ $is{timeout} }, @{ $is{signalled} };
my $caught = @{ $is{caught} } + @{ $is{listed_timeout} } + @{ $is{listed_signalled} };
my $scored = $caught + @missed;

# Outcomes a listed mutant counts as caught, and an unlisted one as missed.
my $split = sub {
    my ($what, $listed, $unlisted) = @_;
    sprintf("%7d %s: %d listed with a reason, counted caught; %d not, counted missed",
        @$listed + @$unlisted, $what, scalar @$listed, scalar @$unlisted);
};
my @report = (
    sprintf("%d mutants in %d shard%s:", scalar @mutants, scalar @dirs, @dirs == 1 ? "" : "s"),
    sprintf("%7d caught", scalar @{ $is{caught} }),
    sprintf("%7d missed", scalar @{ $is{missed} }),
    $split->("timed out", $is{listed_timeout}, $is{timeout}),
    $split->("ended by a signal", $is{listed_signalled}, $is{signalled}),
    sprintf("%7d unviable", scalar @{ $is{unviable} }),
    sprintf("%7d not built for this target", scalar @{ $is{unbuilt} }),
);
# A percentage with at most two decimals, in hundredths of a point.
my $hundredths_of = sub {
    my ($what, $value) = @_;
    my ($whole, $hundredths) = $value =~ /^(\d+)(?:\.(\d{1,2}))?$/ or refuse("the $what $value is not a percentage");
    $whole * 100 + ($hundredths // 0) * (length($hundredths // "00") == 1 ? 10 : 1);
};
my $points = sub { sprintf "%d.%02d", $_[0] / 100, $_[0] % 100 };
my $percent = sub { $points->($_[0]) . "%" };
my ($target_bp, $file_target_bp) = ($hundredths_of->("target", $target), $hundredths_of->("file target", $file_target));
# The fewest of the missed that would have to be caught for caught / scored
# to reach a percentage given in hundredths of a point.
my $to_reach = sub {
    my ($bp, $caught, $scored) = @_;
    my $short = $bp * $scored - $caught * 10000;
    $short > 0 ? int(($short + 9999) / 10000) : 0;
};
my $failed = 0;
my @blocking;
if ($diff) {
    @blocking = grep { !exists $allowed{ $_->{listed_as} } } @missed;
    $failed = @blocking > 0;
    push @report, "", @blocking
        ? (scalar(@blocking) . " missed, timed out or ended by a signal, and not listed in .cargo/mutants-allowed.txt:", map { "  $_->{name}" } @blocking)
        : "every mutant in the change was caught or is listed in .cargo/mutants-allowed.txt";
} else {
    refuse("no mutant was caught or missed, so there is no score") if $scored == 0;
    my ($floor_bp, $margin_bp) = ($hundredths_of->("floor", $floor), $hundredths_of->("margin", $margin));
    my $score_bp = int($caught * 10000 / $scored);
    $failed = $caught * 10000 < $floor_bp * $scored;
    push @report, "", "score: " . $percent->($score_bp) . ", $caught caught of $scored caught or missed; floor $floor%, target $target%";
    if ($failed) {
        push @report, "the score is under its floor of $floor%";
    } else {
        my $needed = $to_reach->($target_bp, $caught, $scored);
        push @report, $needed == 0 ? "the target of $target% is met"
            : $points->($target_bp - $score_bp) . " points under the target of $target%; catching $needed of the missed would meet it";
        my $raise_bp = $score_bp - $margin_bp;
        push @report, "the floor in scripts/check-mutants.sh can be raised to " . $percent->($raise_bp) . ", the score less its margin of $margin points"
            if $raise_bp > $floor_bp;
    }
}
# Each file'"'"'s score, worst first, and how many of its missed would have to
# be caught for it to reach the file target.
my %per_file;
$per_file{ $_->{file} }{caught}++ for @{ $is{caught} }, @{ $is{listed_timeout} }, @{ $is{listed_signalled} };
$per_file{ $_->{file} }{missed}++ for @missed;
my @files = map {
    my ($caught, $missed) = ($per_file{$_}{caught} // 0, $per_file{$_}{missed} // 0);
    +{ file => $_, caught => $caught, missed => $missed, needed => $to_reach->($file_target_bp, $caught, $caught + $missed) }
} keys %per_file;
@files = sort { $a->{caught} * ($b->{caught} + $b->{missed}) <=> $b->{caught} * ($a->{caught} + $a->{missed}) || $a->{file} cmp $b->{file} } @files;
push @report, "", sprintf("per file, worst first (%d of %d under %s%%):", scalar(grep { $_->{needed} } @files), scalar @files, $file_target),
    sprintf("%7s %7s %7s %7s  %s", "score", "caught", "missed", "to $file_target%", "file"),
    map { sprintf "%7s %7d %7d %7d  %s", $percent->(int($_->{caught} * 10000 / ($_->{caught} + $_->{missed}))), @$_{qw(caught missed needed file)} } @files
    if @files;
my %ended_as = (build => "its build", test => "its tests", check => "its check");
my @signalled = sort $by_place @{ $is{listed_signalled} }, @{ $is{signalled} };
push @report, "", "ended by a signal:", map {
    sprintf "  %s (%s ended by signal %d, %s)", $_->{name}, $ended_as{ $_->{phase} } // "its run", $_->{signal},
        exists $allowed{ $_->{listed_as} } ? "listed, counted caught" : "counted missed"
} @signalled if @signalled;
my @unbuilt = sort $by_place @{ $is{unbuilt} };
push @report, "", "not built for this target:", map { "  $_->{name}" } @unbuilt if @unbuilt;
print "$_\n" for @report;
if (@missed) {
    print "\nmissed:\n";
    print "  $_->{name}\n" for @missed;
}
if ($ENV{GITHUB_ACTIONS}) {
    for my $mutant (@blocking) {
        # A workflow command message escapes %, CR and LF; a mutant can be
        # "replace % with /".
        my $what = $mutant->{what} =~ s/%/%25/gr =~ s/\r/%0D/gr =~ s/\n/%0A/gr;
        print "::error file=$mutant->{file},line=$mutant->{line},col=$mutant->{column}::Not caught, and not listed in .cargo/mutants-allowed.txt: $what\n";
    }
    print "::error::the mutation score is under its floor of $floor%\n" if $failed && !$diff;
}
if (length $missed_out) {
    open my $out, ">", $missed_out or refuse("$missed_out: $!");
    print $out "$_->{name}\n" for @missed;
    close $out or refuse("$missed_out: $!");
}
if (my $summary = $ENV{GITHUB_STEP_SUMMARY}) {
    open my $out, ">>", $summary or refuse("$summary: $!");
    print $out "## ", $diff ? "Mutants in the change" : "Mutation score", "\n\n```text\n", join("\n", @report), "\n```\n";
    print $out "\n<details><summary>", scalar(@missed), " missed</summary>\n\n```text\n", join("", map { "$_->{name}\n" } @missed), "```\n\n</details>\n" if @missed;
    close $out;
}
exit($failed ? 1 : 0);
' -- "$@"
