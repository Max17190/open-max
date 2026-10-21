#!/bin/bash
# Count the workspace's Rust lines two ways, per crate and in total.
#
# Usage: scripts/count-core-lines.sh [--excluded] [checkout]
#
# core-excluding-tests: the lines of crates/<crate>/src/**/*.rs less those
#   of test-only code, so the lines a build without cfg(test) reads, except
#   that a file no module declaration loads still counts.
# total-including-tests: every line of every .rs file under crates/<crate>
#   (src, tests and fixtures alike; build output in crates/<crate>/target
#   aside).
# A test's lines count toward total-including-tests wherever they sit, and
# toward core-excluding-tests nowhere. A line is a physical line: blank lines
# and comments count where they stand. The checkout defaults to the one this
# script is in. --excluded also lists each run of excluded lines as
# path:first-last.
#
# Method. Each file under src/ is split into tokens, so that a bracket, `#`
# or `;` inside a literal or a comment stays part of it: line and block
# comments (block comments nest), strings, byte and C strings with their
# escapes, raw strings with any number of `#`, and char and byte literals,
# told apart from lifetimes and labels. The body of a test-only inline
# module, excluded whole, is matched at once by the same rules rather than
# split, unless the word `mod` appears in it, since a module declared there
# could load a file. The input of
# stringify! and the rules of a macro_rules! definition are text a build
# keeps, not items, so no attribute or `mod` declaration in them is read;
# the input of any other macro is read as code, since item macros such as
# thread_local! pass the attributes in it to the items they make. An item is
# test-only when one of its outer attributes is #[test], a #[<path>::test]
# such as #[tokio::test], or #[cfg(P)] where P cannot hold without `test`:
# `test` itself, all(...) with such an argument, or any(...) with only such
# arguments, so an item under #[cfg(not(test))] or #[cfg(any(unix, test))]
# counts. A test-only item runs from its first outer attribute or doc
# comment to its last token, found by matching brackets over the tokens:
#   - fn, mod, impl, trait, struct, enum, union, macro_rules!, macro calls
#     in braces (`name! {` after any path) and extern blocks end with the
#     brace that closes their body (the first brace outside generic
#     arguments, so not a const argument's), or at their `;` when they have
#     none (`mod tests;`, `struct S(u8);`), after any of const, async,
#     unsafe and extern "ABI";
#   - type, let and static items, and any other item after const, async,
#     unsafe or extern (a const item, an extern crate), end at their `;`;
#   - a block, an unsafe or const block, and if (with its else chain), match,
#     loop, while and for, after any label, end with their closing brace, or
#     after a method call or `?` on them, and a match arm whose body is one
#     of these ends with it and a `,` after it; the body of if, while, match
#     and for is read as rustc reads it, the first brace after a whole
#     operand of the condition or after `..`, so not a block the condition
#     holds, and a brace right after `>` there stops the count;
#   - a parameter, a tuple field, and a field where it is declared (in a
#     struct, union or enum variant) end at the first `,` outside brackets
#     and generic arguments, where every `<` opens them;
#   - a field in a struct expression or pattern, and any other item,
#     statement, match arm or variant, ends at its first `;` or `,` outside
#     brackets, turbofish generics and closure parameters, or before the
#     bracket that closes what holds it;
# and an item that ends with a closing brace takes a `;` right after it.
# An item read as running on past a braced group into the next item or
# statement (an item of a kind this script does not know), or with generic
# arguments that do not balance, stops the count with an error that names
# its file and line, rather than take the code after it for test code.
# Test-only items with only white space and comments between them form one
# run, and the lines from a run's first token to its last are excluded, less
# a first or last line it shares with a token outside the run. An inner
# #![cfg(test)] makes the item whose body it opens test-only, or its whole
# file (the `mod` line that loads the file still counts). A file that a
# test-only `mod name;` loads is excluded whole, and so are the files its
# own `mod` declarations load, unless a crate root (src/lib.rs, src/main.rs,
# src/bin/*.rs or src/bin/*/main.rs) reaches the file through declarations
# that are not test-only. Each file is found as rustc finds it: from the
# declaring file's directory, each inline module around the declaration adds
# its name, then the declaration adds name.rs or name/mod.rs (a raw name
# without its r#). The declaring file's own name goes before the first name
# added, unless it is a crate root, a mod.rs or a file a #[path] loads. A
# #[path] on an inline module or on the declaration takes the place of the
# name it would add, read from the directory reached without that file name.
set -u -o pipefail

list=0
if [ "${1:-}" = --excluded ]; then
  list=1
  shift
fi
if [ $# -gt 1 ]; then
  echo "usage: $0 [--excluded] [checkout]" >&2
  exit 2
fi
checkout=${1:-$(dirname "$0")/..}

exec perl - "$list" "$checkout" <<'PERL'
use strict;
use warnings;

my ($list, $root) = @ARGV;
chdir $root or die "count-core-lines: cannot enter $root: $!\n";
-d 'crates' or die "count-core-lines: no crates/ in $root\n";

my %OPEN = ('(' => 1, '[' => 1, '{' => 1);
my %CLOSE = (')' => '(', ']' => '[', '}' => '{');
my %QUAL = map { $_ => 1 } qw(const async unsafe extern);
my %ITEM = map { $_ => 1 } qw(fn mod impl trait struct enum macro_rules);
my %SEMI = map { $_ => 1 } qw(type let static);
my %BLOCK = map { $_ => 1 } qw(if match loop while for);
# Words an operand follows, which a block can be.
my %DUE = map { $_ => 1 } qw(unsafe const async move return break);
my $CRATE_ROOT = qr{/src/(?:lib|main|bin/[^/]+(?:/main)?)\.rs\z};

# The current file's tokens: type (w word, l literal, t lifetime or label,
# d outer doc comment, p punctuation), text, the offset just past it, and
# the index of the bracket that matches a bracket. Then the offset of each
# newline, the tokens the scan visits (`#`, doc comments and `mod`), each
# stringify and macro_rules word, and the file and the first token of the
# test-only item being read.
our (@T, @X, @Z, @M, @NL, @HOT, @QUOTING, $FILE, $AT);
# The files a #[path] loads.
my %BY_PATH;

sub slurp {
  my ($file) = @_;
  open my $fh, '<:raw', $file or die "count-core-lines: $file: $!\n";
  local $/;
  my $text = <$fh>;
  return defined $text ? $text : '';
}

sub line_count {
  my ($text) = @_;
  my $n = ($text =~ tr/\n//);
  return $n + (length($text) && substr($text, -1) ne "\n" ? 1 : 0);
}

# Every .rs file under a directory, less those under the directories @skip
# names or under a symbolic link to a directory.
sub rust_files {
  my ($top, @skip) = @_;
  my %skip = map { $_ => 1 } @skip;
  my @dirs = ($top);
  my @out;
  while (defined(my $dir = shift @dirs)) {
    opendir my $dh, $dir or next;
    for my $entry (readdir $dh) {
      next if $entry eq '.' || $entry eq '..';
      my $path = "$dir/$entry";
      if (-d $path) { push @dirs, $path unless $skip{$path} || -l $path }
      elsif ($entry =~ /\.rs\z/) { push @out, $path }
    }
  }
  return sort @out;
}

# The line of a byte offset in the current file.
sub line_of {
  my ($offset) = @_;
  my ($lo, $hi) = (0, scalar @NL);
  while ($lo < $hi) {
    my $mid = ($lo + $hi) >> 1;
    if ($NL[$mid] < $offset) { $lo = $mid + 1 } else { $hi = $mid }
  }
  return $lo + 1;
}

sub start_of { $Z[$_[0]] - length $X[$_[0]] }

# The body of an inline module, from its `{` to its `}`, read with the
# literals and comments of the tokens below so that it ends at the brace
# they would match.
my $BODY = qr{
  (?<body> \{ (?:
      [^{}"'/A-Za-z_0-9]++
    | [bc]?r(\#*)".*?"\g{-1}
    | "[^"\\]*(?:\\.[^"\\]*)*"
    | '(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]+\}|.)|[^'\\\n])'
    | [A-Za-z_0-9]\w*+
    | '[A-Za-z_]\w*+
    | //[^\n]*+
    | (?<comment> /\*(?:[^/*]++|/(?!\*)|\*(?!/)|(?&comment))*+\*/ )
    | ['/]
    | (?&body)
  )*+ \} )
}xs;

# Whether the brace at token $n opens the body of a test-only inline module.
sub test_module {
  my ($n) = @_;
  return 0 unless $n >= 2 && $T[$n - 1] eq 'w' && $X[$n - 2] eq 'mod';
  my $k = $n - 3;
  if ($k >= 0 && $X[$k] eq ')' && $M[$k] > 0 && $X[$M[$k] - 1] eq 'pub') { $k = $M[$k] - 2 }
  elsif ($k >= 0 && $X[$k] eq 'pub') { $k-- }
  while ($k >= 0) {
    if ($T[$k] eq 'd') { $k--; next }
    last unless $X[$k] eq ']' && $M[$k] > 0 && $X[$M[$k] - 1] eq '#';
    return 1 if (attribute($M[$k] + 1, $k - 1))[0];
    $k = $M[$k] - 2;
  }
  return 0;
}

sub lex {
  my ($file, $text) = @_;
  @T = @X = @Z = @M = @NL = @HOT = @QUOTING = ();
  push @NL, $-[0] while $text =~ /\n/g;
  my @open;
  # A byte or C string, or a byte literal, is a word (b, c) and a literal;
  # a raw identifier (r#name) is one word.
  while ($text =~ m{\G[ \t\n\r\f]*+(?:
      ( [bc]?r(\#*)".*?"\g{-1}
      | "[^"\\]*(?:\\.[^"\\]*)*"
      | '(?:[\xC0-\xFF][\x80-\xBF]+|\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]+\}|.)|[^'\\\n])' )
    | ((?:r\#)?[A-Za-z_]\w*|[0-9]\w*)
    | ('[A-Za-z_]\w*)
    | (//[^\n]*)
    | (/\*(?:[^/*]++|/(?!\*)|\*(?!/)|(?-1))*+\*/)
    | (=>|->|::|.)
    )}gcxs) {
    my $n = @X;
    if (defined $3) {
      push @HOT, $n if $3 eq 'mod';
      push @QUOTING, $n if $3 eq 'stringify' || $3 eq 'macro_rules';
      push @T, 'w';
      push @X, $3;
    } elsif (defined $7) {
      my $p = $7;
      if ($p eq '{' && test_module($n)) {
        # A test-only inline module is excluded whole, so its body is
        # matched at once rather than read token by token, unless the word
        # `mod` appears in it: a module declared there could load a file.
        my $from = pos($text) - 1;
        pos($text) = $from;
        if ($text =~ m{\G$BODY}gc && substr($text, $from, pos($text) - $from) !~ /\bmod\b/) {
          push @T, 'p', 'p';
          push @X, '{', '}';
          push @Z, $from + 1, pos $text;
          ($M[$n], $M[$n + 1]) = ($n + 1, $n);
          next;
        }
        pos($text) = $from + 1;
      }
      if ($OPEN{$p}) { push @open, $n }
      elsif (my $opener = $CLOSE{$p}) {
        die "count-core-lines: $file:" . line_of(pos $text) . ": unmatched '$p'\n" unless @open && $X[$open[-1]] eq $opener;
        my $o = pop @open;
        ($M[$o], $M[$n]) = ($n, $o);
      } elsif ($p eq '#') { push @HOT, $n }
      push @T, 'p';
      push @X, $p;
    } elsif (defined $1) { push @T, 'l'; push @X, $1 }
    elsif (defined $4) { push @T, 't'; push @X, $4 }
    else {
      my $comment = defined $5 ? $5 : $6;
      next unless $comment =~ m{\A(?:///(?!/)|/\*\*(?![*/]))};
      push @HOT, $n;
      push @T, 'd';
      push @X, $comment;
    }
    push @Z, pos $text;
  }
  die "count-core-lines: $file:" . line_of(start_of($open[-1])) . ": unclosed '$X[$open[-1]]'\n" if @open;
}

# Stop the count at a test-only item whose end cannot be found.
sub unbounded {
  die "count-core-lines: $FILE:" . line_of(start_of($AT)) . ": cannot tell where this test-only item ends\n";
}

# The first token from $k on, outside brackets, whose text is one of @want,
# or -1 when the enclosing bracket closes first.
sub skip_to {
  my ($k, @want) = @_;
  my %want = map { $_ => 1 } @want;
  for (; $k < @X; $k++) {
    return $k if $want{$X[$k]};
    if ($OPEN{$X[$k]}) { $k = $M[$k]; next }
    return -1 if $CLOSE{$X[$k]};
  }
  return -1;
}

# The bracket that closes the one at $k, inside an item. After braces the
# item goes on only with punctuation other than `#` and `{`, or with `as`,
# `else` or a match guard's `if`: anything else starts the next item or
# statement, so the item is of a kind this script does not read.
sub past {
  my ($k) = @_;
  my $n = $M[$k] + 1;
  unbounded() if $X[$k] eq '{' && $n <= $#X
    && ($T[$n] eq 'p' ? $X[$n] eq '#' || $X[$n] eq '{' : !($T[$n] eq 'w' && $X[$n] =~ /\A(?:as|else|if)\z/));
  return $M[$k];
}

# Each function below takes an item's first token (past its attributes)
# and returns its last.
sub semi_end {
  my ($k) = @_;
  for (; $k < @X; $k++) {
    if ($OPEN{$X[$k]}) { $k = past($k); next }
    return $k - 1 if $CLOSE{$X[$k]};
    return $k if $X[$k] eq ';';
  }
  return $#X;
}

# An item's header: `->` is one token, so every `<` opens generic arguments
# and every `>` closes them, and the body is the first brace outside them.
sub block_end {
  my ($k) = @_;
  my $depth = 0;
  for (; $k < @X; $k++) {
    my $x = $X[$k];
    return $M[$k] if $x eq '{' && $depth <= 0;
    if ($OPEN{$x}) { $k = $M[$k]; next }
    return $k - 1 if $CLOSE{$x};
    return $k if $x eq ';';
    $depth += $x eq '<' ? 1 : $x eq '>' ? -1 : 0;
  }
  return $#X;
}

# A type, or a pattern and its type: `->` is one token, so every `<` opens
# generic arguments and every `>` closes them.
sub type_end {
  my ($k) = @_;
  my $depth = 0;
  for (; $k < @X; $k++) {
    my $x = $X[$k];
    if ($OPEN{$x}) { $k = $M[$k]; next }
    if ($CLOSE{$x}) { unbounded() if $depth; return $k - 1 }
    return $k if $x eq ',' && !$depth;
    $depth += $x eq '<' ? 1 : $x eq '>' ? -1 : 0;
  }
  return $#X;
}

sub expr_end {
  my ($k) = @_;
  for (; $k < @X; $k++) {
    my $x = $X[$k];
    if ($x eq '=>' && $k < $#X && (my $from = blocklike_from($k + 1)) >= 0) {
      my $last = blocklike_end($from);
      return $last < $#X && $X[$last + 1] eq ',' ? $last + 1 : $last;
    }
    if ($x eq '|' && $k > 0 && $X[$k - 1] =~ /\A(?:=>|=|:|move)\z/) {
      my $bar = skip_to($k + 1, '|');
      $k = $bar if $bar > 0;
      next;
    }
    if ($x eq '::' && $k < $#X && $X[$k + 1] eq '<') {
      my $depth = 0;
      for ($k++; $k < @X; $k++) {
        if ($OPEN{$X[$k]}) { $k = $M[$k]; next }
        if ($CLOSE{$X[$k]}) { $k--; last }
        $depth += $X[$k] eq '<' ? 1 : $X[$k] eq '>' ? -1 : 0;
        last if $depth == 0;
      }
      next;
    }
    if ($OPEN{$x}) { $k = past($k); next }
    return $k - 1 if $CLOSE{$x};
    return $k if $x eq ';' || $x eq ',';
  }
  return $#X;
}

# Where a block-like expression at $k is read from (past its label), or -1
# when none starts there.
sub blocklike_from {
  my ($k) = @_;
  my $x = $X[$k];
  return $k if $x eq '{' || $BLOCK{$x};
  return $k if ($x eq 'unsafe' || $x eq 'const') && $k < $#X && $X[$k + 1] eq '{';
  return $k + 2 if $T[$k] eq 't' && $k + 2 <= $#X && $X[$k + 1] eq ':';
  return -1;
}

# The last token of a block-like expression at $k, with its else chain and,
# unless $bare, a method call or `?` after it.
sub blocklike_end {
  my ($k, $bare) = @_;
  while (1) {
    my $open;
    if ($X[$k] eq 'for') {
      my $in = skip_to($k + 1, 'in');
      $open = $in < 0 ? -1 : body_open($in + 1);
    } elsif ($X[$k] eq 'if' || $X[$k] eq 'while' || $X[$k] eq 'match') {
      $open = body_open($k + 1);
    } else {
      $open = skip_to($k, '{', ';');
    }
    return semi_end($k) if $open < 0;
    return $open if $X[$open] eq ';';
    my $close = $M[$open];
    return $close if $close == $#X;
    my $next = $X[$close + 1];
    if ($next eq 'else') { $k = $close + 2; next }
    return !$bare && ($next eq '.' || $next eq '?') ? expr_end($close + 1) : $close;
  }
}

# The brace that opens the body of an if, while, match or for, from $k just
# past its keyword or `in`, read as rustc reads a condition: a brace where
# an operand is due is a block the condition holds, and the body is the
# first brace after a whole operand, or after `..`. A brace right after `>`,
# which can close generic arguments or compare, stops the count. -1 when a
# `;` or the enclosing bracket comes first.
sub body_open {
  my ($k) = @_;
  my $due = 1;
  for (; $k < @X; $k++) {
    my ($t, $x) = ($T[$k], $X[$k]);
    if ($x eq '{') {
      return $k unless $due;
      unbounded() if $X[$k - 1] eq '>';
      ($k, $due) = ($M[$k], 0);
    } elsif ($t eq 'w' && $BLOCK{$x}) {
      ($k, $due) = (blocklike_end($k, 1), 0);
    } elsif ($x eq 'let') {
      # A `let` pattern, which can hold braces, runs to its `=`.
      $k = skip_to($k + 1, '=');
      return -1 if $k < 0;
    } elsif ($OPEN{$x}) {
      ($k, $due) = ($M[$k], 0);
    } elsif ($CLOSE{$x} || $x eq ';') {
      return -1;
    } else {
      $due = $t eq 'p' ? $x ne '?' && !($x eq '.' && $X[$k - 1] eq '.') : $t eq 'w' && $DUE{$x} ? 1 : 0;
    }
  }
  return -1;
}

# The brace that opens a macro call's body at $k (`name! {` after any
# path), or -1.
sub macro_brace {
  my ($k) = @_;
  $k++ if $X[$k] eq '::';
  $k += 2 while $k + 2 <= $#X && $T[$k] eq 'w' && $X[$k + 1] eq '::';
  return $k + 2 <= $#X && $T[$k] eq 'w' && $X[$k + 1] eq '!' && $X[$k + 2] eq '{' ? $k + 2 : -1;
}

# The struct, union or enum keyword of the item whose body opens at $open,
# or ''.
sub keyword_before {
  my ($open) = @_;
  for (my $k = $open - 1; $k >= 0; $k--) {
    my $x = $X[$k];
    last if $x eq '}' || $x eq ';' || $OPEN{$x};
    if ($CLOSE{$x}) { $k = $M[$k]; next }
    return $x if $T[$k] eq 'w' && ($x eq 'struct' || $x eq 'enum' || ($x eq 'union' && $T[$k + 1] eq 'w'));
  }
  return '';
}

# Whether the braces at $open declare fields, a type after each colon (a
# struct's, a union's or an enum variant's), rather than give them in a
# struct expression or pattern.
sub declares_fields {
  my ($open) = @_;
  return 1 if keyword_before($open) =~ /\A(?:struct|union)\z/;
  my $outer = enclosing($open);
  return $outer >= 0 && keyword_before($outer) eq 'enum' ? 1 : 0;
}

# An item that ends with a closing brace takes a `;` right after it.
sub item_end {
  my $last = item_last(@_);
  return $last < $#X && $X[$last] eq '}' && $X[$last + 1] eq ';' ? $last + 1 : $last;
}

sub item_last {
  my ($i) = @_;
  my $k = $i;
  if ($X[$k] eq 'pub') { $k++; $k = $M[$k] + 1 if $k < @X && $X[$k] eq '(' }
  return $i - 1 if $k > $#X;
  my $open = enclosing($i);
  # A parameter or a tuple field (not a macro call's arguments).
  return type_end($k) if $open > 0 && $X[$open] eq '(' && $X[$open - 1] ne '!';
  return declares_fields($open) ? type_end($k + 2) : expr_end($k + 2) if $T[$k] eq 'w' && $k < $#X && $X[$k + 1] eq ':';
  my $from = blocklike_from($k);
  return blocklike_end($from) if $from >= 0;
  my %qualifiers;
  while ($k < @X && $T[$k] eq 'w' && $QUAL{$X[$k]}) {
    $qualifiers{$X[$k]} = 1;
    $k++;
    $k++ if $X[$k - 1] eq 'extern' && $k < @X && $T[$k] eq 'l';
  }
  return $i - 1 if $k > $#X;
  my $x = $X[$k];
  return block_end($k) if $ITEM{$x} || ($x eq 'union' && $k < $#X && $T[$k + 1] eq 'w');
  # An extern block.
  return $M[$k] if $x eq '{' && $qualifiers{extern};
  return semi_end($k) if %qualifiers || $SEMI{$x};
  my $brace = macro_brace($k);
  return $brace >= 0 ? $M[$brace] : expr_end($k);
}

# Whether the cfg predicate in tokens $a..$b cannot hold without `test`.
sub needs_test {
  my ($a, $b) = @_;
  # A comma may follow the predicate: #[cfg(test,)].
  $b-- if $b > $a && $X[$b] eq ',';
  return $X[$a] eq 'test' ? 1 : 0 if $a == $b;
  my $op = $X[$a];
  return 0 unless ($op eq 'all' || $op eq 'any') && $X[$a + 1] eq '(' && $M[$a + 1] == $b;
  my ($from, @args) = ($a + 2);
  for (my $k = $from; $k < $b; $k++) {
    if ($OPEN{$X[$k]}) { $k = $M[$k]; next }
    next unless $X[$k] eq ',';
    push @args, [$from, $k - 1] if $k > $from;
    $from = $k + 1;
  }
  push @args, [$from, $b - 1] if $b > $from;
  my $held = grep { needs_test(@$_) } @args;
  return $op eq 'all' ? ($held ? 1 : 0) : (@args && $held == @args ? 1 : 0);
}

# The attribute in tokens $a..$b (inside its brackets): whether it makes
# its item test-only, and the file a #[path] names.
sub attribute {
  my ($a, $b) = @_;
  my ($k, @path) = ($a);
  while ($k <= $b && $T[$k] eq 'w') {
    push @path, $X[$k++];
    last unless $k <= $b && $X[$k] eq '::';
    $k++;
  }
  return (0) unless @path;
  return (1) if $path[-1] eq 'test';
  return (needs_test($k + 1, $M[$k] - 1)) if "@path" eq 'cfg' && $k <= $b && $X[$k] eq '(' && $M[$k] > $k + 1;
  if ("@path" eq 'path' && $k < $b && $X[$k] eq '=' && $T[$k + 1] eq 'l') {
    (my $file = $X[$k + 1]) =~ s/\Ar?#*"|"#*\z//g;
    return (0, $file);
  }
  return (0);
}

# The opening bracket of the brackets that hold token $i, or -1.
sub enclosing {
  my ($i) = @_;
  for (my $k = $i - 1; $k >= 0; $k--) {
    if ($CLOSE{$X[$k]}) { $k = $M[$k]; next }
    return $k if $OPEN{$X[$k]};
  }
  return -1;
}

# The current file's excluded line numbers, its `mod name;` declarations,
# and whether an inner #![cfg(test)] makes the whole file test-only.
sub scan {
  my ($file_test, $excluded_to, @extents, @mods, %path_at) = (0, -1);
  # Text a build keeps: the brackets of stringify!'s input and of a
  # macro_rules! definition's rules, in order. One inside another adds
  # nothing, since the outer one holds every token of it.
  my @quoted;
  for my $k (@QUOTING) {
    my $open = $k + 2;
    $open++ if $X[$k] eq 'macro_rules' && $open <= $#X && $T[$open] eq 'w';
    push @quoted, [$open, $M[$open]] if $k < $#X && $X[$k + 1] eq '!' && $open <= $#X && $OPEN{$X[$open]};
  }
  my $q = 0;
  for my $i (@HOT) {
    $q++ while $q < @quoted && $quoted[$q][1] < $i;
    next if $q < @quoted && $quoted[$q][0] < $i;
    if ($X[$i] eq 'mod') {
      next unless $i + 2 <= $#X && $T[$i + 1] eq 'w' && $X[$i + 2] eq ';';
      # The inline modules around it, outermost first, each with its #[path].
      my @inline;
      for (my $o = enclosing($i); $o >= 0; $o = enclosing($o)) {
        unshift @inline, [$X[$o - 1], $path_at{$o - 2}] if $X[$o] eq '{' && $o >= 2 && $X[$o - 2] eq 'mod' && $T[$o - 1] eq 'w';
      }
      push @mods, { name => $X[$i + 1], path => $path_at{$i}, inline => \@inline,
        test => $file_test || $i <= $excluded_to, line => line_of(start_of($i)) };
    } elsif ($X[$i] eq '#' && $i + 2 <= $#X && $X[$i + 1] eq '!' && $X[$i + 2] eq '[') {
      next unless (attribute($i + 3, $M[$i + 2] - 1))[0];
      my $body = enclosing($i);
      if ($body < 0) { $file_test = 1; next }
      # The item whose body this opens starts after the previous `;` or
      # closing brace in the same brackets.
      my $k = $body - 1;
      $k = $CLOSE{$X[$k]} ? $M[$k] - 1 : $k - 1 while $k >= 0 && $X[$k] ne ';' && $X[$k] ne '}' && !$OPEN{$X[$k]};
      push @extents, [$k + 1, $M[$body]];
      $excluded_to = $M[$body] if $M[$body] > $excluded_to;
    } else {
      # An item's outer attributes and doc comments, then the item.
      my ($k, $test, $path) = ($i, 0);
      while ($k <= $#X) {
        if ($T[$k] eq 'd') { $k++; next }
        last unless $X[$k] eq '#' && $k < $#X && $X[$k + 1] eq '[';
        my ($is_test, $file) = attribute($k + 2, $M[$k + 1] - 1);
        $test ||= $is_test;
        $path = $file if defined $file;
        $k = $M[$k + 1] + 1;
      }
      next if $k == $i;
      # An item inside one already excluded is excluded with it.
      if ($test && $i > $excluded_to) {
        $AT = $i;
        my $last = item_end($k);
        push @extents, [$i, $last];
        $excluded_to = $last if $last > $excluded_to;
      }
      if (defined $path) {
        if ($X[$k] eq 'pub') { $k++; $k = $M[$k] + 1 if $k < @X && $X[$k] eq '(' }
        $path_at{$k} = $path;
      }
    }
  }
  # Excluded: the lines from a run of test-only tokens' first to its last,
  # less a first or last line that a token outside the run shares.
  my (@runs, %excluded);
  for (sort { $a->[0] <=> $b->[0] } @extents) {
    if (@runs && $_->[0] <= $runs[-1][1] + 1) { $runs[-1][1] = $_->[1] if $_->[1] > $runs[-1][1] }
    else { push @runs, [@$_] }
  }
  for (@runs) {
    my ($first, $last) = @$_;
    my ($from, $to) = (line_of(start_of($first)), line_of($Z[$last] - 1));
    $from++ if $first > 0 && line_of($Z[$first - 1] - 1) == $from;
    $to-- if $last < $#X && line_of(start_of($last + 1)) == $to;
    $excluded{$_} = 1 for $from .. $to;
  }
  return ([sort { $a <=> $b } keys %excluded], \@mods, $file_test);
}

sub normalize {
  my @parts;
  for (split m{/}, $_[0]) {
    next if $_ eq '' || $_ eq '.';
    if ($_ eq '..' && @parts && $parts[-1] ne '..') { pop @parts } else { push @parts, $_ }
  }
  return join '/', @parts;
}

# The file a `mod name;` declared in $file loads, by the rules the header
# states, or undef when a declaration that is not test-only has none.
sub module_file {
  my ($file, $mod) = @_;
  (my $dir = $file) =~ s{/[^/]*\z}{};
  my ($own) = $BY_PATH{$file} || $file =~ $CRATE_ROOT || $file =~ m{/mod\.rs\z} ? () : $file =~ m{([^/]+)\.rs\z};
  # A raw name (r#type) names its file without the r#.
  for (@{$mod->{inline}}) {
    my ($name, $path) = @$_;
    $dir = defined $path ? "$dir/$path" : join '/', $dir, (defined $own ? $own : ()), $name =~ s/\Ar#//r;
    undef $own;
  }
  my $name = $mod->{name} =~ s/\Ar#//r;
  my @candidates = defined $mod->{path} ? "$dir/$mod->{path}"
    : map { join '/', $dir, (defined $own ? $own : ()), $_ } "$name.rs", "$name/mod.rs";
  for (@candidates) {
    next unless -f $_;
    my $found = normalize($_);
    $BY_PATH{$found} = 1 if defined $mod->{path};
    return $found;
  }
  return undef unless $mod->{test};
  die "count-core-lines: $file:$mod->{line}: cannot find the file of test-only module `$mod->{name}`\n";
}

my @crates = grep { -d } glob 'crates/*';
my (%lines, %excluded, %mods, %file_test);
for my $crate (@crates) {
  for my $file (rust_files("$crate/src")) {
    my $text = slurp($file);
    $lines{$file} = line_count($text);
    $FILE = $file;
    lex($file, $text);
    ($excluded{$file}, $mods{$file}, $file_test{$file}) = scan();
  }
}
# A file a #[path] loads keeps its modules beside it, so every such file is
# known before a module is looked up from any file, whatever order the files
# sort in. Repeated until no more are found, since a #[path] inside an
# inline module is read from where its own file was loaded.
for (my $known = -1; $known < keys %BY_PATH;) {
  $known = keys %BY_PATH;
  for my $file (sort keys %lines) {
    module_file($file, { %$_, test => 0 }) for grep { defined $_->{path} } @{$mods{$file}};
  }
}
# The module tree. A crate root reaches the files a build without
# cfg(test) reads through declarations that are not test-only. Any other
# file is test-only when an inner #![cfg(test)] makes it so, or a test-only
# declaration loads it, or any declaration in a test-only file does.
my (%production, %test_file);
my @work = grep { $_ =~ $CRATE_ROOT } sort keys %lines;
while (defined(my $file = shift @work)) {
  next if $file_test{$file} || $production{$file}++;
  push @work, grep { defined } map { module_file($file, $_) } grep { !$_->{test} } @{$mods{$file} || []};
}
@work = grep { $file_test{$_} } sort keys %lines;
push @work, map { my $file = $_; map { module_file($file, $_) } grep { $_->{test} } @{$mods{$file}} } sort keys %lines;
while (defined(my $file = shift @work)) {
  next if $production{$file} || $test_file{$file}++;
  push @work, map { module_file($file, { %$_, test => 1 }) } @{$mods{$file} || []};
}

my ($core_sum, $total_sum) = (0, 0);
my @rows;
for my $crate (@crates) {
  my ($core, $total) = (0, 0);
  for my $file (rust_files("$crate/src")) {
    my $excluded = $test_file{$file} ? [1 .. $lines{$file}] : $excluded{$file};
    $core += $lines{$file} - @$excluded;
    next unless $list && @$excluded;
    my @runs = ([$excluded->[0], $excluded->[0]]);
    for (@$excluded[1 .. $#$excluded]) {
      if ($_ == $runs[-1][1] + 1) { $runs[-1][1] = $_ } else { push @runs, [$_, $_] }
    }
    print "$file:$_->[0]-$_->[1]\n" for @runs;
  }
  for my $file (rust_files($crate, "$crate/target")) {
    $total += exists $lines{$file} ? $lines{$file} : line_count(slurp($file));
  }
  (my $name = $crate) =~ s{\Acrates/}{};
  push @rows, [$name, $core, $total];
  $core_sum += $core;
  $total_sum += $total;
}
printf "%-6s core-excluding-tests=%-6s total-including-tests=%s\n", @$_ for @rows, ['TOTAL', $core_sum, $total_sum];
PERL
