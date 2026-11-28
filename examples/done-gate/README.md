# Refuse "Done." while your check fails

An agent can say it is finished while the tests still fail. These two files
hold every turn of a session, interactive or `openmax -p`, to one command you
name: while it fails, Open Max refuses the end of the turn and sends the
failing output back to the model, which works on from there. When it passes,
the end stands.

They are a completion gate, a `turn_end` hook with `blocking = true`
([hooks](../../docs/extending.md#hooks)). Nothing here is new to Open Max; this
directory is the smallest set of files that turns the gate on.

- `done-gate.toml`: the hook. Its `args` line names your check.
- `done-gate.sh`: runs the check with `sh -c` in the project root, exits 0
  when it passes, and otherwise prints the command, its exit code and the end
  of its output.

## Try it in a minute

In a scratch repository, with `~/open-max` your clone of this repository and a
model already configured:

```sh
mkdir /tmp/done-gate-demo && cd /tmp/done-gate-demo && git init -q
printf 'echo hullo\n' > greet.sh
printf 'test "$(sh greet.sh)" = hello || { echo "expected hello, got $(sh greet.sh)"; exit 1; }\n' > test.sh
mkdir -p .openmax/hooks
cp ~/open-max/examples/done-gate/done-gate.sh .openmax/hooks/
sed 's/"cargo test"/"sh test.sh"/' ~/open-max/examples/done-gate/done-gate.toml > .openmax/hooks/done-gate.toml
openmax "Reply with the single word Done. and nothing else."
```

Answer `y` at the trust prompt. When the model replies `Done.`, the transcript
shows the gate's refusal and its reason as the next message:

```text
turn_end gate 'done-gate' refused completion (refusal 1 of this turn; this gate may refuse 2 more); its reason continues the turn:
Not done: `sh test.sh` exits 1. Fix the cause, then end the turn again.
expected hello, got hullo
```

The model then fixes `greet.sh` or tries again, and the next end is judged the
same way.

## Use it on your project

```sh
mkdir -p .openmax/hooks
cp ~/open-max/examples/done-gate/done-gate.toml ~/open-max/examples/done-gate/done-gate.sh .openmax/hooks/
```

Then edit the `args` line of `.openmax/hooks/done-gate.toml` and put your check
where `"cargo test"` is, for example `"npm test"` or `"make test && make lint"`.
Run `openmax --check`: the hook's row reads `ok ... (hook on blocking
turn_end)` once it will run. The gate takes effect at the next turn, with no
restart.

The hook names its script by a path relative to the project root, so it is a
project hook. Commit both files to share the gate with everyone who runs Open
Max in the repository.

## Outside auto mode

`y` at the trust prompt saves `auto`, where a valid hook runs without
approval. In `ask` and `readonly` a hook is inert until a person approves its
exact content, and `openmax --check` reports it as `err ... inert because its
content is not approved`. Approve it from a shell in the project root:

```sh
openmax --approve .openmax/hooks/done-gate.toml
```

That approves the manifest and the script it runs in one act and prints both
hashes. An edit to either file revokes the approval, and an approved gate that
changed fails closed: tool calls are refused and every turn ends `unverified`
until the approved content is back or you approve the new content. So in `ask` an edit that swaps your check for one that passes
turns every end `unverified` instead of passing. In `auto` no approval is
consulted, and the model may edit any file the gate reads, the hook files among
them; see the limits below.

## When the check cannot pass

`max_refusals = 3` in the hook file: after three refusals in one turn, the
next failing end is not sent to the model, and the turn ends `unverified`
(the TUI says so; `openmax -p` exits 4). Raise it up to 8, Open Max's cap per
turn, or lower it to stop a stuck turn sooner. Each refusal is one more round
on a longer transcript.

`timeout_secs = 600` is the most a completion gate may run. A gate that times
out refuses, so lower it to a little over your check's time.

## Limits

- A passing check means what that check tests and no more.
- In `auto`, the model's shell can change the code your check reads, a test
  among them, and the hook files themselves. Review the diff. For a caller that
  must not trust the session's own verdict, [run-task](../run-task/README.md)
  runs the checks again after the process exits.
- At an end the model did not ask for (a cancel, a spent budget, an error) the
  gate still runs, with at most 60 seconds, but nothing can be refused.

## Test

```sh
cargo build -p openmax
python3 examples/done-gate/test_done_gate.py target/debug/openmax
```

It copies these files into throwaway repositories as above and drives the
binary against a scripted model on 127.0.0.1: a `Done.` while the test fails is
refused with the test's output, and the end after the fix is recorded as
`turn_verified`; a check that keeps failing ends the turn `unverified` after
three refusals; and in `ask` the gate is inert until `openmax --approve`, then
refuses, then verifies once a person fixes the code, and an edit to the
approved hook makes the next end `unverified`.
