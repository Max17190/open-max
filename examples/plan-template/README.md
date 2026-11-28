# Plan into PLAN.md before editing

`plan.md` is a prompt template that turns `/plan <task>` into one message:
read what the task touches, write the plan to `PLAN.md`, change nothing else,
and stop for your review. The planning step is a file, not a mode: you read
and edit the plan like any other file, and a later turn or a new session reads
it again instead of planning from scratch. It suits a smaller or local model:
each later turn can work from one written step instead of the whole task.

## Install

```sh
mkdir -p .agents/prompts
cp ~/open-max/examples/plan-template/plan.md .agents/prompts/
```

`~/open-max` is your clone of this repository. Put it in
`~/.openmax/prompts/` instead to have `/plan` in every project. A template is
message content, read again on every use: it adds nothing to the system prompt
and needs no restart or approval ([prompt templates](../../docs/extending.md#prompt-templates)).

## Use

```text
/plan add a --verbose flag to the CLI
```

`$ARGUMENTS` in the template becomes the text after `/plan`. The same line
works as `openmax -p "/plan add a --verbose flag to the CLI"`. When the turn
ends, read `PLAN.md`, change what you disagree with, then send a follow-up such
as:

```text
Carry out step 1 of PLAN.md, run its check, and mark it done in PLAN.md.
```

## A planning turn without edits

The template asks the model to write only `PLAN.md`; it is an instruction, and
the model can still call other tools. To have the harness refuse its edits,
run the planning turn in `readonly`:

```text
/permissions readonly
/plan add a --verbose flag to the CLI
```

In `readonly` every call to a tool classified as mutating is refused,
`write_file` of `PLAN.md` and `bash` among them, while the model can still read
and search. That is the harness's gate, not a sandbox: an approved hook, or an
approved external tool marked `mutating = false`, still runs and can write
files. When you need a hard guarantee that nothing writes, run the session in
an OS sandbox or container with the project mounted read-only. The template
tells it to give the plan in its reply when the write is refused. Then switch
back and keep the plan:

```text
/permissions ask
Write the plan from your last reply to PLAN.md, then stop.
```

`/permissions` saves the mode for this project, so switch it back before you
want edits again. From a shell, `openmax --permissions readonly` does the same.
In `ask` the planning turn can write, and each mutating call raises an
approval card unless an approved `allow` rule in `.openmax/permissions.toml`
covers it. To review every write, first remove or narrow `allow` rules that
could permit one, `bash` rules included; then approve the card for `PLAN.md`
and decline the rest.

## Test

```sh
cargo build -p openmax
python3 examples/plan-template/test_plan_template.py target/debug/openmax
```

It copies the template into throwaway repositories and drives the binary
against a scripted model on 127.0.0.1. It checks that `openmax --check --all`
lists the template, that `/plan <task>` sends the template's body with the task
in place of `$ARGUMENTS` and adds nothing to the system prompt or tools, that a
`write_file` of `PLAN.md` lands in `auto`, and that in `readonly` both that
write and a `bash` write are refused and no file changes. Whether a given model
follows the plan is up to the model; the test does not judge that.
