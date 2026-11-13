# Contributing

Found a bug or want a change? Open an issue or send a pull request. To report a vulnerability, follow [SECURITY.md](SECURITY.md) instead.

## Build and test

[Development](docs/development.md) lists the commands that build and check the workspace, and [Testing](docs/testing.md) explains the gates and how to run them locally.

## What a change has to meet

Every pull request runs the CI workflow, and these jobs have to pass. [Development](docs/development.md) states each budget and floor.

- **Test**, on ubuntu-latest and macos-latest: every test under cargo-nextest, the doc tests, the examples' tests, and `scripts/check-test-count.sh` against the test count floor.
- **Suite time**: `scripts/check-suite-time.sh 3` on ubuntu-latest, against three times the suite's time budget.
- **Clippy**: `cargo +1.97.0 clippy --workspace --all-targets --locked -- -D warnings`, with no warnings.
- **Thesis**, on ubuntu-latest and macos-latest: the thesis tests in a release build, each holding a bound, such as the frozen prefix's byte cap and the core source's line ratchet.
- **Release plan**: `dist plan --output-format=json`, so a release config it cannot plan fails the pull request.

The **Core share** job reports the share of recent commits that change `crates/core/`.

Commit subjects follow the conventional commit form, `type(scope): summary`, such as `fix(agent): never run the last call of a reply cut at max_tokens`. A pull request description has three sections: Why, Summary and Test Plan.

## Growing the core

Open Max stays small: one agent loop, seven built-in tools and a short system prompt. New capability arrives as tools, skills, prompt templates and hooks, files the agent can write, test and remove, rather than as code in the core. [Design](docs/design.md) gives the reasoning and [Extending](docs/extending.md) the file formats.
