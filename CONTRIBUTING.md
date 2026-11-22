# Contributing

Found a bug or want a change? Open an issue or send a pull request. To report a vulnerability, follow [SECURITY.md](SECURITY.md) instead.

## Build and test

[Development](docs/development.md) lists the commands that build and check the workspace, and [Testing](docs/testing.md) explains the gates and how to run them locally.

## What a change has to meet

Every pull request runs the CI workflow, and these jobs have to pass. [Development](docs/development.md) states each budget and floor.

- **Test**, on ubuntu-latest and macos-latest: every test under cargo-nextest, the doc tests, and `scripts/check-test-count.sh` against the test count floor.
- **Examples**, on ubuntu-latest and macos-latest: the repository memory and run task examples' scripts against a debug build of the binary.
- **Clippy**: `cargo +1.97.0 clippy --workspace --all-targets --locked -- -D warnings`, with no warnings. `clippy.toml` refuses `std::fs::write`, `File::create` and `File::set_len` outside tests, since each can leave a saved file cut short: write through `open_max_core::sessions::write_atomic`, or put `#[expect(clippy::disallowed_methods, reason = "...")]` on the call with the reason it is safe.
- **Thesis**, on ubuntu-latest and macos-latest: the thesis tests in a release build, each holding a bound, such as the frozen prefix's byte cap and the core source's line ratchet.
- **Release plan**: `dist plan --output-format=json`, so a release config it cannot plan fails the pull request.

The **Core share** job reports the share of recent commits that change `crates/core/`.

Commit subjects follow the conventional commit form, `type(scope): summary`, such as `fix(agent): never run the last call of a reply cut at max_tokens`. A pull request description has three sections: Why, Summary and Test Plan.

## Growing the core

Open Max stays small: one agent loop, seven built-in tools and a short system prompt. New capability arrives as tools, skills, prompt templates and hooks, files the agent can write, test and remove, rather than as code in the core. [Design](docs/design.md) gives the reasoning and [Extending](docs/extending.md) the file formats.
