<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openmax-lockup-dark.svg">
    <img src="docs/assets/openmax-lockup.svg" alt="open max" height="52">
  </picture>
</p>

<h3 align="center">A coding agent you can hold to your own checks</h3>

<p align="center">
  A coding agent for your terminal. Its built-in prompt is under 700 tokens, and each<br>
  agent step in a session sends it as the same bytes, so providers can cache it.
</p>

<p align="center">
  <a href="https://github.com/Max17190/open-max/releases/latest"><img src="docs/assets/release-badge.svg" alt="Latest release"></a>
  <a href="https://github.com/Max17190/open-max/actions/workflows/ci.yml"><img src="https://github.com/Max17190/open-max/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-155e3d" alt="MIT license"></a>
</p>

<p align="center">
  <img src="docs/assets/demo.svg" width="780" alt="Terminal demo: Open Max fixes a failing test, then writes and runs a tool that searches the docs.">
</p>

Open Max is one Rust binary: an agent loop, seven tools, a terminal UI that streams every step, and a core that keeps the trust decisions. When it needs a new capability, the agent writes it as a file, such as a [tool](docs/extending.md) or a skill, and can call a tool it wrote on its very next step. Within 100 ms of a `kill -9`, no process in the process groups of its tools is still running ([how](docs/execution.md)).

**Held to your checks.** [examples/run-task](examples/run-task/README.md) runs one task in your repository and refuses the end of the turn while a command you name fails, sending its output back to the model; after eight refusals the turn ends `unverified`. Once Open Max has exited, the script runs your commands again itself and prints one JSON result: each check before and after the work, the patch, and a command that replays every step offline from one file, byte for byte.

## Install

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh    # macOS and Linux
brew install Max17190/tap/openmax                    # Homebrew
npm install -g openmax-cli                           # or: bun install -g openmax-cli
```

The script adds `openmax` to your `PATH` for new shells; open one, or run `source ~/.openmax/bin/env`.

## Get started

```sh
openmax --login     # ChatGPT, an OpenAI or Anthropic key, or any compatible API
cd ~/code/my-app
openmax             # pick the model with /model, then describe a task
```

Open Max asks before it works in a new project. Its shell commands, tools and hooks run with your user's access, and it has no telemetry. For your own model server, CI and headless runs, see [getting started](docs/getting-started.md). Setting up with a coding agent? Point it at [useopenmax.dev/setup.md](https://useopenmax.dev/setup.md).

## Docs

[Install](docs/install.md) · [Getting started](docs/getting-started.md) · [Configuration](docs/configuration.md) · [Usage](docs/usage.md) · [Extending](docs/extending.md) · [Design](docs/design.md) · [Native execution and privacy](docs/execution.md) · [Hold a run to your own checks](docs/verified-execution.md) · [stdio protocol](docs/stdio-protocol.md) · [Development](docs/development.md)

## License

MIT
