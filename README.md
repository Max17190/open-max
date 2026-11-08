<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/openmax-lockup-dark.svg">
    <img src="docs/assets/openmax-lockup.svg" alt="open max" height="52">
  </picture>
</p>

<h3 align="center">Most powerful & token efficient harness for agents</h3>

<p align="center">
  A coding agent for your terminal. Its prompt is under a thousand tokens and<br>
  never changes mid-session, so the cache keeps serving it.
</p>

<p align="center">
  <a href="https://github.com/Max17190/open-max/releases/latest"><img src="https://img.shields.io/github/v/release/Max17190/open-max?label=release&color=155e3d" alt="Latest release"></a>
  <a href="https://github.com/Max17190/open-max/actions/workflows/ci.yml"><img src="https://github.com/Max17190/open-max/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-155e3d" alt="MIT license"></a>
</p>

<p align="center">
  <img src="docs/assets/demo.svg" width="780" alt="An Open Max session in the terminal: the agent runs the failing tests, reads and edits the file, and runs them again until they pass. Then it writes itself a tool that searches docs/ and calls it on its next step.">
</p>
<p align="center"><sub>A real session. Only the model's replies are scripted.</sub></p>

Open Max is one Rust binary: an agent loop, seven tools, a terminal UI that streams every step, and a core that keeps the trust decisions. When it needs a new capability, the agent writes it as a file, such as a [tool](docs/extending.md) or a skill, and can call a tool it wrote on its very next step.

## Install

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh    # macOS and Linux
brew install Max17190/tap/openmax                    # Homebrew
npm install -g openmax-cli                           # or: bun install -g openmax-cli
```

## Get started

```sh
openmax --login     # ChatGPT, an OpenAI or Anthropic key, or any compatible API
cd ~/code/my-app
openmax             # pick the model with /model, then describe a task
```

Open Max asks before it works in a new project. Its shell commands, tools and hooks run with your user's access, and it has no telemetry. For your own model server, CI and headless runs, see [getting started](docs/getting-started.md). Setting up with a coding agent? Point it at [useopenmax.dev/setup.md](https://useopenmax.dev/setup.md).

## Docs

[Install](docs/install.md) · [Getting started](docs/getting-started.md) · [Configuration](docs/configuration.md) · [Usage](docs/usage.md) · [Extending](docs/extending.md) · [Design](docs/design.md) · [Native execution and privacy](docs/execution.md) · [stdio protocol](docs/stdio-protocol.md) · [Development](docs/development.md)

## License

MIT
