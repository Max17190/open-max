# Install

Homebrew, on macOS or Linux:

```sh
brew install Max17190/tap/openmax
```

Or without Homebrew:

```sh
curl -fsSL https://useopenmax.dev/install.sh | sh
```

Or with bun or npm, if you already have a JavaScript runtime:

```sh
bun install -g openmax-cli    # or: npm install -g openmax-cli
```

The command is `openmax` either way; only the package name carries the `-cli`,
because `openmax` on npm belongs to an unrelated 2015 package. This path fetches
the same native binary through a small launcher script. Up to and including
2026.10.1 that script runs under `node` only, so with those versions this is
the one install that needs Node.js on your `PATH`, even when bun did the
installing. Releases after 2026.10.1 run it under `node` when Node.js is on
your `PATH` and under `bun` when it is not, so either runtime is enough. Bun
blocks install scripts by default, which is fine here: the binary is fetched on
first run instead, once, and cached.

Either way there is no Rust toolchain and no build. The script picks the right
prebuilt binary for your machine, checks it against the published SHA-256, and
puts `openmax` in `~/.openmax/bin` next to the settings and sessions the
harness already keeps there. It adds that directory to your `PATH` through your
shell profile, which only a new shell reads, so open one (or
`source ~/.openmax/bin/env`) before typing a bare `openmax`. The shell
installer of releases after 2026.10.1 ends with the first command to run (see
[Use](getting-started.md#use)), naming the binary by its absolute path, with the home
directory spelled out (`/Users/you/.openmax/bin/openmax`, for example), so it
works in the shell that ran the install.

The check uses `sha256sum`, or `shasum -a 256` where there is no `sha256sum`
(macOS 14 and earlier), and is skipped, with a message, only on a system with
neither. The installers of 2026.10.1 and earlier skip it on any system without
`sha256sum`.

macOS and Linux, x86_64 and arm64. Linux gets a glibc build, or a static musl
build automatically when glibc is older than 2.35, so it also runs on Alpine and
inside slim containers. Windows is not supported.

Set `OPENMAX_NO_MODIFY_PATH=1` to leave your `PATH` alone, or
`OPENMAX_INSTALL_DIR` to install somewhere else:

```sh
curl -fsSL https://useopenmax.dev/install.sh | OPENMAX_NO_MODIFY_PATH=1 sh
```

`useopenmax.dev/install.sh` redirects to the current release's installer, so it
follows new releases without the link changing.

Every [release](https://github.com/Max17190/open-max/releases) also carries the
plain tarballs and a `sha256.sum`, so you can download and verify by hand
instead of piping a script into a shell.

**From source** instead, which needs [Rust](https://rustup.rs) 1.89 or newer:

```sh
git clone https://github.com/Max17190/open-max.git
cd open-max
cargo install --path crates/tui --locked
```

Or run it in place with `cargo run --release -p openmax`.
