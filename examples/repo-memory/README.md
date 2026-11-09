# Carry a repository verification recipe into a fresh session

This optional example uses Open Max's existing project memory. It records one
successful public verification command and stages one memory file in a fresh
checkout. The first line goes in the memory index. Open Max reads the recipe
body on demand.

## Capture and stage

Use Python 3.11 or later and Git 2.36 or later on Linux or macOS. Use absolute
entries in `PATH` and run capture from the repository root. Select the files
that define the verification path, including its entry point and relevant
manifests and lockfiles. Use public argv and an environment identifier that
binds the runtime and toolchain you provisioned.
Use an immutable runtime image identifier when comparing sandbox runs.
The following commands use the Open Max repository root.

```sh
recipe_dir=$(mktemp -d)
recipe_dir=$(cd "$recipe_dir" && pwd -P)
recipe="$recipe_dir/recipe.json"
repo=https://github.com/Max17190/open-max
environment=rust-1.94.0-offline

sha=$(python3 examples/repo-memory/repo_memory.py --repository "$repo" --artifact "$recipe" capture \
  --environment "$environment" --input Cargo.toml --input Cargo.lock \
  --input crates/core/Cargo.toml --input crates/core/src/memory.rs \
  -- cargo test --offline -p open-max-core memory::tests)

python3 examples/repo-memory/repo_memory.py --repository "$repo" --artifact "$recipe" stage \
  --environment "$environment" --sha256 "$sha"
```

Keep the capture SHA-256 with the artifact. Pass it as a trusted input when
staging in another checkout with `--repo <checkout>`. The caller supplies the
repository identity in both commands. The utility also checks the configured
origin when present.

Capture runs the explicit argv without an implicit shell and records the
successful exit, elapsed time, and stdout and stderr hashes. Use a verification
command you trust and keep the artifact outside the checkout. Stage uses the
record as data. It checks the source ancestry, watched committed and working
files, executable hash, platform and supplied environment identifier. These
checks cover the declared fingerprints. Rerun public verification when other
repository or environment changes could affect the recipe. The environment
identifier is caller-supplied and should cover tools resolved through wrappers.

Stage prints `staged` or `stale`. Require `staged` for a warm experiment.
If a stale copy is already staged, evict it using the same artifact and pin:

```sh
python3 examples/repo-memory/repo_memory.py --repository "$repo" --artifact "$recipe" evict --sha256 "$sha"
```

Staging is idempotent. Eviction requires the exact staged bytes. Modified,
tracked or symlinked memory paths are refused. Stage and evict in a quiescent
checkout before launching a session. Open Max's normal memory index can fade
the entry from discovery while leaving the file available for recall.

## Offline check

```sh
cargo build --offline -p openmax
python3 examples/repo-memory/test_repo_memory.py target/debug/openmax
```

The test exercises capture, cold and warm sessions, stale omission and
eviction, and a scripted loopback model response. To compare real task cost,
use distinct held-out tasks and fresh sessions, vary only staged memory, keep
the model and grader fixed, and account for capture and revalidation alongside
all task attempts.
