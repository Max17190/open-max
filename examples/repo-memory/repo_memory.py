#!/usr/bin/env python3
"""Capture and stage one verified repository recipe for Open Max."""
import argparse
from contextlib import contextmanager, suppress
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time

MEMORY = Path(".openmax/memory/repo-verification.md")
DESCRIPTION = "# Repository verification recipe"
DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
MAX_ARTIFACT = 8192
MAX_MEMORY = 4096
MAX_INPUT = 4 * 1024 * 1024
REPOSITORY = r"https://[a-z0-9.-]+/[A-Za-z0-9._/-]+"
TOKEN = r"[A-Za-z0-9_./:+,-]{1,160}"
SHA256 = r"[0-9a-f]{64}"
COMMIT = r"(?:[0-9a-f]{40}|[0-9a-f]{64})"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(value, root_ok=False):
    require(matches(r"[A-Za-z0-9_./-]{1,160}", value), "invalid repository-relative path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value and
            (root_ok or value != ".") and ".git" not in path.parts, "invalid repository-relative path")
    return path


@contextmanager
def directory(path, create=False):
    path = path.absolute()
    require(".." not in path.parts, "parent traversal is not supported")
    fd = os.open(path.anchor, DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, DIRECTORY, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_at(fd, name, limit):
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(name, flags, dir_fd=fd), "rb") as file:
        info = os.fstat(file.fileno())
        require(stat.S_ISREG(info.st_mode), "expected a regular file")
        data = file.read(limit + 1)
    require(len(data) <= limit, "file exceeds the size limit")
    return data, info


def create_at(fd, name, data):
    temp = f".repo-memory-{secrets.token_hex(12)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(temp, flags, 0o600, dir_fd=fd), "wb") as file:
        try:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
            os.link(temp, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        finally:
            os.unlink(temp, dir_fd=fd)


def git(repo, *argv, missing_ok=False):
    command = ["git", "--literal-pathspecs", "-C", str(repo), *argv]
    env = {**process_env(), "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""}
    result = subprocess.run(command, env=env, capture_output=True, check=False)
    if missing_ok and result.returncode == 1:
        return None
    require(result.returncode == 0, "repository metadata could not be verified")
    return result.stdout


def head(repo, identity):
    with directory(repo):
        require(Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()) == repo,
                "pass the repository root")
        origin = git(repo, "config", "--get", "remote.origin.url", missing_ok=True)
        require(origin is None or origin.decode().strip().removesuffix(".git") == identity,
                "origin does not match the declared public repository")
        return git(repo, "rev-parse", "--verify", "HEAD").decode().strip()


def committed_inputs(repo, commit, paths):
    files = git(repo, "ls-tree", "-z", commit, "--", *paths).split(b"\0")
    hashes = {}
    for entry in filter(None, files):
        metadata, name = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode().split()
        require(mode in {"100644", "100755"} and kind == "blob", "watched inputs must be tracked regular files")
        require(int(git(repo, "cat-file", "-s", blob)) <= MAX_INPUT, "watched input exceeds the size limit")
        hashes[name.decode()] = {"sha256": digest(git(repo, "cat-file", "blob", blob)), "mode": mode}
    require(hashes.keys() == set(paths), "watched inputs must be tracked regular files")
    return hashes


def checkout_matches(repo, hashes):
    for name, expected in hashes.items():
        path = repo / name
        with directory(path.parent) as fd:
            data, info = read_at(fd, path.name, MAX_INPUT)
            if digest(data) != expected["sha256"] or bool(info.st_mode & stat.S_IXUSR) != (expected["mode"] == "100755"):
                return False
    return not git(repo, "diff-index", "--cached", "--name-only", "HEAD", "--", *hashes)


def process_env():
    return {**{name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}, "LC_ALL": "C"}


def fingerprint(command, tag):
    env = process_env()
    found = shutil.which(command, path=env.get("PATH", os.defpath))
    require(found is not None, "verification executable is unavailable")
    executable = Path(found).resolve(strict=True)
    with executable.open("rb") as file:
        executable_sha = hashlib.file_digest(file, "sha256").hexdigest()
    return executable, {"tag": tag, "platform": platform.system(), "machine": platform.machine(),
                        "executable_sha256": executable_sha}


def valid_argv(argv):
    return isinstance(argv, list) and 0 < len(argv) <= 16 and matches(r"[A-Za-z0-9_-]{1,80}", argv[0]) and all(
        matches(TOKEN, arg) and not re.search(r"(^|[+:,])/|[A-Za-z]:/", arg) and
        not (arg.startswith("-") and "/" in arg) and ".." not in Path(arg).parts for arg in argv
    )


def artifact_at(path, pin):
    with directory(path.parent) as fd:
        data, _ = read_at(fd, path.name, MAX_ARTIFACT)
    require(digest(data) == pin, "artifact SHA-256 does not match")
    artifact = json.loads(data)
    require(isinstance(artifact, dict) and set(artifact) == {
        "schema", "repository", "source_commit", "inputs", "argv", "cwd", "environment", "result"
    } and type(artifact["schema"]) is int and artifact["schema"] == 1, "invalid recipe artifact")
    require(matches(REPOSITORY, artifact["repository"]) and matches(COMMIT, artifact["source_commit"]) and
            valid_argv(artifact["argv"]), "invalid recipe provenance or argv")
    relative(artifact["cwd"], root_ok=True)
    inputs = artifact["inputs"]
    require(isinstance(inputs, dict) and 0 < len(inputs) <= 16, "invalid watched inputs")
    for name, file in inputs.items():
        relative(name)
        require(isinstance(file, dict) and set(file) == {"sha256", "mode"} and
                matches(SHA256, file["sha256"]) and file["mode"] in ("100644", "100755"), "invalid watched input fingerprint")
    environment = artifact["environment"]
    require(isinstance(environment, dict) and set(environment) == {"tag", "platform", "machine", "executable_sha256"} and
            all(matches(r"[A-Za-z0-9_.-]{1,80}", environment[key]) for key in ("tag", "platform", "machine")) and
            matches(SHA256, environment["executable_sha256"]), "invalid environment fingerprint")
    result = artifact["result"]
    require(isinstance(result, dict) and set(result) == {"exit_code", "elapsed_ms", "stdout_sha256", "stderr_sha256"} and
            type(result["exit_code"]) is int and result["exit_code"] == 0 and
            type(result["elapsed_ms"]) is int and result["elapsed_ms"] >= 0 and
            all(matches(SHA256, result[key]) for key in ("stdout_sha256", "stderr_sha256")), "invalid verification result")
    return artifact


def capture(args):
    require(valid_argv(args.argv), "supply public verification argv with relative paths")
    require(0 < len(args.inputs) <= 16 and len(set(args.inputs)) == len(args.inputs), "supply up to 16 distinct watched inputs")
    for name in args.inputs:
        relative(name)
    cwd = relative(args.cwd, root_ok=True)
    commit = head(args.repo, args.repository)
    inputs = committed_inputs(args.repo, commit, args.inputs)
    require(checkout_matches(args.repo, inputs), "watched inputs differ from the source commit")
    executable, environment = fingerprint(args.argv[0], args.environment)
    with directory(args.repo / cwd), directory(args.artifact.parent) as parent:
        with suppress(FileNotFoundError):
            os.stat(args.artifact.name, dir_fd=parent, follow_symlinks=False)
            raise ValueError("artifact already exists")
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            start = time.monotonic()
            result = subprocess.run(args.argv, executable=executable, cwd=args.repo / cwd, env=process_env(),
                                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=600, check=False)
            elapsed = round((time.monotonic() - start) * 1000)
            require(result.returncode == 0, f"verification exited {result.returncode}")
            stdout.seek(0)
            stderr.seek(0)
            evidence = {"exit_code": 0, "elapsed_ms": elapsed, "stdout_sha256": hashlib.file_digest(stdout, "sha256").hexdigest(),
                        "stderr_sha256": hashlib.file_digest(stderr, "sha256").hexdigest()}
        require(head(args.repo, args.repository) == commit and checkout_matches(args.repo, inputs) and
                fingerprint(args.argv[0], args.environment)[1] == environment, "verification inputs changed while running")
        data = encoded({"schema": 1, "repository": args.repository, "source_commit": commit, "inputs": inputs,
                        "argv": args.argv, "cwd": args.cwd, "environment": environment, "result": evidence})
        require(len(data) <= MAX_ARTIFACT, "recipe artifact exceeds the size limit")
        create_at(parent, args.artifact.name, data)
    return digest(data)


def current(args, artifact, commit):
    if args.environment != artifact["environment"]["tag"]:
        return False
    source = artifact["source_commit"]
    paths = list(artifact["inputs"])
    try:
        if git(args.repo, "merge-base", "--is-ancestor", source, commit, missing_ok=True) is None:
            return False
        with directory(args.repo / relative(artifact["cwd"], root_ok=True)):
            return (committed_inputs(args.repo, source, paths) == artifact["inputs"] ==
                    committed_inputs(args.repo, commit, paths) and checkout_matches(args.repo, artifact["inputs"]) and
                    fingerprint(artifact["argv"][0], args.environment)[1] == artifact["environment"])
    except (FileNotFoundError, ValueError):
        return False


def apply_memory(args, artifact, fresh):
    command = json.dumps(artifact["argv"])
    text = (f"{DESCRIPTION}\n\nRun the verified argv `{command}` from `{artifact['cwd']}`.\n"
            f"Captured at commit `{artifact['source_commit']}` in environment `{artifact['environment']['tag']}`.\n"
            f"Recipe artifact SHA-256 is `{args.sha256}`.\n").encode()
    require(len(text) <= MAX_MEMORY, "staged recipe exceeds the size limit")
    path = args.repo / MEMORY
    try:
        with directory(path.parent, create=fresh) as parent:
            fcntl.flock(parent, fcntl.LOCK_EX)
            require(not git(args.repo, "ls-files", "-z", "--", str(MEMORY)) and
                    not git(args.repo, "ls-tree", "--name-only", "HEAD", "--", str(MEMORY)),
                    "memory path is tracked")
            try:
                existing, info = read_at(parent, path.name, MAX_MEMORY)
            except FileNotFoundError:
                if fresh:
                    create_at(parent, path.name, text)
                    return "staged"
                return "absent" if args.action == "evict" else "stale"
            require(existing == text, "memory path belongs to the user or a different artifact")
            if args.action == "evict":
                found = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                require(os.path.samestat(info, found) and info.st_ctime_ns == found.st_ctime_ns,
                        "memory path changed during eviction")
                os.unlink(path.name, dir_fd=parent)
                return "evicted"
            require(fresh, "staged recipe is stale; explicitly evict it before starting a session")
            return "staged"
    except FileNotFoundError:
        return "absent" if args.action == "evict" else "stale"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--repository", required=True, help="canonical public HTTPS repository URL without .git")
    parser.add_argument("--artifact", type=Path, required=True)
    actions = parser.add_subparsers(dest="action", required=True)
    record = actions.add_parser("capture")
    record.add_argument("--environment", required=True)
    record.add_argument("--cwd", default=".")
    record.add_argument("--input", dest="inputs", action="append", required=True)
    record.add_argument("argv", nargs=argparse.REMAINDER)
    stage = actions.add_parser("stage")
    stage.add_argument("--environment", required=True)
    stage.add_argument("--sha256", required=True)
    evict = actions.add_parser("evict")
    evict.add_argument("--sha256", required=True)
    args = parser.parse_args()
    try:
        args.repo = args.repo.absolute()
        require(matches(REPOSITORY, args.repository) and not args.repository.endswith(".git") and
                not any(part in {"", ".", ".."} for part in args.repository.removeprefix("https://").split("/")),
                "pass a canonical public HTTPS repository URL")
        if args.action != "evict":
            require(matches(r"[A-Za-z0-9_.-]{1,80}", args.environment), "pass a public environment identifier")
        if args.action == "capture":
            if args.argv[:1] == ["--"]:
                args.argv = args.argv[1:]
            print(capture(args))
        else:
            require(matches(SHA256, args.sha256), "pass the trusted artifact SHA-256")
            artifact = artifact_at(args.artifact, args.sha256)
            require(artifact["repository"] == args.repository, "artifact belongs to a different repository")
            commit = head(args.repo, args.repository)
            fresh = args.action == "stage" and current(args, artifact, commit)
            print(apply_memory(args, artifact, fresh))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"repo memory: {error}\n")


if __name__ == "__main__":
    main()
