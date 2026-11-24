# Native execution and privacy

The built-in file tools (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`) are confined to the project root by the harness.

`write_file` and `edit_file` write the new contents to a temp file beside the target and rename it into place, so a write that fails (a full disk, a quota) leaves the file as it was.

- The temp file that replaces a file is created with mode `0600`, readable and writable only by its creator, until it has the target's owner, group and permissions.
- The temp file takes the target's permissions (less setuid and setgid), owner, group and access control list. It does not take the directory's default list, or the target's other extended attributes or file flags.
- The temp file needs room for the whole new file beside the old one, so on a full disk even a write that would shrink a file is refused, and the file is left unchanged.
- A replacement is a new file under the old name, so whatever holds the old file keeps it. A mount of the file itself (a container's single-file bind mount) and another process's open descriptor or inode watch go on seeing the old contents, so mount or watch the directory instead.

A file is written in place instead, and its old contents are written back if that write fails, when:

- it has other hard links;
- Open Max cannot give a new file its owner or group (only the superuser can give a file to another user or to a group it is not in);
- its access control list cannot be given to a new file (on macOS, where Open Max does not copy one, a file that has a list, or whose directory's inheritable entries would give the new file one);
- its directory takes no new file;
- the rename over it is refused (a file mounted on its own);
- or its path leaves no room for the longer path of a temp file beside it.

Such a file whose contents cannot be read first is not written at all, since there would be nothing to write back, and the write reports it unchanged. In a directory flagged append-only, which refuses both the rename and the removal of the temp file, the file is written in place and the temp file, holding the new contents, is left beside it.

One `edit_file` call carries a list of replacements in `edits` (a top-level `old_string` and `new_string` are still read, before the list): each is matched against the file as it was before the call, no two may overlap, and the file is written only when every one matches, so a failed call names the replacement that failed and changes nothing. The result shows the changed lines in `read_file`'s numbered form with two lines of context, at most 2,000 bytes, so an edit needs no read-back.

`bash`, external TOML tools, and hooks are native processes. The harness does not confine them to the project root. They inherit the host filesystem, environment, credentials, and network access of Open Max, except that an external tool's environment is a scrubbed baseline (`PATH`, `HOME`, `LANG`, `TERM`) plus the variables its manifest's `env` names. They run without a controlling terminal, so a command that prompts on the terminal (a password, an ssh host key) fails at once instead of drawing over the TUI and waiting out its timeout. `bash` and hooks also get `GIT_TERMINAL_PROMPT=0`, so git names the cause.

When unset in Open Max's environment, these commands get `EMAIL=openmax@localhost` as Git's fallback email and `CARGO_CACHE_AUTO_CLEAN_FREQUENCY=never` to keep Cargo from deleting cached downloads during a build. Git still uses any configured identity and still needs a configured name if the OS account has no display name. An explicit value is left alone, including an empty value. External tools only inherit explicit values if their manifests name the variables. Sandboxed example runs do not get these defaults.

`bash` runs each command with `bash -c` (`/bin/sh -c` on a host with no bash at a fixed path), but the command text is not that argument: it reaches the shell in the `OPENMAX_COMMAND` environment variable, and the shell runs the fixed script `eval "unset OPENMAX_COMMAND; $OPENMAX_COMMAND"`. So the shell's argument list never holds the text, and since `eval` unsets the variable before the command's first word, no process the command starts inherits it, whether or not bash starts with `allexport` on (as an exported `SHELLOPTS=allexport` or a `BASH_ENV` file that runs `set -a` arranges). A `BASH_ENV` file runs before that `unset`, so it, and anything it starts, can still read the text there. `OPENMAX_COMMAND` is a name the harness reserves for `bash` calls: a value exported in the harness's own environment is overwritten for the call and unset before the command runs. A `pkill -f` or `pgrep -f` pattern that occurs only in the command text therefore matches the processes it names and not the shell running it. `$0` is the shell's path and there are no positional parameters, as under a plain `bash -c`.

Running the text through `eval` shows in a few places. Bash's `set -x` trace lines start with `++` instead of `+`. Bash's `$_` is `OPENMAX_COMMAND` when the command starts, and once the command text has run, as an `EXIT` trap sees it, `$_` holds `unset OPENMAX_COMMAND; ` and the text; bash gives each child its own `_`, so none inherits it. Bash's `BASH_EXECUTION_STRING` holds the fixed script, not the text. A syntax error names `eval` where it named `-c` (bash's `-c: line N:` becomes `eval: line N:`, and dash adds `eval: ` after its line number), and a first line that bash quotes back in the error starts with `unset OPENMAX_COMMAND; `. Bash 3.2 (the macOS `/bin/bash`) exits 1 for a syntax error where `bash -c` exited 2; bash 5.3 and dash exit 2 either way. A command killed by a signal can be reported as the shell's exit code, 128 plus the signal's number (139 for a segfault), instead of as the signal: under `-c`, bash and dash exec'd a lone simple command in the shell's place, and dash and bash 5.1 and later also the last command of a list, but none of them does inside `eval`. Bash's or dash's own stderr line names most signals (not SIGINT or SIGPIPE).

Each `bash` or external tool call runs in its own process group, and processes it leaves running in the background are terminated when it returns. Outside a sandbox the call also leads a session of its own, and on Linux every stop reaches the session's other process groups too, such as the one GNU `timeout` moves its command to; a process that starts a session of its own with `setsid` is out of reach and keeps running. The first call in a session that terminated some gets a result that explains why and how to keep one running (a named tmux session); every later one gets one line that says so on its own, which still holds if the model never saw the first (its turn was cancelled) or compaction dropped it. A session loaded again from its file, as a resume does, or one whose extensions were refrozen, explains once more.

## Signals and cancellation

Tools run in a session of their own, so Ctrl+C or Ctrl+\ at the shell reaches
openmax and not them. SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGUSR1, or SIGUSR2 to
a print or stdio run cancels the running turn, which stops each tool's process
group (SIGTERM, then SIGKILL), and exits 128 plus the signal number (130, 143,
129, 131; for SIGUSR1 and SIGUSR2, 158 and 159 on macOS and 138 and 140 on
Linux) once the turn has ended, waiting at most 3 seconds for it; a `turn_end`
hook still running then is killed. A SIGQUIT therefore writes no core dump.

Removing a new session that nothing was saved to takes its share of the same 3
seconds: while another openmax process holds the session index past them, the
session stays indexed and the run says so on stderr. Once they are used up,
the run still takes up to 100 ms to check that the session is empty before it
says so; a check that takes longer is reported as a session that may be empty.
A signal while the run waits on that index to open its session or to start a
turn ends it the same way.

A second signal exits without waiting for the turn, with 128 plus its own
number, and kills the tools outright. Either exit first waits up to 250 ms for
a tool command still being started, so one that starts within that wait is
killed with the rest.

Neither stops a process that leaves its tool's session (as `setsid` does), on
macOS one that leaves its tool's group, or a tool command whose start outlasts
that wait: it runs on after openmax has exited. On Linux a process that only
moves to another group of its tool's session, as GNU `timeout` moves its
command, is stopped with the rest.

While a consumer that stopped reading holds stdout full, the run cannot act on
the first signal, and only a second one ends it. A signal the run inherited as
ignored, as under `nohup`, stays ignored.

The TUI, where Ctrl+C is a key, ends the session on one of these signals as
`/quit` does, then exits with the same status, unless its terminal has already
gone away. Until the session's loop starts (while its first frame waits on
trust, settings, the prompt history or the session `--continue` resumes, or
the trust question waits on the shell), the first signal ends the run at once
with that status, and Ctrl+C at the terminal is SIGINT.

## Watcher and cleanup

SIGKILL, or a crash that runs no handler, runs none of openmax's code, so its
watcher stops its tools instead: each `bash` call, external tool and hook it
started, with every process each started that is still in its group. A
sub-agent's own git commands are not tools, and run on
([sub-agents](usage.md#sub-agents)).

On macOS and Linux, once a run's [tool group record](#process-records) is
first written, the run starts `openmax --lifeline <record> <start>`: its own
binary again (on Linux `/proc/self/exe`, the binary it runs; on macOS the file
at its path), in a session of its own, whose stdin is the read end of a pipe
the run alone holds the write end of (close-on-exec, so no tool inherits it).
Its stdout and stderr are `/dev/null`, its environment is empty and its
directory is `/`. The watcher gets an absolute record path. If the data
directory is relative, Open Max resolves it from the run's working
directory. This can happen with a relative `HOME`. An unset `HOME` uses the
system's home directory lookup, and an empty `HOME` refuses to start a run
that needs state (see [Configuration](configuration.md)). Like every tool,
the watcher inherits each descriptor the run holds without close-on-exec,
such as one the run was itself started with open across exec.

Its start holds the lock that every tool and hook spawn takes, so once in a
run, a tool or hook that starts in that moment waits for the watcher's fork
and exec.

Once the run has exited, by any cause, the pipe reads end of file. The watcher
waits up to 250 ms for the run to read as exited, then handles the
[`run` directory](#process-records) like a launch. It sends SIGTERM to each
still-recorded group of an exited run, including the run it watched, then
SIGKILL if it is still there 50 ms later, so that each group is gone within
100 ms of the kill. It removes the records, removes the directory if no
record is left, and exits.

Its polls, for the run's exit and through the grace, wait on a timer that a
QoS clamp the run was started under (`taskpolicy -c`, a background launchd
job) does not stretch: on macOS a kevent timer marked critical, where a plain
2 ms sleep took a mean 11 ms under a utility clamp and 98 ms under background.

After a run that stopped its own tools, it finds no record of the run's and
exits at once. A record whose owner still runs, such as one that a process
which took the run's pid since wrote, is left alone. It reads no settings,
trust or permissions and opens no connection.

A watcher that is itself killed, or a run killed in the moment between its
record's first write and its watcher's start, leaves the tools to the next
openmax launch (the TUI, `-p`, `--stdio`, `--subagent`, or `--check`), as does
a run whose watcher cannot start.

On Linux the watcher starts even once an upgrade has replaced or removed the
run's binary, whose path then reads `<path> (deleted)`. On macOS none starts
once the file was removed before the run's first tool, and once it was
replaced, the binary that replaced it starts, and watches only if it serves
`--lifeline`.

`zero_orphans_after_kill_9` in `crates/tui/tests/thesis.rs` gates this on
Linux and macOS: a print run's process group gets SIGKILL in the middle of a
`bash` call whose shell, child and grandchild all ignore SIGTERM, and with no
launch since, each of the three, and the watcher, which leads a session of its
own, is gone, exited and collected,
within 100 ms of the signal, and none of the three before the 50 ms grace.
CI's thesis job asserts that bound in a release build on both runners and
keeps each pid and when it was gone as an artifact.

`zero_orphans_after_kill_9_with_relative_home` holds the same for `kill -9
<pid>` of a run with a relative `HOME`,
`zero_orphans_after_kill_9_once_its_binary_is_replaced` holds it on Linux for
a run whose binary an upgrade replaced, and `a_runs_watcher_ends_with_the_run`
holds the watcher to a run that exits on its own,
which it outlives by less than 50 ms in a release build.

A watcher killed too leaves the groups to the next launch of the TUI, `-p`,
`--stdio`, `--subagent`, or `--check` (except `--check --stdio`), which stops
them the same way before it starts a session, so before any tool of its own: a
launch refused for a settings flag, which starts no session, stops them too.

## Process records

On macOS and Linux, openmax records the process groups of its running tools,
each leader with its start time, in `~/.openmax/run/<pid>.groups`: one line
when a group starts and one when it ends. Once the file has passed 4 KiB, it
is cut back to its first line the next time a group ends with no other
running. An exit that leaves no tool running removes it, and the `run`
directory once no other record is left there.

A launch reads every record whose owner pid no process of the same user holds
(none does, or another user's does), or names a process that has exited or
started at another time, and sends each group the record still lists whose
leader still has its recorded start time SIGTERM, then SIGKILL if the group is
still there 50 ms later, before it starts anything. Neither signal goes to a
group whose leader pid has since been taken by a process started at another
time, and none goes to a group once it has been seen gone, since its id can
then name another. It then removes those records, and the `run` directory once
no record is left there.

On Linux, where start times count from boot, a record's first line also
carries the boot's id, and a record from an earlier boot is removed with
nothing signalled. A record a running openmax owns is left alone. A killed
openmax holds its pid until its parent waits for it, and a launch stops its
tools before that wait too.

A launch never signals the group it runs in: an openmax that a left-behind
tool's script starts leaves that tool's group running and, while the group's
leader runs, its record in place, so the next launch outside that group stops
it.

A group is recorded just after its spawn has returned, when its command has
already started, so a SIGKILL in that moment leaves it unrecorded, and a home
where the record cannot be written (read-only, or on a full disk) records
nothing while it stays so.

No tool and no exit waits on the record's file, as a home whose file server
stopped answering would make them: one thread, started by the first tool or
hook a launch runs, makes every write to it and starts the watcher, and an
exit waits at most 100 ms for that thread to remove it. The watcher, or
failing that the next launch, removes a record so left.

A sub-agent openmax among the tools that a stop kills before it has stopped
its own tools leaves them to its own watcher. A group whose leader had exited
and been waited for before the watch or the launch, and a process that left
its tool's group, are not stopped.

Nor are the tools of an openmax in another pid namespace that shares the home
directory, as a container with the home mounted does: a launch or a watcher
outside it takes its records for a dead openmax's and removes them, and its
own watcher then finds none.

## Permissions and privacy

Permissions, approvals, and `mutating` metadata control dispatch and user experience, not operating-system isolation.

Open Max itself has no telemetry. Apart from native child processes, the harness contacts your configured model endpoint, for an explicitly connected ChatGPT account, OpenAI's authentication service for sign-in, renewal, and sign-out, and, at most once a day from an interactive session of an installed release, `api.github.com` for the latest release's version, which `OPENMAX_NO_UPDATE_CHECK=1` turns off (see [updating](install.md#updating)).

Sessions, settings, tools, and skills stay under `~/.openmax/` and your project directory. `/status` lists the destinations configured by the harness and detailed runtime information; it does not enforce or enumerate child-process network access, and external tools you install may open their own network connections.
