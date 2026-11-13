# Security

## Supported versions

Only the latest release gets security fixes.

## Reporting a vulnerability

Report it privately: open the repository's Security tab and choose "Report a vulnerability". Do not open a public issue for it.

Include:

- the version, from `openmax --version`;
- your platform: operating system and architecture;
- the steps that reproduce it;
- what you expected to happen, and what happened instead.

## What counts

`bash`, external TOML tools and hooks are native processes. They run with your user's access, and the harness does not confine them to the project root. Permissions, approvals and `mutating` metadata decide whether Open Max dispatches a tool call; they are not operating-system isolation. So a model running a command you allowed, under the approval mode and permission rules you chose, is not a vulnerability.

These are:

- a way around the trust, approval or permission checks, such as an agent turn in a project that was never trusted, or a tool call that runs when the approval mode or a permission rule should have stopped it or asked first;
- a built-in file tool (`list_dir`, `read_file`, `write_file`, `edit_file`, `glob` or `grep`) reaching a path outside the project root;
- a credential sent to a server it was not configured for, such as a request and its key following a redirect to a server other than the one `base_url` names;
- a release archive that does not match the `sha256.sum` published with its release.
