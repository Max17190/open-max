# JSON-RPC bridge

`openmax-bridge` is a separate binary that exposes a running session over
line-delimited JSON-RPC 2.0 on stdio, so a host that speaks JSON-RPC can drive
openmax without handling its native JSONL envelope. It is a client of the
[`openmax-stdio/1`](stdio-protocol.md) contract: it spawns `openmax --stdio`
(or `$OPENMAX_BIN`) and translates both directions. The harness core is
unchanged.

## Host methods

A request carries `id`; a notification omits it.

| Method | Params | Result and behavior |
| --- | --- | --- |
| `initialize` | none | `{ server, protocol_version, version, session_id }` |
| `prompt` | `text` | Streams `update` notifications, then resolves with `{ stop_reason }` at turn end |
| `cancel` | none | `{ ok: true }`, and cancels the running turn |
| `approve` | `approval_id`, `approved` | Answers a pending approval |
| `shutdown` | none | `{ ok: true }`, then the bridge drains and exits |

## Streaming

The bridge streams one `update` notification per child event,
`{"jsonrpc":"2.0","method":"update","params":<event>}`, where `params` is the
event exactly as the stdio protocol defines it (approval requests included;
answer them with `approve`). A `prompt` request stays open until the turn's
`done`, which resolves it. Closing stdin drains the in-flight turn, like the
underlying protocol.

```sh
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"prompt","params":{"text":"list the crates"}}' \
  | openmax-bridge
```
