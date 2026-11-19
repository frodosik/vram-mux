# gpu-arbiter

A reverse proxy in front of **one GPU shared by several engine processes** (Strata and llama.cpp server/router). It owns model
loads — admission, a free-VRAM gate, a drain queue — saves conversations to disk, and restores one from disk when a request
needs a conversation that is no longer in RAM. Clients change nothing: point them at the proxy.

Strict mode is on: a request must name a model the arbiter serves. A model it does not serve, or no model at all, is answered
404 — never by the route that happens to hold the card. Check `model=` next to `route=` in `/requests`.

## Quick start

Go 1.26 or newer. Each engine must have session files turned on (Strata: `"slot_save_path"` / `--slot-save-path DIR`,
`parallel` at 1; llama.cpp: a router with `--slot-save-path DIR`, `--models-preset presets.ini`, `--models-max 1` and `--no-models-autoload`).

```sh
go install github.com/frodosik/gpu-arbiter/cmd/gpu-arbiter@latest

# Engines first. The llama route is a router: presets loaded on demand, slots
# persisted, nothing preloaded (a preloaded model takes the card at once).
llama-server --models-preset /etc/gpu-arbiter/presets.ini --models-max 1 \
  --no-models-autoload --slot-save-path /var/lib/gpu-arbiter/sessions \
  --host 127.0.0.1 --port 8001 & # no api key: the route's api-key field is empty
strata --serve --host 127.0.0.1 --port 8095 \
  --native /srv/models/flash-next-iq2_xs.gguf \
  --slot-save-path /var/lib/gpu-arbiter/sessions --parallel 1 &

# The arbiter: Strata route from --upstream + --strata-model (the resident),
# each --route adds one more. A llama-only box omits both.
gpu-arbiter \
  --upstream http://127.0.0.1:8095 --strata-model qwen3.8-flash-next-iq2_xs \
  --route "llama|llama/27b|qwen3.8-27b|http://127.0.0.1:8001||18000" \
  --session-dir /var/lib/gpu-arbiter/sessions
```

Clients use `http://<host>:8080` the way they used the engine. Without `--session-dir` it is a plain pass-through proxy.
Run it as the user the engines run as, so it can delete the files they write. `examples/` carries runnable setups and
`gpu-arbiter.service` is an example systemd unit; `examples/bench.sh` reproduces the rows below.

## Results

One RDNA3 card (24 GiB VRAM, 30 GiB RAM), Strata v0.1.61 with resident experts, binary `d30a76b`, time to first token as the
client sees it:

| scenario | tokens | reused + read | time |
| --- | ---: | --- | ---: |
| fork launch, restore from disk | 8,062 | 8,057 + 5 | **84 ms** |
| grown parent return, restore from disk | 143,325 | 143,277 + 48 | **389 ms** |
| 67k conversation return, restore from disk | 67,760 | 67,690 + 70 | 630 ms |
| fresh 8k launch (nothing cached) | 8,055 | 0 + 8,055 | 7.1 s |
| fresh 20k launch (nothing cached) | 20,248 | 0 + full | ~19.7 s |
| yield save at a switch (143k file) | 143,277 | 1,918 MB written | 704 ms |
| yield save at a switch (67k file) | 67,690 | 1,025 MB written | 451 ms |

Engine decode 90–102 tok/s, prefill 1,150–1,330 tok/s. A route handoff (drain + gate + load) costs ~4 s typical, 8–24 s
worst. The point of the table: a disk restore replaces a 7–20 s re-prefill with 84 ms–400 ms.

## How it works

**Keying.** Each request is hashed as a chain: the header link (path, system prompt, tool schemas, reasoning settings)
hashed **together with the first message**, then one key per message. A later turn repeats the earlier messages, so its
chain starts with the earlier turn's chain — that is how the proxy recognises a conversation without a session id. There is
no header-only key: sharing the system prompt alone shares nothing. A wrong guess costs a wasted restore, never a wrong
answer.

**Files.** A file is keyed by the **last key of the conversation that saved it**, named
`dc-<ns>-<fingerprint>-<key>-<tokens>.session`. `Covered` — "this is already on disk" — checks only the last key. The
prefix index never aliases forward to a longer descendant's file. The fingerprint covers the engine's model, version,
context, KV type and control vector; when any change, the old files are deleted, since the engine rejects them anyway.
Files unused for `--retention-days` are deleted; the total is capped at `--max-gib`, least recently used first; before a
save the store evicts until the file fits under the cap and leaves `--min-free-mib` free, and skips the save if it doesn't.
The store's counts are the folder: a key keeps one file, so a save of the same last chain key deletes the older file of
that key, and the cap counts the turn-boundary files the prefix index holds — `gpu_arbiter_files` and
`gpu_arbiter_bytes` cover every session file the store owns, not the last-key files alone.

**Restoring.** A request that continues the conversation the engine is holding goes straight through. On a switch, if the
conversation is not parked in RAM and a file exists for it (or for an earlier turn, bound by the prompt estimate), the proxy
restores that file first — the engine reads only what came after it.

**Saving.** The engine can only save the conversation it holds live. The proxy saves:
- **idle**: `--idle-save-s` quiet seconds after a turn, in the background, when the conversation is at least
  `--min-tokens`, its file is missing, and it grew more than `--regrow-tokens` past it. `--max-idle-save-tokens` bounds this:
  a 100k idle save writes 1.4 GB and holds the save lock ~110 s, so a queued request waits for it; the yield save at a
  switch writes the same file in ~4 s.
- **at a switch / yield**: when the conversation would be lost — a disk restore is about to replace it, the engine would
  refuse to park it, or the arbiter is giving the card to another route.
- Turn-1 boundary files are admitted at `--min-tokens/4`, so a 1–2-message spawn can be restored later.

**Admission counts.** A save's size comes from **the engine's own count** — `cache + prompt + predicted` tokens for the
conversation it just served — not the route's stored count, which describes the conversation served before. An engine that
reports nothing gets the `--body-divisor` estimate (request bytes per token). Bytes-per-token is learned only from saves of
at least `--min-tokens` and clamped to 4 KB–256 KB/token, so a small conversation's file never poisons the next reserve.

**Arbiter.** One card, many routes. A request for a non-resident route is queued; the resident route drains up to
`--max-resident-streak` requests, then the card yields: the held conversation is saved, the engine is unloaded, the free-VRAM
gate checks `--vram-margin-mib` plus the new route's `--strata-need-mib` floor, and the new model is loaded within
`--load-timeout-s`. Loads are owned by the arbiter: it never wakes an engine by sending it a request.

**Fail-open.** A failed save/restore, or a panic in the proxy's own logic, passes the request to the engine untouched. A
file the engine rejects is deleted; a file it couldn't load for lack of RAM is kept for next time.

[docs/design.md](docs/design.md) lists the edge cases (concurrent requests, disconnects, restarts, corrupt files, full
disks), what the proxy does in each, and the test that covers it.

## Configuration

Every flag also has its UPPERCASE env var; the flag wins.

| flag | env | default | |
| --- | --- | --- | --- |
| `--listen` | `LISTEN` | `:8080` | address clients connect to |
| `--upstream` | `UPSTREAM` | `http://127.0.0.1:8081` | Strata's URL (single-route mode) |
| `--metrics-listen` | `METRICS_LISTEN` | `127.0.0.1:8082` | `/metrics`, `/requests`, `/health`; keep it private |
| `--session-dir` | `SESSION_DIR` | (empty) | the engine's `slot_save_path`; empty = pass-through only |
| `--max-gib` | `MAX_GIB` | `40` | total size cap for the files |
| `--retention-days` | `RETENTION_DAYS` | `7` | delete files unused for this long |
| `--min-free-mib` | `MIN_FREE_MIB` | `4096` | free space to keep on the session disk |
| `--min-tokens` | `MIN_TOKENS` | `15000` | don't save shorter conversations |
| `--park-min-tokens` | `PARK_MIN_TOKENS` | `-1` | engine's `--conversation-cache-min-tokens`; `-1` reads it from Strata |
| `--regrow-tokens` | `REGROW_TOKENS` | `4096` | save again once a conversation grew this much past its file |
| `--idle-save-s` | `IDLE_SAVE_S` | `3` | quiet seconds after a turn; `0` = save only when a switch would lose it |
| `--max-idle-save-tokens` | `MAX_IDLE_SAVE_TOKENS` | `32768` | skip a background idle save of a longer conversation; `0` = no cap |
| `--snapshot-bytes-per-token` | `SNAPSHOT_BYTES_PER_TOKEN` | `65536` | RAM per token of a parked conversation, to predict park refusals |
| `--max-body-mib` | `MAX_BODY_MIB` | `64` | larger bodies stream through without caching |
| `--shutdown-s` | `SHUTDOWN_S` | `20` | on SIGTERM, grace for running requests and a save under way |
| `--strata-model` | `STRATA_MODEL` | (empty) | the model the Strata route answers for; required with a Strata route |
| `--strata-api-key` | `STRATA_API_KEY` | (empty) | bearer token, plain or `file:PATH` so it never appears in `ps` |
| `--strata-need-mib` | `STRATA_NEED_MIB` | `0` | VRAM the Strata route needs to load: its floor, not what it fills; `0` = gate on margin only |
| `--route` | `ROUTES` | (empty) | one per occurrence: `dialect\|id\|model\|base\|api-key\|need-mib`; dialects `strata` and `llama`. `ROUTES` is space-separated, so one systemd `Environment=` line carries several routes (a comma inside a spec is one value) |
| `--max-resident-streak` | `MAX_RESIDENT_STREAK` | `32` | resident requests drained before a waiter takes the card; `0` = unbounded |
| `--vram-margin-mib` | `VRAM_MARGIN_MIB` | `1024` | free VRAM that must remain before a route loads |
| `--drain-wait-s` | `DRAIN_WAIT_S` | `300` | how long a yield waits for a running turn |
| `--load-timeout-s` | `LOAD_TIMEOUT_S` | `600` | the whole swap: drain, gate, load |
| `--vram-sysfs` | `VRAM_SYSFS` | `/sys/class/drm/card0/device` | amdgpu device node; auto-detects the card with most VRAM when left default |
| `--gpu-index` | `GPU_INDEX` | `-1` | pin a card for the nvidia-smi backend; `-1` = largest |
| `--vram-poll-s` | `VRAM_POLL_S` | `2` | how often the gate re-reads the card |
| `--body-divisor` | `BODY_DIVISOR` | `4` | request bytes per token for engines that report no count; `0` = never estimate |
| `--log-output` | `LOG_OUTPUT` | `stderr` | `stderr`, `file` or `both` |
| `--log-file` | `LOG_FILE` | `/var/log/gpu-arbiter/gpu-arbiter.log` | the log file |
| `--log-format` | `LOG_FORMAT` | `text` | `text` or `json` |
| `--log-level` | `LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error` |
| `--log-max-size-mib` | `LOG_MAX_SIZE_MIB` | `100` | rotate the log at this size |
| `--log-max-backups` | `LOG_MAX_BACKUPS` | `10` | rotated files to keep (`0` = all) |
| `--log-max-age-days` | `LOG_MAX_AGE_DAYS` | `30` | delete rotated files older than this (`0` = never) |
| `--log-compress` | `LOG_COMPRESS` | `true` | gzip rotated files (`--no-log-compress` turns off) |

## Observability

On `--metrics-listen`, keep it private:

- `/metrics` — Prometheus text: `gpu_arbiter_requests_total{action}` (`continue`, `ram`, `disk`, `miss`, `busy`,
  `unkeyed`, `pass`, `error`, `refused`, `denied`), `gpu_arbiter_saves_total`,
  `gpu_arbiter_saves_by_trigger_total{trigger=idle|switch|yield}`, `gpu_arbiter_saves_skipped_total{reason}`,
  `gpu_arbiter_restores_total`, bytes counters and the save/restore histograms, evictions,
  `gpu_arbiter_resident_info{route,model}` (1 for the route holding the card),
  `gpu_arbiter_swaps_total`, `gpu_arbiter_swap_seconds`, `gpu_arbiter_swap_blocked_total{reason}`,
  `gpu_arbiter_reloads_total{route}`, `gpu_arbiter_queue_length`.
- `/requests` — the last 256 generation requests: client, User-Agent, model named, route that answered, action
  (`miss`/`continue`/`disk`/`refused`) + reason, tokens, prepare ms, whole ms. A refused request is listed with an empty route.
- `/health` — the resident's health.

Logs are structured (`slog`), one `request` line per generation request; saves, restores, skipped saves and evictions get a
line each. Prompt and answer text is never logged.

## Limitations

- **One card.** Multi-GPU is not implemented: the arbiter owns one card, all routes share it.
- **VRAM gate:** amdgpu sysfs (`mem_info_vram_total` / `mem_info_vram_used`), with a nvidia-smi fallback; NVIDIA needs `nvidia-smi` installed and `--gpu-index` to pin a card. Strata + llama-server runs on NVIDIA CUDA the same way as on AMD.
Pinned to Strata v0.1.45–v0.1.61 slot APIs: `POST /slots/0?action=save|restore`, `parallel` at 1, `slot_save_path`, `/v1/status`
  and the JSON `/metrics`. A later engine change makes the proxy fail open into a pass-through.
- A route that reports no token count (llama.cpp) uses the `--body-divisor` estimate for its saves.
- There is no header-only chain key, so summarizer-style prompts that share only the system prompt get no reuse.
- All traffic must go through the proxy; direct requests to an engine make its live view wrong, costing speed until a
  request runs alone. A conversation evicted from RAM before its first quiet moment has no file.

## Development

```sh
make test    # gofmt check, go vet, go test -race
make lint    # golangci-lint
make build   # ./gpu-arbiter
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Provenance and Acknowledgements

`gpu-arbiter` is the evolution of [tfyl/strata-diskcache](https://github.com/tfyl/strata-diskcache) (MIT, © 2026 tfyl),
the single-engine disk cache this project grew from. Fork point: upstream `709faf6` ("feat: initial release"). The upstream
remains the home of single-Strata deployments; `examples/single-strata/` is that setup, and the store, keying, save/restore
policy and fail-open behavior are its design, carried forward with the multi-route arbiter, strict mode, the free-VRAM gate,
the prefix index, and the engine-count admission fixes.

## License

MIT, like Strata. See [LICENSE](LICENSE).
