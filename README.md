# vram-mux

**Multiplex multiple LLM engines (Strata + llama.cpp) on a single GPU with sub-400 ms NVMe KV-cache teleportation.**

**The problem.** A multi-agent coding rig on one 24 GB GPU — a fast MoE coder on Strata plus a lossless 27B
architect/auditor on `llama-server` — normally pays for every model switch and every subagent fork with a wiped KV cache:
**20 s–120 s+ of prompt re-prefill** at 100k–200k contexts. That is the tax that makes single-GPU agent setups feel slow.

**What vram-mux does.** A drop-in reverse proxy that multiplexes one GPU across engine processes: **resident-drain
scheduling** (kills model ping-pong), a **hardware free-VRAM gate** (AMD `amdgpu` sysfs + NVIDIA `nvidia-smi`), and
**prefix-indexed NVMe KV session save/restore** — turning a 2-minute re-prefill into an **84–389 ms** disk restore.
Clients change nothing: point them at the proxy. Strict mode is on: a request must name a model the mux serves, and a model
it does not serve gets 404 — never an answer from the route that happens to hold the card. Check `model=` next to `route=`
in `/requests`.

## Measured, on one card

One RX 7900 XTX (24 GiB VRAM, 30 GiB RAM): Strata v0.1.61 with resident experts, `llama-server` b12445 as a preset router,
time to first token as the client sees it. Engine decode 90–102 tok/s, prefill 1,150–1,330 tok/s, so "cold" is pure prompt
re-prefill — the last column is the same turn with the session file:

| event | tokens | cold, no cache | with `vram-mux` |
| --- | ---: | ---: | --- |
| parent agent returns after a model swap | 143,325 | ~120 s re-prefill | **389 ms** (143,277 reused + 48 read) |
| `subagent_fork` spawn off a turn boundary | 8,062 | 7.1 s cold (19.7 s at 20k) | **84 ms** (8,057 reused + 5 read) |
| session switch return to a 67k conversation | 67,760 | ~55 s re-prefill | **630 ms** (67,690 reused + 70 read) |
| yield save to NVMe before the swap | 143,277 | — | **704 ms** (1,918 MB written) |

A route handoff (drain + gate + load) costs ~4 s typical, 8–24 s worst. `examples/bench.sh` reproduces the rows on your box.

## The shape

```text
      coding harness · agents, subagents, one OpenAI endpoint
                        │
                        ▼
      ┌──────────── vram-mux  (:8080) ─────────────┐
      │  Resident-Drain queue   (:8080 requests)    │
      │  Free-VRAM gate         amdgpu / nvidia-smi │
      │  Prefix-indexed NVMe store  dc-*.session    │
      └───────┬───────────────────────────┬─────────┘
              │ drain, then swap          │ save / restore
              ▼                           ▼
      Strata :8095                 llama-server router :8001
      MoE coder, resident experts   presets, slots persisted
```

## Why not just `llama-swap`?

- **Multi-engine.** `llama-swap` swaps models inside one llama.cpp server. `vram-mux` multiplexes *separate engine
  processes* — Strata and `llama-server` in the same box — each route with its own base URL, api key, VRAM floor and
  dialect (`strata` and `llama`), and the arbiter owns every load: it never wakes an engine by sending it a request.
- **KV slots survive unloads, across forks.** Conversations are written to NVMe keyed by the hash chain of their messages,
  with a prefix index over turn boundaries, so a `subagent_fork`, a return to an older turn, or a return after a full model
  swap restores the engine's cache — 84–389 ms instead of a 7–120 s re-prefill.
- **It drains before it swaps.** The resident route keeps the card for `--max-resident-streak` requests and yields only to
  requests already queued, so an agent fleet runs on the resident model without ping-pong, and the swap happens once, at a
  quiet moment, with the conversation written first.

## Quick start

Go 1.26 or newer. Each engine must have session files turned on (Strata: `"slot_save_path"` / `--slot-save-path DIR`,
`parallel` at 1; llama.cpp: a router with `--slot-save-path DIR`, `--models-preset presets.ini`, `--models-max 1` and
`--no-models-autoload`).

```sh
go install github.com/frodosik/vram-mux/cmd/vram-mux@latest

# Engines first. The llama route is a router: presets loaded on demand, slots
# persisted, nothing preloaded (a preloaded model takes the card at once).
llama-server --models-preset /etc/vram-mux/presets.ini --models-max 1 \
  --no-models-autoload --slot-save-path /var/lib/vram-mux/sessions \
  --host 127.0.0.1 --port 8001 & # no api key: the route's api-key field is empty
strata --serve --host 127.0.0.1 --port 8095 \
  --native /srv/models/flash-next-iq2_xs.gguf \
  --slot-save-path /var/lib/vram-mux/sessions --parallel 1 &

# The mux: Strata route from --upstream + --strata-model (the resident),
# each --route adds one more. A llama-only box omits both.
vram-mux \
  --upstream http://127.0.0.1:8095 --strata-model qwen3.8-flash-next-iq2_xs \
  --route "llama|llama/27b|qwen3.8-27b|http://127.0.0.1:8001||18000" \
  --session-dir /var/lib/vram-mux/sessions
```

Clients use `http://<host>:8080` the way they used the engine. Without `--session-dir` it is a plain pass-through proxy.
Run it as the user the engines run as, so it can delete the files they write. `examples/` carries runnable setups and
`vram-mux.service` is an example systemd unit.

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
that key, and the cap counts the turn-boundary files the prefix index holds — `vram_mux_files` and `vram_mux_bytes` cover
every session file the store owns, not the last-key files alone.

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
| `--log-file` | `LOG_FILE` | `/var/log/vram-mux/vram-mux.log` | the log file |
| `--log-format` | `LOG_FORMAT` | `text` | `text` or `json` |
| `--log-level` | `LOG_LEVEL` | `info` | `debug`, `info`, `warn` or `error` |
| `--log-max-size-mib` | `LOG_MAX_SIZE_MIB` | `100` | rotate the log at this size |
| `--log-max-backups` | `LOG_MAX_BACKUPS` | `10` | rotated files to keep (`0` = all) |
| `--log-max-age-days` | `LOG_MAX_AGE_DAYS` | `30` | delete rotated files older than this (`0` = never) |
| `--log-compress` | `LOG_COMPRESS` | `true` | gzip rotated files (`--no-log-compress` turns off) |

## Observability

On `--metrics-listen`, keep it private:

- `/metrics` — Prometheus text, namespace `vram_mux_*`: `vram_mux_requests_total{action}` (`continue`, `ram`, `disk`,
  `miss`, `busy`, `unkeyed`, `pass`, `error`, `refused`, `denied`), `vram_mux_saves_total`,
  `vram_mux_saves_by_trigger_total{trigger=idle|switch|yield}`, `vram_mux_saves_skipped_total{reason}`,
  `vram_mux_restores_total`, `vram_mux_saved_bytes_total` / `vram_mux_restored_bytes_total`, the
  `vram_mux_save_seconds` / `vram_mux_restore_seconds` / `vram_mux_swap_seconds` histograms,
  `vram_mux_evictions_total{reason}` and `vram_mux_evicted_bytes_total{reason}`, `vram_mux_files` and
  `vram_mux_bytes` (the whole folder the store owns), `vram_mux_max_bytes`, `vram_mux_resident_info{route,model}`
  (1 for the route holding the card), `vram_mux_swaps_total`, `vram_mux_swap_blocked_total{reason}`,
  `vram_mux_reloads_total{route}`, `vram_mux_re_prefill_tokens_total`, `vram_mux_queue_length`.
- `/requests` — the last 256 generation requests: client, User-Agent, model named, route that answered, action
  (`miss`/`continue`/`disk`/`refused`) + reason, tokens, prepare ms, whole ms. A refused request is listed with an empty
  route. The action field is named `diskcache` in the JSON — the cache component's name, not the project's.
- `/health` — the resident's health.

Logs are structured (`slog`), one `request` line per generation request; saves, restores, skipped saves and evictions get a
line each. Prompt and answer text is never logged.

## Limitations

- **One card.** Multi-GPU is not implemented: the arbiter owns one card, all routes share it.
- **VRAM gate:** amdgpu sysfs (`mem_info_vram_total` / `mem_info_vram_used`), with a nvidia-smi fallback; NVIDIA needs
  `nvidia-smi` installed and `--gpu-index` to pin a card. Strata + llama-server runs on NVIDIA CUDA the same way as on AMD.
- Pinned to Strata v0.1.45–v0.1.61 slot APIs: `POST /slots/0?action=save|restore`, `parallel` at 1, `slot_save_path`,
  `/v1/status` and the JSON `/metrics`. A later engine change makes the proxy fail open into a pass-through.
- A route that reports no token count (llama.cpp) uses the `--body-divisor` estimate for its saves.
- There is no header-only chain key, so summarizer-style prompts that share only the system prompt get no reuse.
- All traffic must go through the proxy; direct requests to an engine make its live view wrong, costing speed until a
  request runs alone. A conversation evicted from RAM before its first quiet moment has no file.

## Development

```sh
make test    # gofmt check, go vet, go test -race
make lint    # golangci-lint
make build   # ./vram-mux
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project status and community

`vram-mux` is built for one daily-driver box: a single-GPU, multi-agent coding rig that runs a MoE coder and a 27B
architect on the same card, every day, for hours. It is shared MIT-licensed with the local-LLM community because that
setup is common and the fix is measured, not theoretical.

It is honest software, not a finished product: the pinned Strata APIs, the single-card assumption, the linux-first
systemd examples and the AMD-first gate are the edges you will find. Issues, forks and PRs are welcome — especially
reproductions of the measured table on your own hardware. If you want to co-maintain, or to add a backend (multi-GPU,
SGLang, vLLM, a second llama dialect), open an issue and reach out; the arbiter, the store and the dialect layer are
separate packages precisely so a new backend fits between them.

## Provenance and Acknowledgements

`vram-mux` is the evolution of [tfyl/strata-diskcache](https://github.com/tfyl/strata-diskcache) (MIT, © 2026 tfyl), the
single-engine disk cache this project grew from. Fork point: upstream `709faf6` ("feat: initial release"). The upstream
remains the home of single-Strata deployments; `examples/single-strata/` is that setup, and the store, keying,
save/restore policy and fail-open behavior are its design, carried forward with the multi-route arbiter, strict mode, the
free-VRAM gate, the prefix index, and the engine-count admission fixes. The keying salt is the upstream's
(`"strata-diskcache v1\0"`) on purpose: files written by older versions keep their keys.

## License

MIT, like Strata. See [LICENSE](LICENSE).
