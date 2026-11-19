#!/bin/bash -eu
# One RDNA3 card, two routes: Strata (resident experts) and a llama.cpp 27B,
# with vram-mux owning the loads. Run as the user the engines run as.
# The card is shared: the arbiter drains the resident route, yields, gates on
# free VRAM, and loads the requested model.

SESSIONS=/var/lib/vram-mux/sessions
API_KEYS=/etc/vram-mux/api.keys

# Strata's engine settings, exported before the launch:
export STRATA_RESIDENT_HEADROOM_GIB=2.6
export STRATA_IO_PF_BUDGET_MB=256
export STRATA_IO_PF_SCHED=1
export STRATA_ROUTE_TAIL_SKIP=0

# Strata: MoE with resident experts on one card. --spec 4 keeps 4 speculative
# drafts; spec-min-p stops speculation below 0.5 acceptance; resident experts
# are capped at 10.0 GiB so the OS keeps ~9 GiB free for 180k+ contexts;
# headroom 2.6 GiB, a fixed 256 MB read-ahead window, scheduled prefetch, and
# no route-tail skip on gfx1100.
strata --serve --host 127.0.0.1 --port 8095 \
  --native /srv/models/flash-next-iq2_xs.gguf \
  --slot-save-path "$SESSIONS" --parallel 1 \
  --prefill 6656 --spec 4 --spec-min-p 0.5 --resident-budget-gib 10.0 \
  --session-min-free-mib 512 &

# The llama route is a router, not a server with a model loaded: presets load on
# demand through POST /models/load, slots persist so the arbiter can save and
# restore them, and nothing is preloaded - a preloaded model takes the card at
# once and the arbiter's free-VRAM gate has nothing left for Strata. No api key
# here: the route's api-key field is empty, so a key on the router would answer
# the arbiter with 401. Give both the same key if you want one.
llama-server --models-preset ./presets.ini --models-max 1 --no-models-autoload \
  --slot-save-path "$SESSIONS" --host 127.0.0.1 --port 8001 &

# The arbiter: clients on :8080, admin on 127.0.0.1:8082.
# --max-gib 80 bounds the session files; --min-tokens 4096 saves turns of 4k+
# (turn-1 boundaries at 1024); --regrow-tokens 4096 re-saves after growth;
# --min-free-mib 512 matches Strata's floor; --snapshot-bytes-per-token 8192
# predicts park refusals for this quant; --idle-save-s 2 saves at turn
# boundaries; --max-idle-save-tokens 32768 keeps a big idle save from holding
# the save lock; --strata-need-mib 3072 is Strata's VRAM floor, not what
# expert-cache fills; --body-divisor 4 estimates tokens for llama.cpp, which
# reports none; --max-resident-streak 32 drains the resident before a waiter
# takes the card; --vram-margin-mib 1024 must stay free before any load;
# --drain-wait-s 300, --load-timeout-s 600 cover a whole swap.
vram-mux \
  --listen :8080 --metrics-listen 127.0.0.1:8082 \
  --session-dir "$SESSIONS" \
  --upstream http://127.0.0.1:8095 --strata-model qwen3.8-flash-next-iq2_xs \
  --strata-api-key "file:$API_KEYS" \
  --route "llama|llama/27b|qwen3.8-27b|http://127.0.0.1:8001||18000" \
  --max-gib 80 --min-tokens 4096 --regrow-tokens 4096 --min-free-mib 512 \
  --snapshot-bytes-per-token 8192 --idle-save-s 2 --max-idle-save-tokens 32768 \
  --strata-need-mib 3072 --body-divisor 4 \
  --max-resident-streak 32 --vram-margin-mib 1024 --drain-wait-s 300 \
  --load-timeout-s 600 --vram-poll-s 2
