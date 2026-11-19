#!/bin/bash -eu
# Single engine, single route: the upstream strata-diskcache setup. Strata on
# :8095 with session files, the proxy on :8080, no arbiter, no card sharing.

SESSIONS=/var/lib/vram-mux/sessions
API_KEYS=/etc/vram-mux/api.keys

# Strata with slot files turned on and parallel at 1 (session files need it).
strata --serve --host 127.0.0.1 --port 8095 \
  --native /srv/models/flash-next-iq2_xs.gguf \
  --slot-save-path "$SESSIONS" --parallel 1 \
  --prefill 6656 --session-min-free-mib 4096 &

# The cache: --upstream points at Strata, --strata-model is the model the
# route answers for, --strata-api-key reads the key from a file so it never
# appears in ps. The idle save writes turn-boundary files; the yield save
# writes the file a switch would lose.
vram-mux \
  --listen :8080 --metrics-listen 127.0.0.1:8082 \
  --upstream http://127.0.0.1:8095 \
  --strata-model qwen3.8-flash-next-iq2_xs \
  --strata-api-key "file:$API_KEYS" \
  --session-dir "$SESSIONS" \
  --max-gib 40 --retention-days 7 --min-free-mib 4096 \
  --min-tokens 15000 --regrow-tokens 4096 --idle-save-s 3 \
  --max-idle-save-tokens 32768 --snapshot-bytes-per-token 65536 \
  --max-body-mib 64 --shutdown-s 20
