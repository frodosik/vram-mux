#!/bin/bash -eu
# NVIDIA, llama.cpp only: two llama-server processes share one card through the
# arbiter. llama.cpp has no amdgpu sysfs to read, so the gate uses the
# nvidia-smi backend: install nvidia-smi, set --gpu-index to pin a card
# (-1 picks the one with most VRAM), and keep --vram-poll-s at 2 s.

SESSIONS=/var/lib/gpu-arbiter/sessions

# First route: a 27B dense model, persistent slots so saves and restores work.
llama-server -m /srv/models/qwen3.8-27b.gguf \
  --host 127.0.0.1 --port 8001 --ctx-size 32768 --slots --persistent &

# Second route: a security-review model, smaller context.
llama-server -m /srv/models/orcacyber.gguf \
  --host 127.0.0.1 --port 8002 --ctx-size 16384 --slots --persistent &

# The arbiter: one card, llama routes only. --body-divisor 4 estimates tokens
# from request bytes because llama.cpp reports no token count; the yield at a
# handoff writes the held conversation's file, so a return restores from disk
# instead of re-prefilling. --load-timeout-s 600 covers drain + gate + load.
gpu-arbiter \
  --listen :8080 --metrics-listen 127.0.0.1:8082 \
  --session-dir "$SESSIONS" \
  --route "llama|llama/27b|qwen3.8-27b|http://127.0.0.1:8001||18000" \
  --route "llama|llama/orcacyber|orcacyber-security-reviewer|http://127.0.0.1:8002||9000" \
  --max-gib 40 --min-tokens 15000 --regrow-tokens 4096 --min-free-mib 4096 \
  --idle-save-s 3 --max-idle-save-tokens 32768 --body-divisor 4 \
  --max-resident-streak 32 --vram-margin-mib 1024 --gpu-index -1 \
  --vram-poll-s 2 --drain-wait-s 300 --load-timeout-s 600
