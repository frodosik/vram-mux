#!/bin/bash -eu
# NVIDIA: a pure llama.cpp router setup, one router, two presets, one card.
# Strata + llama-server runs on NVIDIA CUDA exactly as in ../amd-strata-llama,
# with --gpu-index for the free-VRAM gate; this example shows the llama half.
# The arbiter reads free VRAM from nvidia-smi when no amdgpu card answers:
# install nvidia-smi, set --gpu-index to pin a card (-1 picks the most VRAM),
# keep --vram-poll-s at 2 s.

SESSIONS=/var/lib/gpu-arbiter/sessions

# One router, presets loaded on demand, slots persisted, nothing preloaded.
llama-server --models-preset ./presets.ini --models-max 1 --no-models-autoload \
  --slot-save-path "$SESSIONS" --host 127.0.0.1 --port 8001 &

# The arbiter: one card, llama routes only. --body-divisor 4 estimates tokens
# from request bytes because llama.cpp reports no token count; the yield at a
# handoff writes the held conversation's file, so a return restores from disk
# instead of re-prefilling. --load-timeout-s 600 covers drain + gate + load.
gpu-arbiter \
  --listen :8080 --metrics-listen 127.0.0.1:8082 \
  --session-dir "$SESSIONS" \
  --route "llama|llama/27b|qwen3.8-27b|http://127.0.0.1:8001||18000" \
  --route "llama|llama/orcacyber|orcacyber-security-reviewer|http://127.0.0.1:8001||9000" \
  --max-gib 40 --min-tokens 15000 --regrow-tokens 4096 --min-free-mib 4096 \
  --idle-save-s 3 --max-idle-save-tokens 32768 --body-divisor 4 \
  --max-resident-streak 32 --vram-margin-mib 1024 --gpu-index -1 \
  --vram-poll-s 2 --drain-wait-s 300 --load-timeout-s 600
