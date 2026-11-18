#!/bin/bash -eu
# Reproduces the README rows against a running arbiter: a fresh 8k root, its
# turn-boundary idle save, a tiny conversation, then a fork of the root. The
# last block prints the rows as the proxy recorded them. Point PROXY and
# METRICS at your ports; MODEL is the model the routes answer for.

PROXY=${PROXY:-http://127.0.0.1:8080}
METRICS=${METRICS:-http://127.0.0.1:8082}
MODEL=${MODEL:-qwen3.8-flash-next-iq2_xs}

# body <system-words> <messages...> -- a chat body with real message objects.
body() {
  python3 - "$MODEL" "$1" "${@:2}" <<'PY'
import json, sys
model, words = sys.argv[1], int(sys.argv[2])
msgs = [{"role": "system", "content": "word " * words}]
msgs += [{"role": "user", "content": t} for t in sys.argv[3:]]
print(json.dumps({"model": model, "max_tokens": 8, "messages": msgs}))
PY
}

post() {
  curl -s -m 150 -o /dev/null -w "%{time_total}s\n" -X POST "$PROXY/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$1"
}

echo "root A (8,000 system words + one message):"
post "$(body 8000 u0)"
sleep 3 # the idle save writes A's turn-boundary file

echo "tiny conversation Z (below --min-tokens, no save):"
post "$(body 1 z0)"

echo "fork F (A + one tail message, restore from A's boundary file):"
post "$(body 8000 u0 fork-tail)"

echo "what the proxy recorded:"
curl -s "$METRICS/requests" | python3 -c '
import json, sys, time
for r in json.load(sys.stdin)["requests"][-4:]:
    print(time.strftime("%T", time.localtime(r["at"])), r.get("diskcache") or r)
'
