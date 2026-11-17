# Design notes

## What Strata gives us (v0.1.45)

- `POST /slots/0?action=save` writes the **live** conversation (the last request's) to a file. A conversation
  parked in RAM can't be saved.
- `action=restore` loads a file as the live conversation and **drops** the one that was live; it isn't parked.
- Both run in Strata's request queue, between requests, so they never interleave with a generation.
- A file is tied to the model and engine settings that wrote it; Strata refuses others. It writes under a
  temporary name and renames into place.
- The RAM cache parks a conversation when another comes in, evicting the oldest. It refuses one bigger than its
  budget or one that would leave less than its RAM floor.
- Strata matches prompts by tokens, so a file of an earlier turn still saves everything up to that turn.
- With `--lazy` / `idle_unload_s` the engine goes away when idle: `/metrics` reports no context and no KV type,
  slot calls answer 503 `not loaded`, and a generation request loads it again.

## Policy

A continuing turn passes straight through. On a switch, the incoming conversation's file is restored unless the
conversation is believed parked in RAM. The outgoing one is saved at the switch only when it would be lost there:
a restore is about to drop it, or Strata would refuse to park it. Otherwise it's saved after `--idle-save-s` of
quiet, if its file is more than `--regrow-tokens` behind.

Every choice affects speed only. Strata decides what to reuse from the tokens, so a wrong guess costs a wasted
restore or a fresh read, never a wrong answer. That makes failing open safe.

## Edge cases that matter

| case | behaviour | test |
| --- | --- | --- |
| Any error or panic in the cache | the request goes through as if the proxy weren't there | `TestFailOpen`, `TestRestoreFailure` |
| Strata version, model or settings changed | new fingerprint; files of the old one are deleted | `TestStrataRestartAndFingerprint`, `TestFingerprint` |
| Partial, truncated or corrupt files | Strata writes atomically; leftover temporaries, empty files and old formats go at start; a file whose size changed is dropped before restore; one Strata rejects is deleted (one it lacked RAM for is kept) | `TestLoadCleansUp`, `TestCommitLookupAndCorruptFiles`, `TestRestoreFailure` |
| Disk nearly full, or the cap too small | the oldest files go until a save fits under the cap and above the free-space floor; if it can't, no save | `TestDiskFloorWins`, `TestCapAndLRU` |
| Overlapping requests (incl. one arriving mid-save or mid-restore) | passed through untouched; the live conversation is forgotten until a request runs alone | `TestOverlappingRequests`, `TestRequestDuringIdleSave` |
| Client cancels mid-stream or mid-restore | the restore finishes and the file stays; a cut-off answer isn't trusted as live | `TestClientCancel` |
| Strata restarted, its engine unloaded, or a 5xx | the RAM view and the live conversation reset; files still restore | `TestStrataRestartAndFingerprint`, `TestUnloadedEngineKeepsFiles` |
| The engine is unloaded (`--lazy`, `idle_unload_s`) | the arbiter owns loads: admission loads the resident route's model when the engine reports unloaded (a status read that fails is not evidence), the restore then runs on the loaded engine, and the load drops the route's live-conversation beliefs; `/metrics` of an unloaded engine is not read as a fingerprint, so the files stay; files of a fingerprint that changed are evicted before any restore; a load the engine refuses is fail-open — the request goes to the engine and its answer reaches the client | `TestColdEngineIsLoadedThenRestored`, `TestColdLoadIntoAnotherModelEvictsFiles`, `TestRefusedLoadFailsOpen`, `TestUnloadedEngineKeepsFiles` |
| A save whose file is mostly fixed overhead | the bytes-per-token estimate is learned only from a save of at least `--min-tokens`, and is clamped to 4 KB..256 KB/token, so the next reserve is neither poisoned nor under-set and the long session still gets its file | `TestSmallSaveDoesNotPoisonTheEstimate`, `TestLearnedBytesPerTokenIsClamped` |
| A request names a model the arbiter does not serve, or names none | in strict mode (the routes came from `--route` flags) it is answered 404 with the model named, and the card does not move; in single-route mode the resident answers it and the request line logs the model that was asked for, so the alias is visible | `TestStrictRefusesAnUnknownModel`, `TestStrictRefusesARequestWithNoModel`, `TestSingleRouteLogsTheRequestedModel` |
| A background save of a long conversation, or a request already queued for the card | the idle save is skipped (`--max-idle-save-tokens`, non-empty queue): a 100K-token save writes 1.4 GB in ~110 s behind the save lock, and the yield save at the switch writes the same file in ~4 s | `TestIdleSaveSkipsALongConversation`, `TestIdleSaveStepsOutForAQueuedRequest` |
| A router instance restarts on a new port | the port is not part of the fingerprint, so the route's slot files survive a swap | `TestLlamaFingerprintIgnoresThePort` |
| A file that restores far fewer tokens than its name claims | forgotten, so the conversation is saved afresh instead of being skipped as on disk | `TestTruncatedSessionFile` |
| The client closes the socket after the stream's terminal event (`data: [DONE]` on its own line, a `finish_reason`, `message_stop`, `response.completed`) | a complete answer: the live conversation is kept and saved; the marker is searched in every chunk written, so a `response.completed` followed by kilobytes counts, and `[DONE]` quoted inside streamed content does not; a cancel while the stream runs is still a cancellation and is not trusted | `TestClientGoneAfterCompletion`, `TestClientCancel` |
| Body over `--max-body-mib`, malformed JSON, other endpoints | passed through unbuffered or as they are; anything that may change Strata's state makes the live conversation unknown | `TestFailOpen` |
| SIGTERM mid-save | requests and the save get `--shutdown-s`; Strata completes a save cut off later, and the next start indexes it | `TestShutdownMidSave` |
| Prompt or answer text | never logged | `TestDialectsAndNoContentInLogs` |

## Known gaps

- A conversation that parks and is evicted before a quiet moment has no file. That's the same as running without
  the proxy.
- The RAM view assumes all traffic goes through the proxy.
- An edit to an earlier message shares too little to restore anything, so it's read fresh.
