# Design notes

## What Strata gives us (v0.1.45, the slot APIs verified through v0.1.61)

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
| The store's counts and the folder | the index owns every session file it counts: a key keeps one file, so a save of the same last chain key deletes the older file of that key; a rename onto a tracked name deletes that file; `Adopt` deletes the file its rename overwrites; a file a rename leaves unindexed is deleted. The turn-boundary files the prefix index holds are counted by the cap, the LRU, the age limit and `Stats`, so `vram_mux_files` and `vram_mux_bytes` are the folder, and `Load()` prunes it under the cap at startup | `TestRepeatedGrowthSavesOnTheSameKeyLeaveNoOrphans`, `TestGrowthAcrossTurnBoundariesSurvivesRestart`, `TestTurnBoundaryFilesAreUnderTheCap`, `TestEvictingATurnBoundaryKeepsTheLastKeyFileOfTheSameKey`, `TestCommitRenameOntoATrackedNameUnlinksIt`, `TestAdoptCollisionOverTheSameKey`, `TestLoadKeepsOneFilePerKey`, `TestCacheSaveKeepsTheFolderMatchingTheIndex` |
| Overlapping requests (incl. one arriving mid-save or mid-restore) | passed through untouched; the live conversation is forgotten until a request runs alone | `TestOverlappingRequests`, `TestRequestDuringIdleSave` |
| Client cancels mid-stream or mid-restore | the restore finishes and the file stays; a cut-off answer isn't trusted as live | `TestClientCancel` |
| Strata restarted, its engine unloaded, or a 5xx | the RAM view and the live conversation reset; files still restore | `TestStrataRestartAndFingerprint`, `TestUnloadedEngineKeepsFiles` |
| The engine is unloaded (`--lazy`, `idle_unload_s`) | the arbiter owns loads: admission loads the resident route's model when the engine reports unloaded (a status read that fails is not evidence), the restore then runs on the loaded engine, and the load drops the route's live-conversation beliefs; `/metrics` of an unloaded engine is not read as a fingerprint, so the files stay; files of a fingerprint that changed are evicted before any restore; a load the engine refuses is fail-open — the request goes to the engine and its answer reaches the client | `TestColdEngineIsLoadedThenRestored`, `TestColdLoadIntoAnotherModelEvictsFiles`, `TestRefusedLoadFailsOpen`, `TestUnloadedEngineKeepsFiles` |
| A save whose file is mostly fixed overhead | the bytes-per-token estimate is learned only from a save of at least `--min-tokens`, and is clamped to 4 KB..256 KB/token, so the next reserve is neither poisoned nor under-set and the long session still gets its file | `TestSmallSaveDoesNotPoisonTheEstimate`, `TestLearnedBytesPerTokenIsClamped` |
| A generation body over `--max-body-mib` | the model is read from the buffered prefix (it is the first key), the request is admitted through the same `Acquire` as any other, and the bytes are streamed to that route unbuffered: no key, so no restore, and the route's live-conversation belief is cleared at the request's end. The request is registered as engine-touching at admission (`inflight`/`seq`, the same bookkeeping `prepare` does), so the idle save of the turn before it finds the card busy and steps out instead of writing behind the request. A route that is not resident never runs a request | `TestOversizedBodyIsAdmittedToItsRoute`, `TestStrictOversizedBodyIsRefusedByTheSameRule`, `TestModelOfReadsATruncatedPrefix`, `TestOversizedRequestKeepsTheIdleSaveOffTheEngine` |
| A request names a model the arbiter does not serve, or names none | in strict mode (the routes came from `--route` flags) it is answered 404 with the model named, and the card does not move; in single-route mode the resident answers it and the request line logs the model that was asked for, so the alias is visible. The refusal is recorded in `/requests` and in `requests_total{action="refused"}` | `TestStrictRefusesAnUnknownModel`, `TestStrictRefusesARequestWithNoModel`, `TestSingleRouteLogsTheRequestedModel` |
| A POST outside the generation paths that names a model | strict mode is one rule, not three paths: a POST that names a model is answered 404 when the arbiter does not serve it, and 404 when its route is not the resident — answered by the resident, the client gets a model the card does not hold; answered by its own route, that model is loaded on an idle card with no yield save of the conversation the card holds. A POST that names no model (`/slots`, `/v1/unload`, a raw prompt) goes to the resident. The name is read from a 64 KiB prefix, so a long control body is not buffered | `TestStrictNonGenPOSTToOtherModelIsRefused`, `TestControlPOSTWithNoModelGoesToResident`, `TestStrictControlPOSTNamingTheResidentPassesThrough` |
| The card was busy for a model the arbiter serves | 503, recorded in `/requests` and counted as `requests_total{action="denied"}`, so a denial is read next to the 404 refusals, not only in the log | `TestBusyCardIsRecordedLikeARefusal` |
| The client walks away while its request waits for the card | no 503, no record, no counter: `Acquire` returns the client's own context error, and a request that never got the card is not a denial the operator has to read | `TestClientGoneWhileWaitingIsNotADenial` |
| A background save of a long conversation, a request already queued for the card, or an oversized request in flight | the idle save is skipped (`--max-idle-save-tokens`, non-empty queue, `quiet` false): a 100K-token save writes 1.4 GB in ~110 s behind the save lock, and the yield save at the switch writes the same file in ~4 s. The queue the cache sees is the arbiter's observer, so the guard holds on the real admission path, and an oversized request registers itself as engine-touching at admission so `quiet` sees the card busy | `TestIdleSaveSkipsALongConversation`, `TestIdleSaveStepsOutForAQueuedRequest`, `TestQueueWiringMakesTheIdleSaveStepOut`, `TestOversizedRequestKeepsTheIdleSaveOffTheEngine` |
| Which model owns the card, and what a refused request did | `resident_info{route,model}` gets every route's series at startup, 1 for the route the arbiter says owns the card (the --lazy Strata route) and 0 for the rest; a refusal appears in `/requests` and in `requests_total{action="refused"}` — a 404 must be visible to an operator, not only in the log | `TestResidentInfoNamesTheModelFromTheStart`, `TestResidentInfoSeedsTheInitialResident` |
| A router instance restarts on a new port | the port is not part of the fingerprint, so the route's slot files survive a swap | `TestLlamaFingerprintIgnoresThePort` |
| A file that restores far fewer tokens than its name claims | forgotten, so the conversation is saved afresh instead of being skipped as on disk | `TestTruncatedSessionFile` |
| The client closes the socket after the stream's terminal event (`data: [DONE]` on its own line, a `finish_reason`, `message_stop`, `response.completed`) | a complete answer: the live conversation is kept and saved; the marker is searched in every chunk written, so a `response.completed` followed by kilobytes counts, and `[DONE]` quoted inside streamed content does not; a cancel while the stream runs is still a cancellation and is not trusted | `TestClientGoneAfterCompletion`, `TestClientCancel` |
| Body over `--max-body-mib`, malformed JSON, other endpoints | the oversized body is admitted to its route and streamed through unbuffered, registered as engine-touching (`inflight`/`seq`) so a background save cannot start on the card it is using; a POST outside the generation paths is read for a model name (a 64 KiB prefix) before a byte reaches an engine, and the rest of the body is forwarded whole, unbuffered; malformed JSON passes through as it is; anything that may change Strata's state makes the live conversation unknown | `TestFailOpen`, `TestOversizedRequestKeepsTheIdleSaveOffTheEngine`, `TestStrictControlPOSTNamingTheResidentPassesThrough`, `TestControlPOSTLongerThanThePrefixIsForwardedWhole` |
| SIGTERM mid-save | requests and the save get `--shutdown-s`; Strata completes a save cut off later, and the next start indexes it | `TestShutdownMidSave` |
| Prompt or answer text | never logged | `TestDialectsAndNoContentInLogs` |

## Known gaps

- A conversation that parks and is evicted before a quiet moment has no file. That's the same as running without
  the proxy.
- The RAM view assumes all traffic goes through the proxy.
- An edit to an earlier message shares too little to restore anything, so it's read fresh.
- A control POST whose `model` key sits past the 64 KiB prefix, or nested inside a value, is read as naming no
  model, so it goes to the resident: `modelOf` reads only the top-level key, and only in the prefix.
- A control POST is not serialized with the arbiter's lease — it pays no `Acquire` — so it can reach its route
  while that route is mid-drain. The strict rule says which route a control call may reach, not when.
