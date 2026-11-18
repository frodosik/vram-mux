# Contributing

Issues and pull requests are welcome.

- `make test` and `make lint` must pass.
- New behaviour needs a test. `internal/strata/stratatest` is a fake Strata to test against;
  `internal/proxy/proxy_test.go` runs the proxy through it.
- Edge cases that protect correctness or the disk go in `docs/design.md` with the test that covers them.
- Add a dependency only when it clearly beats a few lines of our own, and say why in the pull request.
- If you change how the proxy talks to an engine, say which engine and version you checked it against (Strata v0.1.45–v0.1.61, llama.cpp b12445).
- Use conventional commit messages (`fix: ...`, `feat: ...`) and add a line to CHANGELOG.md.

## Pull request description

Every pull request body carries these four sections, in this order:

1. **Idea / Motivation** — the problem being solved and why it matters.
2. **What Changed** — a concise summary of the architectural and code changes.
3. **How It Was Tested** — the exact commands run (`make test`, `make lint`), the unit tests added,
   and any mock (`stratatest`) or live repro steps, with the observed output.
4. **Before vs. After Results** — when the change affects behavior, measured numbers: latency,
   memory/VRAM, token reuse, file sizes, saves skipped. A behavior-only change says so explicitly.

`.github/pull_request_template.md` pre-fills the sections; fill them in, do not delete a section.
