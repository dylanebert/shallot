# Contributing

## issues

The most useful thing you can file is a bug report with a minimal reproduction: a scene file, a small project, or a failing test. Feature requests and questions are welcome too, and so is the [discord](https://discord.gg/eEY75Nqk3C).

## pull requests

Pull requests are by invitation only. Shallot is a single-author engine at this stage: its conventions are load-bearing and coherence matters more than throughput. If you want to work on something, open an issue or ask in discord first — don't start from a PR.

## developing

The gate before pushing is `bun check` and `bun run test`. The slow suites are separate: run the one covering what you changed, by path.

- `bun test ./packages/shallot/tests/avbd/*.oracle.ts` — AVBD/physics
- `bun test ./examples/gym/src` — the engine or host layer
- `bun run --cwd packages/shallot test:fixture` — tumble kernel changes
- `cargo test` in `packages/shallot/rust/audio` — rust audio changes
- `bun run test:install` — packaging, CLI, manifest, or scaffold changes
- `bun run flows` / `bun run recipes` — serialize-restore, overlay UI, physics recipes
- `bun bench --for <the files you changed>` — after GPU changes; it names the gating scenarios, and `--sweep` runs them

The GPU suites are display-gated: without a display they skip, and a skip is not a pass.

Engine-internal layout, the command table, and the rules index are in [AGENTS.md](AGENTS.md); the conventions themselves (ECS shape, GPU patterns, testing tiers) are path-scoped under [`.claude/rules/`](.claude/rules/), so read the ones matching files you touch. The consumer-facing contract, how a game uses the engine, is [packages/shallot/AGENTS.md](packages/shallot/AGENTS.md).

Build-from-source instructions are in the [README](README.md#from-source).
