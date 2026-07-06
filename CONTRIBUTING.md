# Contributing

## issues

Issues are open. Bug reports with a minimal reproduction (a scene file, a small project, or a failing test) are the most useful thing you can file. Feature requests and questions are welcome too — so is the [discord](https://discord.gg/eEY75Nqk3C).

## pull requests

Pull requests are by invitation only. Shallot is a single-author engine at this stage: its conventions are load-bearing and coherence matters more than throughput. If you want to work on something, open an issue or ask in discord first — don't start from a PR.

## developing

Build-from-source instructions live in the [README](README.md#from-source). The gate before pushing is `bun check` and `bun run test:full`; `bun bench` after GPU changes.

Engine conventions — the ECS shape, plugin layout, GPU patterns, testing tiers — live in [packages/shallot/AGENTS.md](packages/shallot/AGENTS.md).
