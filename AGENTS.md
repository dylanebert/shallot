Read `CONTRIBUTING.md`.

Pre-land check: `bun check && bun run lint && bun run test --path-ignore-patterns '**/*.gpu.test.ts' && bun test --todo ./scripts/check-*.node.ts && { git diff --quiet origin/main -- crates || cargo fmt --check; } && sh scripts/clippy.sh`.
