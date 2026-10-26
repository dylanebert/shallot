Read `CONTRIBUTING.md`.

Pre-land check: `bun check && bun run lint && bun run test --path-ignore-patterns '**/*.gpu.test.ts' && bun test --todo ./scripts/check-*.node.ts && { git diff --quiet origin/main -- crates/physics || { cargo clippy -p shallot-physics --all-targets -- -D warnings && cargo clippy -p shallot-physics --lib --target wasm32-unknown-unknown -- -D warnings; }; }`.
