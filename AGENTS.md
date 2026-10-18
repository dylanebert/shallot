Read `CONTRIBUTING.md`.

Pre-land check: `bun check && bun run lint && bun run test --path-ignore-patterns '**/*.gpu.test.ts' && bun test --todo ./scripts/check-*.node.ts && bun test --changed=origin/main $(find ./src ./examples ./scripts ./diagnostics -name '*.node.ts') && { git diff --quiet origin/main -- examples || bun test ./src/cli/create.node.ts ./scripts/icons.node.ts ./scripts/icons-packed.node.ts ./scripts/packed-install.node.ts; }`.
CI: `test.yml`.
