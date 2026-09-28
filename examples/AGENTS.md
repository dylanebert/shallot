# Examples

From `bun run format` and `examples/*/shallot.json`; edit manifests.
Each example owns its `index.html` and Vite config with `plugins: [shallot()]`; Vite runs it.
From the repository root, use `bunx shallot dev examples/<name>`.
Run cheap checks with `bun run test` and one example's browser checks with `bunx playwright test -c examples/<name>`.

## Recipes

| name | description | add |
| --- | --- | --- |
| `first-person` | I want a first-person character to climb a route with a moving lift. | `bunx shallot add first-person` |
| `loading-screen` | I want the host page to stay visible while a dedicated scene area loads and reveals its first frame. | `bunx shallot add loading-screen` |
