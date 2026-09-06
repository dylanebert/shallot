# Style

Applies to all repo code, including tests, tooling and examples. Read a sibling before adding a function, system or plugin; follow the naming and shape in `src/engine/` and `src/standard/`. Data-over-methods lives in `packages/shallot/AGENTS.md`; primitive choices live in `ecs.md`.

## Naming

Use the shortest clear word, a single verb where it fits. Module scope supplies context: don't repeat the file or type name. Qualify only to distinguish real alternatives, not to describe the body. A multi-word name may hide several jobs; split those jobs rather than disguise a call chain. Use PascalCase for components, plugins and singletons; camelCase for functions and locals.

## A function is a transform; a system is a loop

Logic is data in, data out. Orchestration is a flat sequence or entity-query loop: query, guard, act. Use early returns instead of nested branches, and named transforms instead of private helpers calling helpers. Extract pure steps that tests can call in isolation; inline steps used only once. Plugins are plain objects with components, systems, dependencies and lifecycle hooks, not classes. Follow `SearPlugin` and `PartPlugin` in `standard/`.

## Comments earn their place

Default to none. Earn a comment with a public export's JSDoc contract or the why behind a non-obvious line; `sear/` and `slab/` set the bar. State today's invariant, never restate the code or narrate an edit. History sections, refuted alternatives and workflow chronologies belong in Git. Algorithm step labels are fine: they name the algorithm, not this repo's workflow.

Never anchor comments to workflow stage IDs, private planning paths or deleted symbols. Write the invariant so a standalone reader can check it. `scripts/check-docs.ts` checks tracked `.ts` comment citations against `.md` basenames tracked in this repo only. It does not validate full paths, scan other source languages or distinguish bare workflow IDs from algorithm labels; those still need a read. `check-docs.test.ts` preserves dead/private refusal and live resolution.

## Instruction budgets

Keep root-to-leaf entry-doc chains within 32768 bytes. `scripts/check-docs.ts` enforces root plus `packages/shallot`, and root plus `examples`, in `bun run check`. Fold detail into scoped rules: exceeding the context-loader budget silently drops the deepest file and its contract.

The same check ratchets instruction bytes and longest blank-line-delimited paragraph per file, plus corpus bytes. After cuts, explicitly run `bun run scripts/check-docs.ts --lower`; checking never writes. Its Git-derived population covers any-depth AGENTS.md, CLAUDE.md and `.claude/rules/*.md`, including untracked files; ignored files and other names are outside that vocabulary. New members and symlinks refuse.
