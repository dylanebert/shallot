---
paths:
    - "packages/shallot/src/engine/**/*.ts"
    - "packages/shallot/src/standard/**/*.ts"
    - "packages/shallot/src/extras/**/*.ts"
    - "packages/shallot/editor/**/*.ts"
    - "packages/shallot/editor/**/*.svelte"
    - "examples/**/*.scene"
    - "examples/**/*.ts"
---

# ECS & Document Model

Reference: `docs/engine/ecs.md` for queries, component dependencies, driver pattern, custom meshes, and API surface.

## Plugin lifecycle gotchas

- `initialize(state)` runs BEFORE scene parse — no entities exist. Use `warm(state)` for anything that needs scene data
- Plugins that register meshes or surfaces MUST declare `dependencies: [RenderPlugin]` — RenderPlugin.initialize calls `clearMeshes()`/`clearDefaultSurfaces()`, wiping earlier registrations
- System `setup(state)` is NOT plugin initialize — called lazily on first frame the system runs

## Runtime state

ECS state is runtime-only — constructed from Document via `load()`, never serialized back. Discarded on stop.

## Edit mode contract

Systems with `mode: "always"` run in edit mode but must be non-destructive — update existing component values only, never add/remove components. Use field values (set to 0) rather than add/remove.

## Choosing the right primitive

- Marker components + `not()` queries — entity-scoped one-time actions ("this entity needs initialization")
- `events<T>` — transient cross-system messages where the receiver isn't tied to a specific entity (placement, hit, level-up). Frame-scoped, auto-drained
- Resources — persistent shared state that genuinely needs a data structure

## Single-writer rule

ReadbackSystem is the sole writer of `attr.value`. Inspector gestures communicate via `onsync` and `doc.setAttr()` with explicit `prev` — never write `attr.value` directly. Never add serialization concerns to ECS.

## Entity reference fields

Component fields that store entity IDs (like `BallJoint.bodyA`) must use `@name` syntax in scene files. ReadbackSystem skips attributes containing `@` to avoid converting name references to literal entity IDs — literal IDs break on serialize→reload because entity ID assignment depends on creation order (which differs between edit and play states). Never store entity IDs in component fields for cross-entity references that appear in scene files. Prefer relations for entity-to-entity links.

## Anti-patterns

- `lastState` / `resetIfNewState` guards — scope the state instead
- `lastCamera` skip-checks — premature dirty tracking
- `entityExists` guards — defensive code for cross-State leaking
- Resource `Map<number, ...>` for ownership — use `ChildOf` relation with marker components
