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

Reference: `docs/engine/ecs.md` for entities, components, systems, queries, plugins, traits, and the auto-generated API surface.

## Plugin lifecycle gotchas

- `initialize(state)` runs BEFORE scene parse — no entities exist. Use `warm(state)` for anything that needs scene data
- Plugins that register meshes or surfaces MUST declare `dependencies: [RenderPlugin]` — RenderPlugin.initialize clears the render registries first (*Reload-safety* below), so the dependency orders the producer's registration after the wipe
- System `setup(state)` is NOT plugin initialize — called lazily on first frame the system runs

## Reload-safety: lifecycle + module scope

A `State` is rebuilt routinely — the editor rebuilds on a scene switch, plugin toggle, or play/stop, re-running every plugin's `initialize`/`warm` against the **same module-level singletons** (registries and services persist across States). In-place plugin hot-reload (swapping a system's behavior on a live State) is the same shape, tighter. Both need the two rules below — better practice regardless, load-bearing once a State outlives one build.

**Module scope holds idempotent definitions + registries, never runtime identity.**

- No module-cached eids or per-entity handles. An eid is a borrow (see *Entity reference fields*) — a rebuild recycles it to a different entity. Hold the reference in a component (`@name`/relation) or re-query each frame.
- No module-level runtime accumulators (`let angle += dt`). Derive from `State` (`state.time.elapsed`), so the value is correct after a rebuild and isn't a hidden second source of truth.
- A module-level registry (the `Surfaces`/`Draws`/`Meshes` shape, or a plugin's own) is **idempotent w.r.t. `initialize`**: clear then rebuild, so re-running `initialize` is a no-op-equivalent. `RenderPlugin.initialize` clearing `Surfaces`/`Draws` (`Registry.clear`) + `clearMeshes()` is the exemplar — a same-set rebuild re-registers identically, and a plugin **toggled off** leaves no stale entry (otherwise its dead draw/mesh is paired against torn-down buffers — a GPU error, the conformance "producer toggle" gate).

**Lifecycle phases are idempotent and re-runnable.** `initialize` — registration only, pre-scene, no entities; clears + rebuilds any registry it owns. `warm` — post-parse GPU setup + derived (non-authored) spawns, idempotent; warm-spawned entities re-create each build (they live in `State`, not the Document). `setup` — per-`State` lazy init. `update` — pure over `State`. `dispose` — teardown.

**Session invariants, fixed for a `State`'s life:** capacity, the registered-component set, each component's schema, and the membership generation count (`build` assigns every component its bit up front). A change to any is a fresh build, never an in-place migration.

**Stable component ids — identity survives a reload.** Membership, queries, and the traits exclusions view key on a component's `idOf(component)` (`ecs/core`), not the object — `intern`ed **by name** at `register` and resolved on re-registration, so a reloaded module's fresh component object resolves to the same id, and `register` copies the prior stores onto the new handle (runtime data + GPU buffers survive). A bare (unregistered) component auto-mints an anonymous, object-stable id. The id key is a **Symbol** on purpose — an enumerable `id` would be misread as a field by the `Object.keys`/`Object.entries` walks (`readFields`, `inspect`, `fields`). All of it is slow-path (register / add / remove / query registration), not the per-frame path. `swap()` (engine barrel) is the in-place plugin hot-reload built on this — see its JSDoc.

## Runtime state

ECS state is built from a Document via `load()`. `serialize(state)` is the on-demand inverse (save / survive-reload / rebuild), never per-frame. Format a component to its scene attribute through the one shared `readComponent` (`scene`), never a second copy of defaults + `readFields` + `formatFields`. A round-trip preserves codec-representable component values; GPU buffers and `warm`-spawned (derived) entities are rebuilt, not serialized.

`load` records each entity's identity on `state.identity` (its scene `id` + the load-authored set; an eid stays a borrow, so this is the durable-by-name half — see *Entity reference fields*). `serialize(state)` reads it to serialize **the authored set only** — `warm`-derived entities are absent by construction, so a restore (`load` then `warm`) never doubles them — and to round-trip an entity-ref field as `@<id>`. Pass `serialize(state, eids)` to serialize entities spawned outside `load`.

## Edit mode contract

Systems with `mode: "always"` run in edit mode but must be non-destructive — update existing component values only, never add/remove components. Use field values (set to 0) rather than add/remove.

The rule guards the **authored** document: a system churning an authored entity's registered components per frame desyncs the inspector + serialize. A one-time, marker-gated **load** is outside it — spawning *derived* entities (not in the Document, like a `warm` spawn) and gating on an *unregistered* marker (out of `entries()`, so never serialized or shown in the inspector) touches no authored state. The shape is `query([Trigger, not(Done)])` → `add(Done)` + spawn: the `not(marker)` one-time gate (the `stagger` system in `examples/zoo/ecs` is the live gate shape) with the spawned entities derived, so a declarative trigger loads content in the editor and at runtime without desyncing anything.

## Choosing the right primitive

- Marker components + `not()` queries — entity-scoped one-time actions ("this entity needs initialization")
- Module-level PascalCase singletons (e.g. `Compute`, `Audio`, `Graph`) — process-scoped shared state. Type name matches singleton name; populate fields in plugin `initialize`, read via direct import. No setters, no `Resource<T>` wrapper

## Component storage contract

Components declare each field as `Single` (scalar), `Pair` (2-lane), or `Quad` (4-lane), produced by `sparse(type)` (CPU, Map-backed — the CPU storage primitive, memory O(live entities)) or `slab(type)` (GPU-mirrored, same Single/Pair/Quad shape plus a dirty-flushed `.gpu` buffer). Pick by direction: `sparse` for CPU-side fields, `slab` for per-entity data a GPU pass reads. Both present the same Single/Pair/Quad surface, so consumers, scene parse, and traits don't see the difference. Type descriptors (`f32, i32, u32, u8, u16, f16, vec2, vec4`, plus the packed mirrors `srgb8x4` / `f16x4`) and the storage interfaces live in `engine/ecs/component.ts`. A packed mirror (`Type.gpu`) keeps the **CPU side lossless** (the full `ctor`×`lanes` — `set`/`read`/serialize see exact floats) and packs only the `.gpu` buffer at flush, so authored scene values never round-trip through the GPU format; never store the packed form on the CPU. Vector fields stay flat — `pos: Quad`, not `posX/posY/posZ` — with lane Singles (`pos.x`) for lane-granular access and bulk `pos.set(eid, x, y, z, w)` for hot-path writes.

A slab's dirty bitset is contract, not implementation detail: one bitset per field, one bit per entity, set on every write, cleared only by the frame flush. That shape is a per-field delta stream (the seam a future replication encoder reads), so keep the granularity — never collapse bitsets across fields or clear bits outside the flush.

Storage fields are pure data — `state.remove`/`state.destroy` do NOT reset them, and a recycled eid inherits whatever the field held until a default re-applies on the next `state.add`. GPU consumers that scan a slab by index (the Part pack reads `surface[eid]` for every slot 0..capacity) skip dead and non-member slots by gating on **component membership**, not on a value smuggled into the data. `SlabPlugin` mirrors the ECS membership bitset to a `"membership"` GPU buffer (one 31-bit word per entity per generation); a pack shader skips `eid` when `(membership[gen * capacity + eid] & mask) == 0`. `state.membership.bit(component)` returns the `{ gen, mask }` to template into that gate (see `Membership` in `engine/ecs/component.ts`). Because membership is the authoritative liveness signal, `state.destroy`/`state.remove` dropping the bit is enough to stop a Part rendering the next frame — no per-field clear, no sentinel value reserved out of the data domain.

A flat component is a real data shape, not a bundle of named scalars. Splitting bloats the schema, mismatches GPU layouts (a vec4 slab is one bind-group entry; four lane Singles is bookkeeping over the same buffer), and would force a name-suffix detector inside scene parse and reflection. TypedArray backing also halves memory vs `number[]` for f32 (4 B vs V8's 8 B doubles), is demand-paged for sparse usage, has no element-kind deopt cliffs, and maps directly to GPU upload.

### Scope

- **All live code is clean** — `engine/`, `standard/`, and `extras/` declare Single/Pair/Quad directly, no `column()` backing, no raw `number[]`. The engine has no split-suffix support: no `detectVecN` detectors, no `${name}X/Y/Z` parse/format branches, no `key.endsWith("X")` schema collapse.

### Migration guidelines

- **One Pair/Quad per logical vector.** `pos: Quad` not `posX/posY/posZ` lane Singles. Bulk authorship: `pos.set(eid, x, y, z, w)`. Lane access on hot paths: `pos.x.get(eid)`.
- **Type per field — pick the narrowest accurate type.** `f32` for floats. `u32` for entity IDs. `u8` for boolean flags. `i32`, `u16`, `f16` as the data calls for. Hex-encoded colors stay `f32` (numeric value, codec at parse time).
- **Defaults are arrays.** `defaults: () => ({ pos: [0, 0, 0, 0], rot: [0, 0, 0, 1] })`. Dotted keys (`"pos.x": 0`) accepted for partial defaults. `applyDefaults` resolves both to one bulk `Quad.set` per field.
- **Scene attributes accept 1, 3, or 4 values for a Quad; 1 or 2 for a Pair.** Trailing-default lanes elide on format (`pos: [1, 2, 3, 0]` → `pos: 1 2 3`). Dotted attributes (`pos.y: 5`) write a single lane.
- **Hot loop pattern — hoist→local→writeback.** Inside a query loop, `let v = field.get(eid)` once at the top, mutate, write back at the scope boundary. Avoid `field.set(eid, field.get(eid) - delta)` chains — they read worse against `slab(...)` where the write boundary is semantic.
- **Single read OR single write — direct call, no hoist.** `field.set(eid, 1)` and `field.get(eid)` are fine standalone.
- **Test float equality.** `toBe(x)` only for f32-exact values (powers of two, small integers). `toBeCloseTo(x)` for fractional values — `Float32Array` round-trip won't preserve them.
- **Euler / hex / matrix conveniences live in an authoring `alias` (`eulerAlias`) or codec helpers (`parse`/`format`), never on the component.** Programmatic authors call `euler()`/`quat()` directly.
- **Import discipline.** `import { sparse, slab, f32, u32, vec4, ... } from "../../engine"` — barrel re-exports them; the deep `engine/ecs/sparse` path fails the import-check rule.

### Pattern references

- `standard/transforms/index.ts` — canonical direct Quad component, slab-backed.
- `engine/scene/xml.test.ts` — scene-parse contract end to end ("direct Pair/Quad — scene parsing").

## Single-writer rule

ReadbackSystem is the sole writer of `attr.value`. Editor gestures communicate via `onsync` (live ECS) + the Document edit API (`doc.setAttr`, or a `doc.begin`/`commit` gesture that coalesces a drag's writes into one undoable entry, prev auto-captured) — never write `attr.value` directly. Never add serialization concerns to ECS.

## Entity reference fields

Component fields that store entity IDs (like `Joint.a`, `Tween.target`, `Player.camera`) use `@name` syntax in scene files. ReadbackSystem skips attributes containing `@` to avoid converting name references to literal entity IDs — literal IDs break on serialize→reload because entity ID assignment depends on creation order (which differs between edit and play states). The scene loader resolves `@name` to a real eid at load time.

A ref field declares itself by **type**: `target: sparse(entity)` (the `entity` descriptor — u32 storage tagged as a ref), not `sparse(u32)`. `serialize` enumerates them (`refs`) and, with the scene `id` `load` recorded on `state.identity`, emits each as `@<id>` (minting an id for a referenced target that lacks one), so a ref round-trips by name across the creation-order eid reshuffle a reload causes. The type is the one source of truth — a plain `sparse(u32)` (e.g. `Tween.field`, which interns a path) is never a ref, so no parallel list can drift.

Scenes are flat: no XML nesting, no engine-level parent component. Consumers that need attachment (player → camera, water → chunks, sequence → tweens) declare a consumer-shaped relation — a numeric eid field on the relevant component, resolved via `@name` at load time. `state.destroy(eid)` removes one entity (dropping its component membership) — no cascade to related entities.

**The transform substrate is flat by definition.** `Transform` is a per-entity world transform (the flat `transforms` firehose sear + the pack read); no `parent` field, no per-frame parent-graph traversal. Relative / hierarchical / animated transforms are consumer concerns that depend inward and emit substrate-native flat output, never an engine parent graph: a static glTF node chain bakes to a flat world matrix at import; runtime attachment is the consumer-shaped relation above (a system writes the follower's flat `Transform` from the target each frame); skinning bakes its clip to per-frame vertex textures (the VAT) the importer's `skin` surface samples per-vertex in its `vs` chunk — the instance root stays flat in the firehose, the VAT a separate binding (a live joint palette would ride the same shape). Don't grow `Transform` a `parent` to make imported hierarchies "just work" — that taxes the entities that don't animate to serve the few that do; the convenience layer produces flat output and the substrate stays blind. This is the transform analogue of the no-Hi-Z call (`render.md` "Culling lives in the producer"): refuse the universal runtime mechanism in the substrate, relocate the real need to an inward-depending layer.

**An eid is a borrow, not a durable handle.** It is valid for the scope you obtain it in, recycled on `state.destroy`, with no version packed in (it stays a bare index — see the storage contract). Recycle is handled by membership, not a sentinel: a dead slot is gated out, a reused slot re-applies defaults on `state.add`, so a system that re-queries each frame is always safe. A *held* (cached-across-frames) bare eid is not — validate it with a `state.has`/membership check. That catches a despawned target but **not** a slot recycled to a new same-component entity; if that realias matters, carry an explicit `(eid, version)` pair against a side-array version counter — never a version packed into the eid (it stays a bare index). Durable cross-session identity (serialization) is a separate concern from the runtime eid.

## Anti-patterns

- `lastState` / `resetIfNewState` guards — scope the state instead
- `lastCamera` skip-checks — premature dirty tracking
- `state.exists` guards — defensive code for cross-State leaking
- Module-level `Map<number, ...>` for entity ownership — use a consumer-shaped relation (eid field on a component) with marker components
- Per-frame gather-and-`return null` on GPU buffers that are stable post-warm — a draw-group consumer runs after `warm()` (slab `.gpu`) and the `first` `MembershipSystem` (`membership`), so they're always up at the call site. Read them directly, build the bind group once (rebuilt only on an identity change), let a missing one throw — a null = wiring bug, not a frame to skip. `standard/transforms` is the exemplar
