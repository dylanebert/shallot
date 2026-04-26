# Shallot Patterns

Conventions for games built on Shallot. The engine quietly relies on these — violations produce silent failures or hard-to-debug breakage.

For full reference, see `docs/engine/` and `docs/standard/`. This file is the load-bearing subset.

## Philosophy

Shallot is data-oriented, ECS, declarative. Code shaped this way composes with the engine; code shaped otherwise fights it.

**Add components and systems, not methods.** New behavior is a new component (data) plus a new system (transform). Not a method on an entity, not a class with state, not a manager. The temptation to write `player.jump()` is the most common consumer antipattern — write a `Jump` event or marker and a system that consumes it. Keeping data and behavior separable is what makes ECS observable, parallelizable, and serializable.

**Onion layers — dependencies point inward.** The core is pure data and logic; the outer shell is domain-aware integration. A solver is pure math; the GPU dispatch wrapping it is the outer ring. New code goes in the innermost layer it can. Don't push GPU concerns into pure logic; don't push domain knowledge into engine primitives.

**Scenes declare; code transforms.** XML scenes are the source of truth for entity composition. Imperative entity setup belongs in procedural generation and tests, not the standard load path. If you're writing entity-construction code that mirrors a scene file, use the scene file.

**Compute graphs declare order.** Pass order emerges from declared inputs and outputs. Don't sequence passes manually — declare what reads and writes what.

**One source of truth.** Every piece of data has exactly one authoritative location. Derive, don't duplicate.

**Plugins compose.** Everything is a plugin; nothing is privileged core. Add a feature by adding a plugin, not by modifying engine internals.

## Imports

- `@dylanebert/shallot` — public API: components, registries, registration functions, types
- `@dylanebert/shallot/extras` — opt-in plugins (orbit, tween, raytracing, etc.)
- `@dylanebert/shallot/runtime` — platform layer (`now`, `requestFrame`, `readFile`)
- `@dylanebert/shallot/{render,physics,audio,compute,transforms,ecs,raytracing}/core` — extension API for custom render nodes, custom compute pipelines, diagnostics. WGSL constants, surface internals, GPU buffer layouts

Don't deep-import from `src/`. If something you need isn't in a barrel or `*/core` subpath, file an issue.

## ECS & Plugins

### Plugin lifecycle

`initialize(state)` runs BEFORE scene parse — no entities exist yet. Use `warm(state)` for anything that touches scene data. System `setup(state)` is NOT plugin initialize; it runs lazily on the first frame the system runs.

Plugins that register meshes or surfaces MUST declare `dependencies: [RenderPlugin]`. RenderPlugin.initialize calls `clearMeshes()` / `clearDefaultSurfaces()`, wiping earlier registrations.

The compute graph runs `prepare()` itself. Don't call it manually.

### Choosing a primitive

- **Marker component + `not()` query** — entity-scoped one-time work
- **`events<T>`** — transient cross-system messages. Frame-scoped, auto-drained
- **Resource** — persistent shared state with a real data structure
- **Relation** — entity-to-entity links (`ChildOf`, custom). Always prefer over storing entity IDs in component fields — IDs depend on creation order and break on serialize → reload

In scene files, entity references use `@name` syntax (`body-a: @gnome-body`). ReadbackSystem skips attributes containing `@`.

### Anti-patterns

- **Methods on components** — components are data; behavior lives in systems
- **Manager classes that own entities** — use queries, relations, and systems
- **`Map<entityId, ...>` for ownership** — use a relation with marker components
- **`lastState` / `entityExists` defensive guards** — symptoms of cross-State leaking or missed scope
- **Editor systems with `mode: "always"` that add/remove components** — must be non-destructive; update existing fields instead

## GPU

Custom render nodes and compute passes are normal extension points. The engine ships a compute graph; consumers add nodes.

### Binding limits

`maxStorageBuffersPerShaderStage` is **10 — hard ceiling.** 99.6% of devices support 10; only 64% support 16. Requesting more rejects `requestDevice()` on a third of users.

Per shader stage across all bind groups — splitting groups doesn't help. Both `storage` and `read-only-storage` count.

When you hit it, don't silently exceed (Chrome fails with no diagnostics). Consolidate, in order:

1. Interleave buffers written in the same compute pass into one struct
2. Fold scalars (counts, flags) into header offsets of a related buffer
3. Block-concatenate CPU uploads — two `writeBuffer` calls, one buffer, different offsets
4. Split the pass — last resort, adds latency

Never add per-entity CPU iteration to save a binding.

### Debug methodology

Shaders can't print. The only way to know what a shader computed is write-to-buffer + readback.

1. Pick the exact value you're uncertain about ("`body.pos.y` for entity 0 after primal", not "is broadphase working")
2. `atomicStore(&debug[SLOT], bitcast<u32>(value))`
3. Read it back via `readBuffer` / `readFloat32` / `readUint32` from `compute/core`
4. Compare actual vs expected. Apply one fix at a time

Verify via readback BEFORE changing shader code. Off-by-one, wrong binding, stale bind group all look correct in source.

### Pipelines

- Always `createComputePipelineAsync` + `Promise.all`. Sync creation blocks sequentially
- Every pipeline must include a `label` — appears in stats overlay and bench output
- Don't hardcode `bgra8unorm`; use `navigator.gpu.getPreferredCanvasFormat()` (Linux/Dawn returns `rgba8unorm`)
- DXC (Chrome on Windows) doesn't DCE and stalls on large functions inside dynamic loops. Constant upper bounds with dynamic `break` are fine

## Render

### Gamma pipeline

End-to-end gamma-correct. Three boundaries do conversion; everywhere else is linear:

1. **Hex decode** — `unpackColor`, `createColorProxy.set`, `hexColorProxy.set`. sRGB byte → linear float
2. **Hex encode** — `createColorProxy.get`, `hexColorProxy.get`. Linear → sRGB byte for round-trip
3. **Display encode** — once, mid-shader, in `present.ts` (`linearToSrgb(saturate(color))`)

Don't add `linearToSrgb` at `textureStore` (already encoded). Don't apply it before tonemap or FXAA. Drivers writing raw hex arrays directly (e.g. light colors from a gradient) must encode linear → sRGB byte before storing — otherwise `unpackColor` decodes the byte as sRGB and the value drifts. See `extras/skylab` `packColor`.

### Content tuning

- Reflectivity for dielectrics: 0.02–0.05. Higher values are artifacts of pre-gamma tuning
- Bright accent hex (`0xd49560`) saturates aggressively under intense lighting (linear * intensity > 1.0 collapses to white). Use darker variants (`0x8b6040`) or lower intensity
- Gamma is non-linear; no global intensity multiplier matches every albedo. Tune for dominant tones; accept drift on saturated-bright accents

## Physics

### Collision filtering

`Body.group` (u32, default 0). Group 0 collides with everything. Same non-zero group = pair skipped.

### Kinematic semantics

Kinematic bodies (mass=0, non-character) are CPU-owned. Write `Transform` directly each frame for platforms, levers, grab anchors. Physics readback skips them.

`Move` marker — opt-in for kinematic bodies whose displacement should impart velocity to contacts (platforms, elevators, conveyors). Without `Move`, position changes are teleports (no friction drag, no ground velocity inheritance).

### Character vs Player

Demonstrates the philosophy: split unopinionated primitive from opinionated driver.

`Character` accepts world-space velocity intent (`moveX`, `moveZ`, `jump`), returns `grounded`. No keyboard, no camera. `Player` reads input, writes velocity to Character, interpolates camera at render rate.

```xml
<a player character body ...>
  <a camera viewport transform />
</a>
```

Player finds the camera via the parent-child relation. Add a new control scheme by writing a new driver, not by extending Character.

### Contact readback

`Contact.bodyA` / `bodyB` are GPU body indices, not ECS entity IDs. Map via `physics.bodyEids[idx]` before any ECS work. `Contacts` resource is double-buffered, persists 2 ticks.

## Testing

### Two layers for GPU features

- **Unit tests** (`bun test`) — fast, bun-webgpu. Catches logic errors
- **Real GPU** (Playwright) — catches validation, pipeline mismatches, format and alignment issues

Unit tests alone miss real hardware failures.

### `.test.ts` vs `.lab.ts`

- **`.test.ts`** — spec tests. First principles, tight tolerances. Permanent. If one fails, the code is wrong
- **`.lab.ts`** — investigation. Trace internals, probe edge cases. Not run by `bun test`. Temporary

### Tolerance discipline

Tolerances earn their place by derivation, not observation:

- **Exact** (mass invariance, quat normalization) → 1e-10 or tighter
- **Truncation error** — derive from integrator order + step size
- **f32 precision** — ~1e-6 relative, accumulates with chain length
- **Solver convergence** — derive from iteration count + penalty schedule

If you can't derive it, use `.lab.ts` to investigate the system.

### GPU timestamps, not FPS

`requestAnimationFrame` measures CPU. WebGPU's `queue.submit()` returns immediately. GPU timestamp queries measure actual hardware execution.
