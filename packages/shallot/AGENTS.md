# Shallot Patterns

Conventions for games built on Shallot. The engine quietly relies on these — violations produce silent failures or hard-to-debug breakage.

For full reference, see `docs/engine/` and `docs/standard/`. This file is the load-bearing subset.

## Philosophy

Shallot is data-oriented, ECS, declarative. Code shaped this way composes with the engine; code shaped otherwise fights it.

**Add components and systems, not methods.** New behavior is a new component (data) plus a new system (transform). Not a method on an entity, not a class with state, not a manager. The temptation to write `player.jump()` is the most common consumer antipattern — write a `Jump` event or marker and a system that consumes it. Keeping data and behavior separable is what makes ECS observable, parallelizable, and serializable.

**Onion layers — dependencies point inward.** The core is pure data and logic; the outer shell is domain-aware integration. A solver is pure math; the GPU dispatch wrapping it is the outer ring. New code goes in the innermost layer it can. Don't push GPU concerns into pure logic; don't push domain knowledge into engine primitives.

**Scenes declare; code transforms.** XML scenes are the source of truth for entity composition. Imperative entity setup belongs in procedural generation and tests, not the standard load path. If you're writing entity-construction code that mirrors a scene file, use the scene file.

**Systems declare order.** A system declares `after` / `before` other systems (and a `group`); the scheduler topo-sorts. Don't sequence frame work manually.

**One source of truth.** Every piece of data has exactly one authoritative location. Derive, don't duplicate.

**Plugins compose.** Everything is a plugin; nothing is privileged core. Add a feature by adding a plugin, not by modifying engine internals.

## Imports

- `@dylanebert/shallot` — public API: components, types, plugins, shape factories. The default plugins (`RenderPlugin`, `SearPlugin`, `GlazePlugin`, `TransformsPlugin`, `PartPlugin`, `InputPlugin`, `SlabPlugin`) auto-register; components register through `Plugin.components`, parse-time metadata via `Plugin.traits`. The orbit camera is opt-in (`OrbitPlugin`, in `extras`)
- `@dylanebert/shallot/extras` — opt-in plugins not in the default set: `lines`, `text`, `tween`, `audio`, `mirror`, `profile` (also reachable on the bare barrel)
- `@dylanebert/shallot/runtime` — platform layer (`now`, `requestFrame`, `readFile`)
- `@dylanebert/shallot/{render,sear,bvh,audio,tween,ecs}/core` + `/glaze` — extension API for custom render producers, compute passes, diagnostics. WGSL constants, surface internals, GPU buffer layouts

Don't deep-import from `src/`. If something you need isn't in a barrel or `*/core` subpath, file an issue.

## ECS & Plugins

### Plugin lifecycle

`initialize(state)` runs BEFORE scene parse — no entities exist yet. Use `warm(state)` for anything that touches scene data. System `setup(state)` is NOT plugin initialize; it runs lazily on the first frame the system runs.

Plugins register meshes/surfaces against `render/core`'s `Surfaces` / `Draws` / `Meshes` (`Registry<T>` instances); registrations accumulate across plugin initializes (no reset). A producer whose per-frame compute writes geometry the renderer reads for *position* declares `before: [PrepassSystem]` (see `render.md` "System ordering").

### Choosing a primitive

- **Marker component + `not()` query** — entity-scoped one-time work
- **Module-level singleton** (`Compute`, `Audio`, `Render`) — process-scoped shared state; populate fields in plugin `initialize`, read via direct import
- **Eid field on a component** — entity-to-entity links. An entity references another by storing its eid in a component field, resolved via `@name` syntax in scenes (`target: @hero`). Scenes are flat — there is no engine-level parent. ReadbackSystem skips attributes containing `@`.

### Anti-patterns

- **Methods on components** — components are data; behavior lives in systems
- **Manager classes that own entities** — use queries with a consumer-shaped relation (eid field) and systems
- **`Map<entityId, ...>` for ownership** — use an eid field on the owning component with marker components
- **`lastState` / `state.exists` defensive guards** — symptoms of cross-State leaking or missed scope
- **Editor systems with `mode: "always"` that add/remove components** — must be non-destructive; update existing fields instead

## UI

DOM UI mounts into **one engine-provided container, sandboxed to the canvas region** — it can never spill into an embedding host (the editor viewport, a host page).

- **One attachment point.** `config.ui(container, state) => () => void` — `run()` creates the container over the canvas and hands it in; mount your UI (any framework) and return a cleanup. A plugin that owns UI (a manifest project has no `config.ui`) calls `mountOverlay(canvas)` for the identical sandboxed container and removes it on `dispose`.
- **Author within the container** — position relative to it; **never `position: fixed`** (escapes to the viewport) and **never `document.body`**. The container is `pointer-events: none`; an interactive panel sets `pointer-events: auto`.
- **The engine guarantees containment** — the container is `contain: layout paint` + `overflow: hidden`, so overflowing or stray-`fixed` UI is bounded + clipped to the canvas region by construction.
- **Anti-pattern: mounting to `document.body` or a self-styled `position: fixed` overlay.** It works standalone but covers the whole window when embedded.

## GPU

Custom render producers and compute passes are normal extension points — register against `render/core` (`Surfaces` / `Meshes` / `Draws`) and run compute on `Render.encoder` from a system.

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
3. Read it back via `readBuffer` / `readFloat32` / `readUint32` from `@dylanebert/shallot`
4. Compare actual vs expected. Apply one fix at a time

Verify via readback BEFORE changing shader code. Off-by-one, wrong binding, stale bind group all look correct in source.

### Pipelines

- Always `createComputePipelineAsync` + `Promise.all`. Sync creation blocks sequentially
- Every pipeline must include a `label` — appears in stats overlay and bench output
- Don't hardcode `bgra8unorm`; use `navigator.gpu.getPreferredCanvasFormat()` (Linux/Dawn returns `rgba8unorm`)
- DXC (Chrome on Windows) doesn't DCE and stalls on large functions inside dynamic loops. Constant upper bounds with dynamic `break` are fine

## Render

### Gamma pipeline

End-to-end gamma-correct; everywhere but the boundaries is linear:

1. **Hex decode** — scene hex colors decode sRGB byte → linear float at parse time (`unpackColor`).
2. **Surfaces output linear** — a surface fs writes a linear `col`; sear returns it verbatim.
3. **Composite encode** — the postfx composite (`GlazePlugin`, or a custom one) `textureStore`s the swapchain, encoding linear→sRGB itself (`LINEAR_TO_SRGB_WGSL`) — the storage swapchain isn't sRGB.

Don't call `linearToSrgb` in a surface fs — the composite does it. A consumer writing its own composite must encode the present gamma itself.

### Content tuning

- Bright accent hex (`0xd49560`) saturates under intense lighting (linear × intensity > 1 collapses to white). Use darker variants (`0x8b6040`) or lower intensity.
- Gamma is non-linear; no global intensity multiplier matches every albedo. Tune for dominant tones; accept drift on saturated-bright accents.

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
