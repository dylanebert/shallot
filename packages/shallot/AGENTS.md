# Shallot

WebGPU game engine. Data-oriented ECS, declarative scenes, plugins compose everything. This file is the working contract for building a game on shallot. The source is the reference: every public export carries JSDoc, and `examples/AGENTS.md` (a sibling of this file) indexes the shipped recipe corpus — grep it for the problem you have, then read that recipe's source, before writing a pattern from scratch. `bunx shallot recipe <name> [dir]` copies a recipe out into a runnable project (bare: lists them).

## Commands

```bash
bun create shallot <name>   # scaffold a project
bunx shallot dev [dir]      # run it (vite, hot reload)
bunx shallot build [dir]    # web build → dist/
bunx shallot build --target <windows|mac|linux> [--portable] [--release]
bunx shallot run [dir]      # build + preview (add --target to build and run native)
bunx shallot verify [dir]   # boot in a headless browser, check it renders, exit 0/nonzero
```

The check is `bunx tsc --noEmit` — run it after every change. `--portable` bundles Chromium (CEF) instead of the system webview; required on Linux, optional elsewhere.

A project is pure data: `shallot.json` (the manifest: scene + plugin enablement) + `public/scenes/*.scene` + plugin modules under `src/`. No index.html, no vite config — the CLI supplies the scaffolding.

## Philosophy

Shallot is data-oriented, ECS, declarative. Code shaped this way composes with the engine; code shaped otherwise fights it.

**Add components and systems, not methods.** New behavior is a new component (data) plus a new system (transform). Not a method on an entity, not a class with state, not a manager. The temptation to write `player.jump()` is the most common consumer antipattern — write a `Jump` event or marker and a system that consumes it.

**Scenes declare; code transforms.** Scene files are the source of truth for entity composition. Imperative entity setup belongs in procedural generation and tests, not the standard load path. If you're writing entity-construction code that mirrors a scene file, use the scene file.

**Systems declare order.** A system declares `after` / `before` other systems (and a `group`); the scheduler topo-sorts. Don't sequence frame work manually.

**One source of truth.** Every piece of data has exactly one authoritative location. Derive, don't duplicate.

**Plugins compose.** Everything is a plugin; nothing is privileged core. Add a feature by adding a plugin, not by modifying engine internals.

**Onion layers — dependencies point inward.** The core is pure data and logic; the outer shell is domain-aware integration. New code goes in the innermost layer it can.

## Imports

- `@dylanebert/shallot` — public API: components, types, plugins, shape factories. The default plugins (`RenderPlugin`, `SearPlugin`, `GlazePlugin`, `TransformsPlugin`, `PartPlugin`, `InputPlugin`, `SlabPlugin`) auto-register; components register through `Plugin.components`, parse-time metadata via `Plugin.traits`. The orbit camera is opt-in (`OrbitPlugin`, in `extras`)
- `@dylanebert/shallot/extras` — opt-in plugins not in the default set: `lines`, `text`, `tween`, `audio`, `mirror`, `profile` (also reachable on the bare barrel)
- `@dylanebert/shallot/runtime` — platform layer (`now`, `requestFrame`, `readFile`)
- `@dylanebert/shallot/{render,sear,bvh,audio,tween,ecs}/core` + `/glaze` — extension API for custom render producers, compute passes, diagnostics. WGSL constants, surface internals, GPU buffer layouts

Don't deep-import from `src/`. If something you need isn't in a barrel or `*/core` subpath, file an issue.

## ECS & Plugins

### Plugin lifecycle

`initialize(state)` runs BEFORE scene parse — no entities exist yet. Use `warm(state)` for anything that touches scene data. System `setup(state)` is NOT plugin initialize; it runs lazily on the first frame the system runs.

Plugins register meshes/surfaces against `render/core`'s `Surfaces` / `Draws` / `Meshes` (`Registry<T>` instances) and must declare `dependencies: [RenderPlugin]` so registration lands after the registry wipe. A producer whose per-frame compute writes geometry the renderer reads for *position* declares `before: [PrepassSystem]`.

### Choosing a primitive

- **Marker component + `not()` query** — entity-scoped one-time work
- **Module-level singleton** (`Compute`, `Audio`, `Render`) — process-scoped shared state; populate fields in plugin `initialize`, read via direct import
- **Eid field on a component** — entity-to-entity links. An entity references another by storing its eid in a component field, resolved via `@name` syntax in scenes (`target: @hero`). Scenes are flat — there is no engine-level parent.

An eid is a borrow, not a durable handle — a rebuild or destroy recycles it. Re-query each frame, or hold the reference in a component field.

### Anti-patterns

- **Methods on components** — components are data; behavior lives in systems
- **Manager classes that own entities** — use queries with a consumer-shaped relation (eid field) and systems
- **`Map<entityId, ...>` for ownership** — use an eid field on the owning component with marker components
- **Module-level runtime accumulators** (`let angle += dt`) — derive from `state.time.elapsed`, so the value survives a rebuild
- **`lastState` / `state.exists` defensive guards** — symptoms of cross-State leaking or missed scope

## UI

DOM UI mounts into **one engine-provided container, sandboxed to the canvas region** — it can never spill into an embedding host page.

- **One attachment point.** `config.ui(container, state) => () => void` — `run()` creates the container over the canvas and hands it in; mount your UI (any framework) and return a cleanup. A plugin that owns UI calls `mountOverlay(canvas, state)` for the identical sandboxed container; passing `state` ties its removal to the State's lifetime.
- **Author within the container** — position relative to it; **never `position: fixed`** (escapes to the viewport) and **never `document.body`**. The container is `pointer-events: none`; an interactive panel sets `pointer-events: auto`.
- **Cleanup means real unmount, registered on the State.** Tie teardown to the State: `state.onDispose(fn)` runs `fn` at `state.dispose()`, and `state.signal` passed as `{ signal }` to `addEventListener`/`fetch` detaches with no removal code. The cleanup you return from `config.ui` registers the same way. Whatever you register must unmount what it mounted — Svelte `unmount()`, React `root.unmount()` — and cancel any rAF/interval the UI started; removing the host DOM alone leaves a framework component's effects running. `onDispose` fires only on `state.dispose()`, so if you mount from `warm` (which re-runs on an in-place rebuild with no `dispose` first) also clear the prior mount at the top of `warm` so it can't stack.
- The engine guarantees containment (`contain: layout paint` + `overflow: hidden`), so overflowing UI is clipped to the canvas region by construction.

## GPU

Custom render producers and compute passes are normal extension points — register against `render/core` (`Surfaces` / `Meshes` / `Draws`) and run compute on `Render.encoder` from a system.

### Binding limits

`maxStorageBuffersPerShaderStage` is **10 — hard ceiling.** 99.6% of devices support 10; only 64% support 16. Requesting more rejects `requestDevice()` on a third of users. Per shader stage across all bind groups — splitting groups doesn't help. Both `storage` and `read-only-storage` count.

When you hit it, don't silently exceed (Chrome fails with no diagnostics). Consolidate: interleave same-pass buffers into one struct, fold scalars into a related buffer's header, block-concatenate CPU uploads. Never add per-entity CPU iteration to save a binding.

### Debug methodology

Shaders can't print. The only way to know what a shader computed is write-to-buffer + readback.

1. Pick the exact value you're uncertain about ("`body.pos.y` for entity 0 after primal", not "is broadphase working")
2. `atomicStore(&debug[SLOT], bitcast<u32>(value))`
3. Read it back via `readBuffer` / `readFloat32` / `readUint32` from `@dylanebert/shallot`
4. Compare actual vs expected. Apply one fix at a time

Verify via readback BEFORE changing shader code. Off-by-one, wrong binding, stale bind group all look correct in source.

### Pipelines

- Always `createComputePipelineAsync` + `Promise.all`. Sync creation blocks sequentially
- Every pipeline must include a `label` — appears in the stats overlay and bench output
- Don't hardcode `bgra8unorm`; use `navigator.gpu.getPreferredCanvasFormat()`
- DXC (Chrome on Windows) doesn't DCE and stalls on large functions inside dynamic loops. Constant upper bounds with dynamic `break` are fine

## Render

### Gamma pipeline

End-to-end gamma-correct; everywhere but the boundaries is linear:

1. **Hex decode** — scene hex colors decode sRGB byte → linear float at parse time (`unpackColor`).
2. **Surfaces output linear** — a surface fs writes a linear `col`; sear returns it verbatim.
3. **Composite encode** — the postfx composite (`GlazePlugin`, or a custom one) encodes linear→sRGB itself (`LINEAR_TO_SRGB_WGSL`).

Don't call `linearToSrgb` in a surface fs — the composite does it.

### Content tuning

- Bright accent hex (`0xd49560`) saturates under intense lighting (linear × intensity > 1 collapses to white). Use darker variants (`0x8b6040`) or lower intensity.
- Gamma is non-linear; no global intensity multiplier matches every albedo. Tune for dominant tones; accept drift on saturated-bright accents.

## Physics

Physics is opt-in — `TumblePlugin` (main barrel) plus the `Body` / `Spring` / `Joint` components author it, and `Tumble.world` is the escape hatch for constraints past the substrate. When you wire a joint by hand after the bodies marshal, **spawn the jointed bodies non-overlapping**: the physics ticks between body creation and the wire mint a persistent contact that fights the joint from then on, so a motor pulling two concentric bodies together stalls — author them apart, or anchor a driven body to ground it does not overlap.

## Testing

- **Unit tests** (`bun test`) — fast, hardware-invariant logic. Catches structure, layout, math errors. **Never bind a GPU device in a unit test** — software adapters flake by construction.
- **Real GPU** (Playwright) — pipeline validation, compile, raster, readback. Unit tests alone miss real hardware failures.
- **`.test.ts`** — spec tests. First principles, tight tolerances, permanent. **`.lab.ts`** — investigation, not auto-run, temporary.
- **Tolerances are derived, not tuned:** exact invariants → 1e-10; f32 precision → ~1e-6 relative; truncation and convergence → derive from order, step size, iteration count. If you can't derive it, investigate in a `.lab.ts` first.
- **GPU timestamps, not FPS.** `requestAnimationFrame` measures CPU; `queue.submit()` returns immediately. GPU timestamp queries measure actual hardware execution.

## Build, run, verify

`shallot verify [dir]` boots the project in a real headless browser, waits for it to render, and exits 0 on pass / nonzero on fail — a self-terminating gate for an agent (or CI) to prove its own work. No dev server left running, no browser tab to close.

**Cross-origin isolation.** Every serve surface (`shallot dev`, `shallot run`'s preview, `verify`'s boots) sends COOP/COEP headers so tumble physics can multithread — a browser grants shared memory only to a cross-origin-isolated page. The tradeoff: a cross-origin subresource must be CORS-approved or carry a CORP header. A plain `<img>` or no-cors fetch from a host that sends neither is blocked on these servers; serve the asset from `public/` instead, or use a CORS-enabled host. A static host that can't set headers (GitHub Pages) still works — physics falls back to single-threaded with one console log.

```bash
bunx shallot verify                    # boot the dev server, check the scene renders
bunx shallot build && bunx shallot verify --dist   # verify the shipped build instead
bunx shallot verify --screenshot out.png --query scenario=fall
```

Playwright is optional — install it once: `bun add -d playwright && bunx playwright install chromium`. It's never a dependency of your app; `verify` finds it in your project and exits with a distinct code (naming the install command) if it's absent.

**Default readiness.** With no `window.__harness`, `verify` passes when the canvas booted, rendered a settled non-blank frame, and threw no page errors — the whole check needs no code in your project.

**Assert something specific.** Install `window.__harness` to have `verify` drive your own pass/fail instead of the render check. `installHarness` sets it up (`ready` flips true once a frame draws, `read(eid)` returns a live entity pose — physics pose for a `Body`, else its `Transform`); replace `run` to assert:

```ts
import { installHarness } from "@dylanebert/shallot/harness";

const app = await run({ scene });
const harness = installHarness(app.state);
harness.run = async () => {
    const box = harness.read!(boxEid)!;           // where did it end up?
    const fell = box.pos[1] < 0.5;
    return { ok: fell, checks: [{ name: "box fell to the floor", ok: fell }] };
};
```

`verify` waits for `ready`, calls `run()`, and its exit code follows the `Verdict.ok` (plus zero page errors). `--json` emits the full result — `checks` and any extra fields your `Verdict` carries pass through. `--query k=v` sets URL params (and mirrors into `run`'s opts).

**In a manifest project** (`shallot.json` + plugins, no `run({ scene })` of your own — the CLI runs the project), install from a plugin hook. `initialize(state)` runs before the scene parses, so it pins the harness before a frame can settle; resolve entities inside `run`, where they already exist:

```ts
import { installHarness } from "@dylanebert/shallot/harness";
import { Body, type Plugin, type State } from "@dylanebert/shallot";

const Verify: Plugin = {
    name: "Verify",
    initialize(state: State) {
        const harness = installHarness(state);
        harness.run = async () => {
            const box = [...state.query([Body])][0]; // the falling body
            const fell = harness.read!(box)!.pos[1] < 0.5;
            return { ok: fell, checks: [{ name: "box fell", ok: fell }] };
        };
    },
};
export default Verify;
```

Add it to `shallot.json` (`"Verify": "./src/verify"`) while you check, and remove it after.

**Verify persistence in one run.** `run(opts)` receives the `--query` values, so you can seed the saved state and assert the restore path without a real reload. `bunx shallot verify --query color=blue` → `run({ color })` writes `localStorage` the way a prior session would, drives the project's restore path, and asserts the restored value came back — reload-persistence proven in a single invocation, no dev server.

**Installing late?** A defined `window.__harness` always wins over the render check, but only once it exists — a static frame can settle (and conclude the render check) before your app finishes a slow build or a self-reload and installs it. Pin the harness path up front: set `window.__harness = { ready: false }` first thing, then install the real one when ready.
