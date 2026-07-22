# Shallot

WebGPU game engine. The repo is the documentation: readable source with JSDoc contracts on every public export, a problem-indexed examples corpus, and this file plus `.claude/rules/` as the behavioral contract. There is no docs site and no editor — agents and humans both read the source, the examples index (`examples/AGENTS.md`), and `packages/shallot/AGENTS.md` (the consumer-facing contract, shipped with the npm package).

For consumer-facing patterns (how to build games on Shallot — ECS, plugins, GPU, render, physics, testing), see `packages/shallot/AGENTS.md`. This file covers engine-internal layout and commands. Behavioral constraints live in `.claude/rules/`.

**Layout:** `packages/shallot/` (engine — `src/engine/`, `src/standard/`, `src/extras/`, `src/document/` (the scene document model: `Document`/`History`/`Session`), `src/project/` (the manifest/generate/vite project toolchain behind the CLI)), `packages/create-shallot/` (the `bun create shallot` scaffold), `examples/` (standalone projects).

Code is source of truth — make it elegant first, document what's non-obvious.

---

## Architectural reference points

**Bevy is the structural reference for ECS, plugin layout, and frame-graph shape.** Take it where it earns its place; skip the parts that exist for Bevy's scale and constraints, not ours.

**Take from Bevy:**
- ECS-component-first, data over inheritance (already the foundation)
- Plugin shape: `name` + lifecycle hooks + dependency declarations
- Frame-graph as named resource publish/subscribe
- Typed, *closed* resource unions (e.g. `Image` bundles texture + view, like `GpuImage`)
- Kind-tagged values at access time

**Skip from Bevy:**
- Separate `RenderApp` / extract schedule. Shallot runs sim + render in one `State`; that's intentional, not an oversight
- `Assets<T>` + `Handle<T>` reference counting. Frame-graph slots aren't assets
- `Box<dyn Reflect>` plugin-extensible type unions. Closed unions beat open at single-author scale; add a variant when a real consumer needs it
- Manual `add_slot_edge` graph topology. Auto-wire from input/output name matches; the existing inference is simpler and works
- Macro-heavy registration, generic-saturated query DSLs, schedule/system ordering DSLs
- **Multi-threaded executor + access-conflict scheduling.** Single-threaded JS, GPU-bound — no parallelism opportunity on the CPU side. Most of Bevy's ECS bulk exists to make this safe; we don't pay the cost
- **`ChangeDetection` ticks / dirty tracking.** Directly contradicts gpu.md's firehose principle. Tracking which entities changed *is* the antipattern
- **Generational `Entity` / mandatory recycle version.** The eid stays a bare index; membership is the liveness signal. A recycle version, if a held-reference consumer ever earns it, is an opt-in side array, never packed in the eid. Detail in `ecs.md`
- **Auto-inserted `apply_deferred` sync + `Commands` deferred mutation.** Hidden inter-system behavior taxes debugging; immediate mutation is fine in single-threaded JS
- **Typed `SystemParam` chains (`Res<T>`, `Query<T>`, custom params).** The type ceremony taxes authoring; module-level singletons + `state.query([A, B])` are the right size for shallot's scale and trip count

**The decision rules.** Two axes when evaluating any Bevy-style upgrade:

1. **Iteration speed + performance.** Is the feature solving a problem shallot has, or one of Bevy's? Most of Bevy's mature ECS apparatus addresses multi-threaded CPU parallelism with safe deferred mutation — constraints shallot doesn't have. Adopt only when a feature compounds across many systems and reduces noise without taxing authoring or hiding behavior.
2. **Layer churn.** Which layer iterates? Substrate stable + pipeline iterates → take the strict typing (typed graph slots, closed resource unions). Substrate iterates → keep it loose.

When something doesn't have a clean Bevy analogue (e.g. shallot's GPU-driven physics, on-GPU graph coloring, fixed-cap SoA component storage), don't invent one. Shallot's "structurally Bevy" ends where the WebGPU-on-integrated-GPU floor, the procedural-first commitment, or the iteration-speed posture forces a different shape.

The minimal scheduler/ECS isn't a TODO — it's a deliberate shape that compounds with TS hot reload and small mental model to give shallot's iteration speed. Bevy 0.16 absorbed the render graph into its schedule because the schedule had grown to ~14.5k lines of typed-param + parallel-executor + auto-sync machinery; shallot's scheduler is 309 lines doing a different, smaller job. The engine runs on plain ECS systems (no compute graph): each camera binds 1:1 to a canvas (a `View` in the `Views` map); `BeginFrameSystem` acquires every view's swapchain texture, renderers attach via `after: [BeginFrameSystem]` and draw directly into each camera's framebuffer, async pipeline compile happens in each plugin's own `warm()`. No compositor, no offscreens; multi-view = multi-canvas.

---

## Platform support floor

Single modern WebGPU feature floor. No conditional fallback paths. Conditional paths compound maintenance cost for one author — every change covers both paths, the slow path bitrots silently because nobody runs it.

**Base floor (every app):** WebGPU 1.0 + `shader-f16` + `timestamp-query` + `indirect-first-instance` + `bgra8unorm-storage` + `rg11b10ufloat-renderable` (the default HDR offscreen + sear's MSAA color target — halves the resolve bandwidth vs rgba16float) + texture compression (one of BC / ETC2 / ASTC). The default renderer + slab substrate need these, so they hold for any Shallot app.

**Plugin-declared features:** a plugin declares what it needs beyond the base floor as `Plugin.features` (required — a `GPUFeatureName[]`) or `Plugin.preferredFeatures` (best-effort). `build()` unions each across the active plugins and passes both to `acquireDevice`, which requests `base ∪ required` (a missing required one throws `UnsupportedError` naming it, before any plugin loads) and adds each preferred only where the adapter has it. `subgroups` is the standing **preferred** case: only the BVH builder (physics broadphase, accel structures) uses subgroup ops, and it has an LDS fallback arm (`createBvh` reads `device.features` to pick), so `AvbdPlugin.preferredFeatures = BVH_FEATURES` (from `bvh/core`) — a `subgroups`-less device (WebKit) still loads physics, on the slower LDS arm. A physics-free app requests neither.

**Targets:** native builds pick a backend by mode. The default is the platform's **system webview** via wry — WebView2 (Windows), WKWebView (macOS), WebKitGTK (Linux) — small but host-dependent. `--portable` bundles the **Chromium runtime (CEF)** instead — larger, self-contained, runs anywhere. WebView2 is full Chromium (every feature); WKWebView meets the base floor but lacks `subgroups` (audited 2026-06-19, Safari 26.5 / Apple Silicon: full floor, only `subgroups` absent; WKWebView shares WebKit's WebGPU/Metal backend) — so a macOS physics app runs the LDS broadphase arm there, `--portable` for the faster subgroup arm, no longer a hard requirement. WebKitGTK has no usable WebGPU, so a default Linux build reaches the diagnostic tier — Linux needs `--portable`. `shallot build` runs a build-time check (`bin/features.ts`) that **warns** (never blocks) when the chosen backend can't satisfy a project's *required* features (subgroups, being preferred, never triggers it). Also: Steam Deck (RDNA2), Chrome / Edge on desktop, recent Android Chrome, Safari 26+ on Apple Silicon (the WebKit floor above; end-to-end render still to validate, Intel-Mac unaudited). Firefox and pre-Gen11 Intel iGPUs sit in the diagnostic tier until they ship the floor.

**Diagnostic tier:** boot, display unsupported-configuration message, exit. Not a degraded path — a clear boundary.

No conditional *runtime* fallback within a *required* feature — a plugin has its required features (base floor ∪ `Plugin.features`) or fails loud. The one sanctioned fallback site is a *preferred* feature (`Plugin.preferredFeatures`): a fast arm where present, a fallback arm where absent — the BVH builder's subgroups→LDS (gpu.md). Assume the base floor + your required features are present; don't gate or write a fallback for those. Use features behind narrow interfaces (codec module, helper chunk) so a spec shift is a contained edit.

---

## Commands

```bash
bun test                                           # Fast unit tests (bun-webgpu) — excludes the .oracle.ts physics-oracle tier
bun run test:oracle                                # The f64 AVBD physics oracle (tests/avbd/*.oracle.ts) — slow, deterministic, run separately
bun run test:full                                  # test + test:oracle — the complete gate before a commit / PR
bun bench [--scenario <name> --seed --count --warmup --frames --param k=v --screenshot <path>]  # Gym scenario via `shallot verify` on a real device (default: render). Scenarios: render (forward-pipeline atom, `--param mode=`-selected), sprite (2D/billboard path), sat (GPU-SAT codegen gate), accel, chain (phase-boundary microbench), stress (bottleneck-saturation atom), the three physics scenes — pile, constraints, character — and backend (the substrate swap gate, `--param backend=tumble|avbd`). --screenshot writes a post-run canvas PNG (visual validation — checks gate numbers, not pixels)
bun run scripts/physics-bench.ts                    # AVBD physics perf + scaling sweep (drives the gym pile scenario + constraints/character rows)
bun check                                          # Format + type check (Biome + tsc)
bun run format                                     # Biome (.ts/.js/.json) + scene formatter
bun run build                                      # All Rust artifacts (WASM + native window)
shallot dev [dir]                                  # Run the project standalone (vite HMR over its shallot.json; native --target = debug build + run)
shallot build [dir]                                # Web build (Vite → dist/)
shallot build --target <os> [--portable] [--release]  # Native build; <os> = windows | mac | linux. Default = system webview (WebView2 / WKWebView / WebKitGTK); --portable = CEF. Linux needs --portable; warns (never blocks) on mismatch
shallot run [dir] [--target <os>] [--portable]     # Build + run (web preview, or native; windows via WSL→Windows)
bun local [name]                                   # Scaffold local test project with packed engine (manual poking)
bun run test:install                               # Real-install gate: pack engine + a plugin lib, bun install, assert build/dev/create flows
bun run flows [--flow <name>]                      # Standalone-app engine flows — ejected apps under `examples/flows/` driven by `shallot verify` (survive-reload, ui-containment)
bun run recipes [--recipe <name>]                  # Physics recipes' dynamics smoke — each ported physics recipe's `window.__harness` asserts its concept's observable (platform slides, rotor spins, joint breaks, …), driven by `shallot verify`
```

The verification vehicle is the **shipped gate**: `shallot verify [dir]` (`packages/shallot/bin/verify.ts`) boots a project in a real headless browser, waits for it to render (or drives the `window.__harness` a project installs — the published protocol on `@dylanebert/shallot/harness`), reads a pass/fail Verdict, and exits 0/nonzero. There is no repo-private harness tier: `bun bench` and `bun run flows` are thin wrappers in `scripts/` over that CLI. `scripts/bench.ts` maps its arg surface onto `shallot verify examples/gym --json` and formats the profiler metrics + checks + memory; `scripts/flows.ts` drives the ejected flow apps under `examples/flows/` (survive-reload, ui-containment). One page load = one verdict.

Gym is the single real-device surface: `bun bench` drives `examples/gym` (default `render`), prints the measure, and gates on the scenario's checks (failure exits nonzero). It holds both the permanent param-driven regression atoms and the in-flight scenarios that run + render before they earn a permanent gate (an `assert` is optional until then) — maintained like the test suite, not held perfectly clean. The gym contract is `examples/gym/src/gym.ts`: a `Scenario` is `params` + `build` + optional `assert` + `live`. `params` is the single source of truth for the scenario's tunables — the URL parses them, `bun bench --param key=value` sets them (they ride `shallot verify --query`), and the live top-right control panel auto-renders from them (a `rebuild` knob reloads, a live knob mutates in place). Adding a scenario is a new file + a `scenarios/index.ts` import. **Assert readback is `Mirror`** (not the legacy `compute/readback`); **timing is the profiler** (`window.__benchmark`, GPU passes incl. `part:pack` / `bvh:sort` / `bvh:build` / `bvh:trace` — the source of truth, don't hand-roll a CPU measure). A scenario has **no environment awareness** — the same page runs headless or in a tab; a scene carries an orbit camera you drive live, while headless leaves it at its deterministic start so the `assert` verifies (e.g. by varying `Camera.far`, orbit-independent). F3 toggles the stats panel. The scenarios and the GPU-driven coverage each carries are enumerated in the `examples/gym/src/scenarios/index.ts` barrel header — the single home for that list. Gym installs the published `window.__harness` (`ready` + `run`); `gym.ts` translates a scenario's internal verdict to the published wire `Verdict` (`ok` + checks, metrics riding through as an extra) that `shallot verify` reads.

### Verification

Run `bun run format`, `bun check`, `bun test` before completing work (`bun run test:full` to also run the slow f64 physics oracle — the complete gate before a commit / PR). `cargo test` after Rust audio changes (run from `packages/shallot/rust/audio`). `bun bench` required after GPU code changes. `bun run flows` after changes to the serialize/restore path, `config.ui` / `mountOverlay`, or the `shallot dev` server — the standalone-app Playwright flows (display-gated, self-terminating; run alone, never concurrent with other heavy work). `bun run recipes` after changes to a physics recipe or the substrate/tumble path it exercises — the physics recipes' dynamics smoke (display-gated, same shape as flows). On WSL these three run for real against the Windows host's GPU via `scripts/wsl-bridge.ts` (a host-launched browser server + reverse tunnel, driving the verify CLI as `--connect`); they skip only when the host lacks the node/bun the bridge provisions with. `bun run test:install` after packaging / CLI / manifest-resolution / asset-shipping changes and after `packages/create-shallot` changes — the dev symlink hides real-install bugs (it caught a devDep import that broke `shallot build` for every installed user).

---

## Examples

Four groups under `examples/`, indexed by `examples/AGENTS.md` — one line per entry: the problem, the path, what it shows. The index is the retrieval surface; grep it before writing a pattern from scratch. The corpus contract (what an entry must be) is `.claude/rules/examples.md`. Every tier runs through the same shipped gate (`shallot verify`); none reaches into repo tooling.

- `recipes/` — the **teaching** tier: one minimal project per problem a game developer actually has (first-person character, physics playground, import a model), named by the problem, compile-gated by `bun check`. Keeps the engine's usage truth honest — a renamed export fails the corpus typecheck.
- `gym/` — the **testing-harness** tier: param-driven atoms that are a correctness gate + benchmark + live demo at once (the triple-duty bar — one scene, no branching; `pile` is the model), one per subsystem path; the code is a harness, not a teaching reference. One project, `?scenario=`-selected. It's the **targeted real-device tier** — `bun bench --scenario X`, one atom run after its domain changes, opposite `bun test`'s hardware-invariant run-all (the split is `testing.md`). A lab scenario migrates here case-by-case, **folded** into the atom whose subsystem it exercises (a new param/mode), never mechanically file-moved. Keeps the engine honest about itself.
- `flows/` — the **standalone-app engine flows**: ejected apps that exercise engine behavior a `bun test` can't reach — `survive-reload` (a real reload through serialize/restore) and `ui-containment` (`config.ui` overlay clipping). Driven by `bun run flows` (`scripts/flows.ts` over `shallot verify`).
- `showcase/` — the **capability** tier: richer exhibits showing how to do something interesting (the voxel editor), one project per subdir. Each is a self-contained real project that **owns and dogfoods its own testing** — a gate written against the published `@dylanebert/shallot` surface + the project's own driver (bring-your-own Playwright, or `shallot verify`; `voxel`'s `src/gate.ts` + `test/voxel.spec.ts` are the worked example), never reaching into repo `scripts/`. Standalone "wow".

The hello path is the scaffold: `bun create shallot` (`packages/create-shallot/index.ts`, the single source — no committed starter copy) emits a minimal project with its own CLAUDE.md/AGENTS.md pointing back at the engine's agent surface. `bun run test:install` gates the create flow.

`gym` is a single project with `?scenario=`-selected scenarios; `recipes` + `flows` + `showcase` hold one project per subdir. A manifest project is pure data (`shallot.json` + plugin modules + `public/`) run through the CLI: `shallot dev examples/recipes/<entry>/` — there is no per-project `bun dev`. The ejected exceptions own their own `index.html` (+ vite): `gym` and `showcase/visualization` run with `cd examples/<project> && bun dev`; the `flows/` apps are verify-only (`shallot verify` boots them, no standalone `bun dev`).

Every example must include `public/icon.svg` (the shallot icon); the examples that own an `index.html` (gym, `showcase/visualization`, the `flows/` apps) add `<link rel="icon" type="image/svg+xml" href="/icon.svg" />` there, while every manifest project has none — the `shallot dev` server and the synthesized `shallot build` entry supply it. For native builds, `public/icon.png` becomes the window icon; if absent, the default shallot icon is used. Always `dispose()` State on HMR/unmount — without it, each hot-reload stacks another State + RAF loop.

**UI convention.** App UI mounts into one engine-provided, canvas-bounded, sandboxed container — the full contract (`config.ui` / `mountOverlay` / never `position: fixed` / containment) lives in `packages/shallot/AGENTS.md` "UI". Examples follow it. **The one exemption:** an *ejected* example that owns its full page and is never embedded (`gym`, `showcase/visualization` — they own `index.html` + their own vite) may own the viewport directly with `position: fixed`. Complex example UI owns its page with Svelte — add `svelte` + `@sveltejs/vite-plugin-svelte` + a `svelte.config.js` to that example's own package, mount via `svelte`'s `mount()`/`unmount()`.
