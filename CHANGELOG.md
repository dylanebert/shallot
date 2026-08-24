# Changelog

Newest first. **Breaking:** marks a change that needs consumer action; [`packages/shallot/MIGRATION.md`](packages/shallot/MIGRATION.md) is the 0.8→0.9 port. Versions follow [semver](https://semver.org).

## 0.9.4 — 2026-08-24

Three paths no gate had ever run are now covered: the bare barrel imports under a runtime with no WebGPU globals, the release workflow can attach its own assets without a hand recovery, and a `verify` red carries the pixel measurement behind it. No consumer action, nothing breaking.

- **packaging** — the bare barrel (`@dylanebert/shallot`) no longer touches `GPUTextureUsage` at module scope: the reads in `COLOR_LANES`'s `usage` field are deferred to first access, so a runtime that defines no WebGPU globals no longer dies at import on `GPUTextureUsage is not defined`. That was all of 0.9.3's browser-only note. What is left has nothing to do with WebGPU: plain `node` still cannot load the barrel's raw TypeScript out of `node_modules`, because node applies no type stripping there. `check-node-import` now imports the barrel under bun beside its existing Node subject.
- **release** — the release workflow's asset-attachment step sets `GH_REPO: ${{ github.repository }}`, so `gh` resolves the repo from the environment rather than a local clone the job never has. Both branches of the step (`gh release create --verify-tag` and `gh release view`) were broken by the same missing context — `gh` verifies the tag against the remote, not a checkout — so one env line restores both while keeping `--verify-tag`. A static arm in `prebuilt.test.ts` gates the shape: any job that invokes `gh` with no `actions/checkout` must set `GH_REPO`.
- **internals** — `shallot verify`'s `Result` now carries a render probe: samples taken, the last `center`/`corner` RGB, the spread against its threshold, the elapsed wait, and how the wait concluded. `--json` carries it. The repo's own install gate prints it on a red instead of the bare `[]` an empty error array used to produce, and a red with no page diagnostic now names the failing predicate (`pass`/`booted`/`rendered`) beside the result's field values, so it can no longer be dismissed as a flake. Arms in `bin/verify.test.ts` pin the branch content over synthetic settle states.
- **internals** — the TGSL metadata assertion that reds on the install gate is repaired at the format version: the stale `externalNames` term (a V1 field no supported version emits) is retired for a metadata-format-version assertion, so a future rename reds under its own name rather than as a mystery. The surviving `TRANSPILED` shape term is untouched.
- **known red** — on Apple Metal, `shallot verify` intermittently reports a blank render: `booted` passes, `rendered` fails, and the render probe reads `spread=0` on a uniform frame until the settle budget expires. It reproduces against both a manifest project and an ejected Vite app, so an ejected recipe is not the cause. Open. The probe above is what records it, and `--json` carries the reading.

## 0.9.3 — 2026-08-21

The packaging patch: prebuilt native shells on release builds, a loading subpath that imports under Node ≥26, and an audit pass over the whole export surface.

Two of those audit changes are breaking on published subpaths — `./utils/core` renamed two shader helpers, and `./document` dropped its tree machinery — and they ship knowingly in a patch rather than as a minor. Both are named in plain terms below; neither has a known consumer, and neither is aliased.

- **cli** — `shallot build --target windows|mac|linux --release` downloads a precompiled `shallot-window` shell from that version's GitHub release, SHA256-verified against the same release's `SHA256SUMS`, so a native release build needs no Rust toolchain on a cache hit. Any miss — no asset, offline, checksum mismatch, source checkout — falls back to compiling from source, and a debug build always compiles. Six archives ship per release, named `shallot-window-<target>-<mode>.tar.gz` over windows/mac/linux × `system` and `portable`.
- **cli** — `shallot verify` refuses a software adapter before any check runs: SwiftShader, llvmpipe, lavapipe, WARP, "basic render" and a missing adapter each exit 4 with a diagnostic. Through 0.9.2 a software adapter cleared the feature floor and then crashed mid-run, which reads as a broken project rather than a missing GPU.
- **packaging** — the `standard/loading` module imports the package manifest with `with { type: "json" }` and reads `version` off its default export, so importing that subpath under Node ≥26 no longer dies at load. Node rejects an attribute-less JSON import and exposes only a default export, where bun tolerates both, so the defect was invisible to every bun-side gate; a spawned real-`node` import check now guards it. The bare barrel is still browser-only — it touches `GPUTextureUsage` at module scope.
- **packaging** — `shallot.json` takes an optional `identifier`, which a mac build uses as its `CFBundleIdentifier`. The default is now `com.shallot.<project directory>`, not `com.multiplekex.<name>`: a mac app that shipped under the old default installs as a different application unless it sets `identifier` explicitly.
- **utils (breaking)** — `LINEAR_TO_OKLAB_WGSL` and `OKLAB_TO_LINEAR_WGSL` on `@dylanebert/shallot/utils/core` are gone; compose raw WGSL with `linearToOklabWgsl()` and `oklabToLinearWgsl()` instead. They were the last raw-constant holdouts among the relocatable shader helpers, and `MIGRATION.md` promised they would not be renamed — that promise is now corrected in the guide rather than kept.
- **document (breaking)** — `@dylanebert/shallot/document` (and the bare barrel, which re-exports it) drops the parent-tree machinery: `add`, `remove` and `reorder` lose their leading `parent` argument, `Document.reparent` is removed, and the `Command` union's reparent variant with it. A document's nodes are a flat list.
- **showcase** — `examples/showcase/roads` is a new procedural-terrain exhibit in the repo: a seeded-FBM heightfield with a road cut into an earthwork corridor, a 2D overlay rasterizer for the network, and an interactive reseed. It runs on a real GPU under its own device gate. Showcases stay in the repo; the tarball ships the recipes.
- **scaffold** — `bun create shallot` pins `typegpu@~0.12.0` and `unplugin-typegpu@~0.12.1`, matching the peer range 0.9.2 moved to. A project scaffolded from 0.9.2 installed the 0.11 minor and broke at pipeline warm.
- **fixes** — six audit passes over the package, whose behavioral half a consumer can observe: `f16` encoding saturates a finite overflow to infinity, matrix `invert` zeroes its output on a singular input, the fixed-step scheduler caps steps on the post-scale accumulator, `State.create` rolls back an over-capacity entity, `Player` and `Text` dispose by identity, off-canvas pointer motion still tracks, and `Document.addAttr` on an attribute that already exists coalesces into `setAttr`, so undo restores the old value instead of deleting the attribute. Scene discovery (`discoverScenes` on `./vite`) catches per directory rather than around the whole walk: an unreadable subtree now warns and is skipped, where it used to truncate the scene list silently.
- **internals** — the rest of the audit work moves nothing a consumer imports by name: roughly thirty zero-consumer exports are deleted or made private (reachable only through the `./src/*` wildcard), duplicated stride constants are re-derived from their schemas, and GPU debug labels read `shallot`.
- **internals** — the export surface is gated rather than reviewed: `check-exports` walks the transitive re-export closure of the `exports` map and fails on an exported symbol with no consumer and no place on the public surface, and an attended end-to-end check proves the prebuilt download path builds with `cargo` unavailable.

## 0.9.2 — 2026-08-14

The TypeGPU 0.12 bump, and native builds that work from a standard npm install.

**Breaking:** custom GPU consumers move to `typegpu@~0.12.0`, `unplugin-typegpu@~0.12.1`, and `eslint-plugin-typegpu@~0.12.0`. Shallot's own API is unchanged (no export moved, no signature changed), so the port is the install line unless your own code calls TypeGPU directly. TypeGPU's own 0.12 migration guide covers that case.

- **gpu** — the TypeGPU peer moves to the 0.12 minor. 0.12 removes `layout.bound`, and the dereferenced `layout.$.x` throws outside an actual TGSL body, so a raw-WGSL surface that bound per-field externals through `$uses` now passes the whole `$` proxy as one external and lets the WGSL text's own dot chain defer the field read to resolution time. `standard/sear/pipelines.ts` and `extras/gltf/live.ts` carry the pattern.
- **gpu** — `eslint-plugin-typegpu` 0.12 correctly stops flagging an unsigned shift confined to a ternary's folded-away CPU arm, so the ten `eslint-disable typegpu/no-unsupported-syntax` directives over the `utils/tgsl.ts` and `bvh/bounds.ts` dual-shape helpers are gone.
- **cli** — `shallot build --target windows|mac|linux` works from a standard install. The `rust/window` host ships as crate source and compiles on first build, so the prerequisite is the Rust toolchain plus the target's system dependencies (the repo README has the per-target table), not a Shallot source checkout. A crate that isn't there reports a corrupt install rather than a raw `ENOENT` out of cargo.
- **docs** — the npm README's install block omitted the required TypeGPU peer and the single `unplugin-typegpu` transform, an install that broke at pipeline warm. `verify`'s optional playwright peer is named in both READMEs, npm-relative links resolve against the package rather than the repo root, and `AGENTS.md` stops listing `audio` and `mirror` under `/extras`, which are bare-barrel only.

## 0.9.1 — 2026-08-12

The packaging patch, and the recommended target for a 0.8→0.9 port. The two Node-consumed exports ship compiled, the duplicate-TypeGPU check catches a same-version double-load, and `Profile` grows the GPU byte and pipeline counters a budget gate can read.

- **packaging** — `./vite` and `./harness/browser` resolve to compiled `dist/*.js`, with types still coming from source. A `vite.config.ts`, a `playwright.config.ts`, or a plain Node script can now import `projectPlugin` and `REAL_GPU_LAUNCH` directly; through 0.9.0 that threw `ERR_UNKNOWN_FILE_EXTENSION`, because Node applies no TypeScript transform to a `node_modules` import. A linked dev checkout builds them once (`bun run --cwd packages/shallot scripts/build-tooling.ts`, which `bun pm pack` runs via `prepack`): the `default` condition points at `dist/` with no source fallback.
- **gpu** — the duplicate-TypeGPU check counts writes to `__TYPEGPU_VERSION__` instead of comparing its value, so two copies of the same pinned minor no longer read as one. TypeGPU stamps that key on every module evaluation, which a value comparison cannot see; the shape that motivated it is a consumer's own direct `typegpu/data` import in a zero-config registry install.
- **gpu limits** — `checkTgsl`'s JSDoc now records what the check still cannot catch: a duplicate whose write lands before the counter installs, and a metadata-free bundle whose canary resolves anyway. The real-device install gate is the backstop for both.
- **profile** — GPU byte totals and honest pipeline accounting on the public seam: `bufferBytes`, `textureBytes`, per-label `allocBytes`, and `lazyBytes`, plus `compiledPipelines` and `pipelineCalls` beside `compile`, which now keys by descriptor label rather than scope name. An allocator declares a lazily-grown pool entry with `LazyAlloc` (`{ lazy: true }` on the create descriptor) so a byte budget can exclude what real GPU backpressure grows rather than inferring it from a label string.
- **verify** — `--timings` reads the exact in-page `__harness` install moment instead of quantizing it to a 500 ms poll, and adds a resource-timing readout (request count, summed transfer duration, slowest N) with a raised buffer, so a symlinked workspace dependency's unbundled modules cannot saturate the spec's 250-entry default and flatten the number. Stdout flushes before exit, so a truncated pipe no longer reports as a crash.
- **cli** — `--target windows|mac|linux` says what it needs. Native builds run from a Shallot source checkout, since the rust crate they link is not in the npm package; `bunx shallot build` from an installed package is web.
- **internals** — the sky's star hash and the glTF PBR path each shipped a second, dead implementation of code the live path already had, and the scene codec's attribute parser carried an unreachable branch. All three are gone, each with the check that keeps it gone. The fountain and voxel showcase gates skip on a software adapter instead of crashing.

## 0.9.0 — 2026-08-03

The GPU substrate is now TypeGPU end to end: one schema defines each CPU↔GPU layout, engine-authored shaders are TGSL, and GPU logic can run on the CPU for exact unit tests. See [`packages/shallot/MIGRATION.md`](packages/shallot/MIGRATION.md) for the complete 0.8→0.9 consumer port.

**Breaking:** custom GPU consumers install `typegpu@~0.11.9` and run exactly one `unplugin-typegpu` transform. Handwritten binding structs, shader strings, and the old `Surface` / `Background` registration contract become schema-backed typed layouts and TGSL functions. Raw WebGPU interop remains available at integration boundaries.

- **gpu** — `requestGPU` adopts Shallot's floor-enforced device into a device-scoped TypeGPU root; typed buffers, bind groups, pipelines, and schemas now carry the layout contract through rendering, BVH, AVBD, and the shipped compute examples. Warm-time zero-workgroup dispatches force pipeline compilation before frame one.
- **shaders** — engine-authored compute, vertex, and fragment kernels use TGSL with CPU-callable logic twins. Relocatable `*Wgsl()` exports preserve raw shader composition where a consumer needs it; the migration guide lists every renamed 0.8 shader export.
- **rendering** — surfaces and backgrounds use schema-derived layouts plus State-owned registration; draw, mesh, mirror, part, instancing, shadow, skin, text, sprite, sky, fog, and post-processing resources retain typed buffer identities without removing raw reach-in.
- **diagnostics** — GPU `console.*` capture and draining join the verify protocol, and one-shot readback/framebuffer probes make shader and resource failures attributable. Shader-side `console.error` fails verification.
- **verification** — kernel differentials, emitted-WGSL discipline checks, real-device gym probes, the `no-walls` raw-interop flow, and the fresh-install gate cover the new substrate from CPU logic through packaged consumers.
- **skin** — the live joint-palette skinning substrate is engine-owned, at `extras/skin`: `SkinPlugin` gives a producer the palette, the `Skin` component, the per-frame flush, and the pose-write API with no glTF asset in the scene. `extras/gltf` is the converter: it turns a rig into substrate data and registers the `skin-live` PBR surfaces that draw it.
- **skin (breaking)** — `LiveSkin`, `skinMatrix`, and `Skin` move off `@dylanebert/shallot/gltf/core` and the gltf barrel onto `@dylanebert/shallot/extras` (also on the bare barrel). The WGSL a custom skin surface splices is the new `@dylanebert/shallot/skin/core`.
- **verify** — `--run k=v` (repeatable) batches configurations through one server boot and one browser, a fresh context and page per run, one verdict each, a JSON array out and nonzero exit if any fails. `--timings` reports per-phase wall clock (server boot, first page load, harness ready, run, memory idle, capture, teardown), so a run that hangs names its phase.
- **packaging** — TypeGPU is a peer pinned to the 0.11 minor; the CLI synthesizes the required Vite transform for manifest projects. The glTF test fixtures no longer ship in the npm tarball.
- **scaffold** — `bun create shallot` and `shallot recipe` emit one agent contract, not two copies: AGENTS.md holds it and CLAUDE.md is `@AGENTS.md`, so editing one can't drift from the other.

## 0.8.1 — 2026-07-23

The fresh-install patch: recipes ship with the package, the GPU floor widens, the verify gate gets pixel-honest.

- **gpu** — the base device floor shrinks to the default path's needs: `shader-f16`, `timestamp-query`, and texture compression no longer gate device acquisition (`shader-f16` off via a bit-identical `vec2<u32>` material binding; `timestamp-query` → `ProfilePlugin.features`; BC/ETC2/ASTC → `GltfPlugin.preferredFeatures`). **Breaking:** `gltf/core`'s `pickTargets` returns `Targets | undefined` instead of throwing — the `UnsupportedError` fires per-image, only when a KTX2 image has no transcode target.
- **recipes** — the recipes corpus ships in the npm tarball, indexed by `examples/AGENTS.md`. `bunx shallot recipe` lists it; `bunx shallot recipe <name> <dir>` copies a recipe out as a standalone project pinned to the installed engine version. Every recipe demonstrates its concept on open.
- **verify** — the `rendered` verdict is pixel-honest: a booted-but-blank canvas fails instead of passing. `--leak <bytesPerSec>` injects a retained allocation (the leak detector's red-proof); the leak flag reads a post-run idle window so GC noise doesn't false-positive.
- **fixes** — the fog-and-light-shafts recipe actually shows shafts and shadows (its sun was missing the `shadow` opt-in); test files no longer ship in the npm tarball.

## 0.8.0 — 2026-07-21

The repo is the documentation: no editor, no docs site. New default physics backend, an audio effect graph, and a shipped verification gate.

**Breaking:** the editor is gone — `bunx shallot` no longer opens it; author scenes as data and run `shallot dev`. The `./editor` subpath is renamed `./document`. Project templates are removed; `bun create shallot` is the only scaffold.

- **physics** — Tumble is the new default backend: built-in physics, a TS engine over a wasm kernel, running on the CPU and multithreaded wherever the host affords shared memory; AVBD moves behind the `./avbd` swap-in. Ragdolls with a live joint palette (`LiveSkin`), tumble/avbd backend swap, and correctness hardening across hot reload and backend swaps: eid restamping, kinematic-sleep wake, constraint-signature folding, hull validation.
- **audio** — an effect node graph: delay, dynamics (compressor/limiter/expander/gate), waveshaper, EQ, and modulation (chorus, flanger, phaser, tremolo).
- **assets** — live joint-palette glTF skinning: a skinned mesh drives a live pose palette on surfaces.
- **verify** — `shallot verify [dir]` boots a project in a headless browser and exits 0/nonzero, a self-terminating gate for an agent or CI. The `window.__harness` protocol (`@dylanebert/shallot/harness`, `installHarness`) drives custom pass/fail; `bun bench` and `bun run flows` are thin wrappers over it.
- **docs** — the repo is the documentation: JSDoc on every public export, the shipped `AGENTS.md` consumer contract, and problem-named recipes under `examples/recipes/` indexed by `examples/AGENTS.md`. The generated docs site and its projection pipeline are removed.
- **scaffold** — `bun create shallot` emits a project with its own CLAUDE.md and AGENTS.md pointing at the engine's agent surface.
- **toolchain** — TypeScript 7.

## 0.6.0 — 2026-07-05

First documented release.

- **engine** — data-driven ECS (entities, components, systems, queries, plugins), XML scene files the editor and runtime round-trip, the `shallot.json` manifest as project source of truth, time control (pause, timescale, fixed step), plugin hot reload
- **rendering** — GPU-driven WebGPU forward renderer: parts and surfaces, PBR materials, MSAA, sun/point/spot shadows, clustered lights, HDR with tonemapping, custom shading and backdrops, procedural sky, volumetric fog
- **physics** — GPU rigid bodies (AVBD solver): boxes, spheres, capsules, hulls, springs and joints, kinematic character controller, first-person player
- **assets** — glTF import: Draco geometry, KTX2 textures, PBR materials, baked skinned animation; drag-drop into the editor
- **audio** — wasm DSP synth with spatial voices, `Sound`/`Listener` components
- **extras** — orbit camera, tweens, world-space sprites and SDF text, debug lines, selection outline, profiler overlay
- **editor** — outliner, reflection inspector, transform gizmos, undo/redo, play mode as faithful preview, autosave, add-entity bundles, in-editor docs
- **cli** — `bunx shallot` (editor), `shallot dev` (hot reload), `shallot build` / `shallot run` (web, or native windows/mac/linux via system webview; `--portable` bundles CEF)
- **scaffold** — `bun create shallot`
- **docs** — generated reference + guides, projected from code and runnable examples
