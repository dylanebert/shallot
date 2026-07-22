# Changelog

Notable changes per release. Versions follow [semver](https://semver.org).

## 0.8.0 — 2026-07-21

The engine goes AI-native: the repo is the documentation, and there's no editor and no docs site. New default physics backend, an audio effect graph, and a shipped verification gate.

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
