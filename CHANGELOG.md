# Changelog

Notable changes per release. Versions follow [semver](https://semver.org).

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
- **docs** — [generated reference + guides](https://dylanebert.github.io/shallot/docs), projected from code and runnable examples
