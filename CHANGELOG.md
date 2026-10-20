# Changelog

## Unreleased

- Desktop targets state their status in `shallot build --help`.
- Mac: the system webview backend needs macOS 26 for WebGPU.
- Mac: the bundled-Chromium shell no longer stores its cookie key in the login Keychain, so a rebuilt app no longer prompts for "Chromium Safe Storage".
- Windows: native builds link with Visual Studio's `link.exe` and no longer need a separate LLVM install.
- AgX tonemapping now uses Bevy's default-contrast LUT instead of an analytic approximation, changing its displayed colors.
- Fog no longer streaks the haze where a ceiling or side wall seen at a grazing angle meets a farther wall with light from the lit air beyond it: cells behind a column's own surface take the light on the ray of the pixel that sees that deep, and the grid filter leaves them out of its neighbours' haze.

## 0.10.0

a foundational reset. 0.10 rebuilds shallot so everything can be run and checked from a terminal: no display, software gpu, ci, etc. this makes the engine far more testable, and lays the groundwork for multiplayer support.

### highlights

- **composable layers.** core, standard and optional extras, with cleaner separation between modules
- **headless.** cpu-only worlds, and software webgpu through dawn, for servers, tests and ci
- **independent worlds.** one page or process can run several ecs worlds
- **box3d physics.** deterministic port of box3d, with heavy lifting in multithreaded wasm
- **rendering rework.** masked and blended materials, sorted transparency, more cameras, infinite reverse-z depth, and more
- **simulation control.** fixed ticks apart from frame pacing, input addressed by tick, and snapshot and restore, so a run can be stepped as data
- **discoverable cli.** command-line interface lists every command and example, copies any example as a project, and prints its agent skill and migration guide

### removed

- `.scene` files, document model, editor mode, audio graph, tweens and sequences

### factored out to optional plugins

- gltf loading is [`@dylanebert/shallot-gltf`](https://www.npmjs.com/package/@dylanebert/shallot-gltf) (experimental, incomplete conformance)
- avbd physics (reworked gpu physics coming soon as an optional plugin. box3d physics recommended in most cases)

to port a 0.9.5 app, run `bunx shallot migrate` or read [`MIGRATION.md`](https://github.com/dylanebert/shallot/blob/v0.10.0/MIGRATION.md). earlier history is at [`archive/changelog-pre-0.10`](https://github.com/dylanebert/shallot/tree/archive/changelog-pre-0.10).
