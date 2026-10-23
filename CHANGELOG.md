# Changelog

## 0.10.1

a release for putting games in people's hands: layered camera views, picture effects, phone controls, and publishing a web game from the terminal. a few uncommon 0.10 behaviors change; see the changes below.

### highlights

- **publish a game.** start with `bunx shallot create my-game --template minimal`, then run `bunx shallot publish` inside `my-game` to build, upload and verify it on cloudflare or itch.io
- **layer cameras.** keep a held item in front of the world by drawing camera views over one another and choosing what each camera sees, as in `view-model`, without outlines leaking through walls
- **shape the picture.** add a glow around bright objects, bend the picture like a lens or separate its color edges, or tune colors and how bright and dark areas appear: `post-processing`, `bloom`, `color-grading` and `tonemapping` each show a part
- **play on phones.** `first-person` has touch movement, look and jump; `interact` adds use controls, and `sound` unlocks audio on touch release. views you rotate around a subject and example panels also work with touch
- **light the air.** walk through haze lit by sunlight, spotlights and light bulbs in the new `volumetric-fog` example, with fog streaks along surfaces viewed edge-on fixed; `particles` now focuses on floating specks simulated by the graphics processor
- **smoother startup.** `minimal` and `materials` prepare rendering under the loading screen rather than stalling the first frames when safari starts without cached graphics work; ios rendering is fixed too. graphics processor failures stop play with an explanation instead of continuing to draw
- **desktop apps.** windows builds need no separate llvm tools install, unsupported attempts to build or run mac or linux apps from another operating system fail early, and mac apps bundled with chromium no longer prompt for the login keychain. `bunx shallot build --help` lists target status and requirements, including macos 26 when using the system's built-in web view
- **checks for new projects.** a new project's CI checks your code for type errors before running its tests, and its agent guidance asks for the same check before finishing

### changes

- cameras without a canvas or texture assigned now all draw on the first canvas, instead of leaving a second camera unused; `attachCanvas`, which assigns a canvas, joins the same drawing order
- if you write your own graphics buffers for lights or meshes, update `PointLightGpu` and `meshInstanceTable` data and bindings for the new fields that choose which cameras see them
- `captureFrame`, which captures a rendered frame, accepts any canvas pixel dimensions; the returned `Capture.identity` no longer includes `deviceScale`
- adding a component an entity already has with `world.add` replaces its values, resetting omitted fields to defaults, instead of warning and leaving it unchanged
- web builds now find their files relative to the page (`base: "./"`); games that change the page's url without loading a new page may need `base: "/"` to keep file addresses relative to the site's root
- if you chose the AgX tonemapping look, which controls how scene brightness becomes screen color, its displayed colors change; review your scenes and expected captures
- if you install Playwright for browser checks or publishing verification, its required version is now `^1.64.1`
- switch the old vignette imports, `@dylanebert/shallot/vignette` and `VignettePlugin`, to `@dylanebert/shallot/effect-stack` and `EffectStackPlugin`; the old names still work but will be removed in 0.11. the `Vignette` effect keeps its fields and defaults

to update a 0.10 app, run `bunx shallot migrate` or read [`MIGRATION.md`](https://github.com/dylanebert/shallot/blob/v0.10.1/MIGRATION.md).

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
