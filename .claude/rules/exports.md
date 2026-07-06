---
paths:
    - "packages/shallot/src/**/*.ts"
    - "packages/shallot/package.json"
---

# Exports

## Tiers

- **`@dylanebert/shallot`** — the engine's public API (re-exports engine + standard + extras + editor). The happy path (`Part` + `mesh()` + `Camera` + the default plugins), the platform primitives (`slab`, `Inputs`), and the renderer (`RenderPlugin` / `SearPlugin` / `GlazePlugin`, `lit`). `DEFAULT_PLUGINS` is the zero-config set (`standard/defaults.ts`). Contract: `render.md`.
- **`@dylanebert/shallot/extras`** — the author-facing convenience + debug plugins (`orbit`, `lines`, `outline`, `sprite`, `text`, `tween`, `profile`, `sky`, added per-scene) plus the glTF importer (`loadGltf` + `placeScene`; add `GltfPlugin` — scenes reference primitives by name (`part="mesh: model.glb#0"`, imported by its preloader + decorated by its route sync), the utility itself stays one-way and creates no entities; its decode/cache/tooling surface is `gltf/core`). Also reachable on the bare barrel. The boundary is engine-substrate vs author-convenience, **not** default-vs-opt-in: opt-in engine capabilities (`audio`, `mirror`) and libraries (`bvh`) live in `standard/`, not here — `standard/` ⊋ `DEFAULT_PLUGINS`. `extras/index.ts` star-exports every module — each module's `index.ts` owns its clean public API (internal/test-seam siblings: "Barrel rules" below).
- **`@dylanebert/shallot/editor`** — editor libraries (document model, session sync).
- **`@dylanebert/shallot/vite`** — build tooling: `projectPlugin`, the vite plugin that resolves `virtual:project` from a project's `shallot.json` (the enabled plugins + scene). The editor, `shallot dev`, and `shallot build` use it internally for every manifest project; only an ejected example that owns its vite config (gym, `showcase/visualization`) imports it directly. Build-config only, not a runtime subsystem (exempt from the doc-page gate).
- **`@dylanebert/shallot/runtime`** — platform layer: `Runtime`, `now`, `requestFrame`, `readFile`, `readBinary` (CPU); `Compute`, `requestGPU`, `UnsupportedError` + the `checkStorageBinding` / `checkTextureLimits` allocation pre-flight guards (named-throw before a large buffer/texture OOMs; contract in their JSDoc) (GPU device wrapper).
- **`@dylanebert/shallot/*/core`** — non-public extension API per module, for custom pipelines / tooling / diagnostics. One line each; detail lives in the domain rule:
  - `render/core` — the `Surfaces` / `Meshes` / `Draws` registries, `Render` singleton, the `BeginFrameSystem` + `OverlaySystem` ordering anchors (the latter the post-color seam's scene-transform↔overlay split), the frame/view/lighting WGSL contracts, `Views` + `attachCanvas` / `attachView` + `sceneTransform` (the scene-effect seam: a compute system reads `view.framebuffer`, writes a ping-pong scratch, repoints it), the instancing convention (a surface declaring `eids` + `transforms`), the froxel cluster substrate (`Clusters` + `LightCull` + the `clusterAabb`/`zSlice`/`lightClusters` oracles, grid constants), and the shared image→`texture_2d_array` upload path (`imageArray` / `arrayFromBitmaps` — gltf baseColor + the sprite atlas, so neither extra reimplements decode/resize/mip). Contract: `render.md`.
  - `sear/core` — `PrepassSystem` + `ColorSystem` ordering anchors, the opt-in `Tag` / `Depth` prepass lanes, plus the **relocatable shading chunks** a screen-space consumer (the `fog` volumetric march) splices to evaluate the same lit, shadowed lights sear's color FS does: `LIGHT_EVAL_WGSL` (`distanceAttenuation` / `spotFactor` / `clusterCell`); the clustered point/spot path — `casterWgsl` + `pointShadowWgsl` (`pointShadowOf(light, normal, fragWorld)`, world-pos a param, atlas/sampler/casters/tile-rects referenced by name) + the `pointAtlasView` / `shadowSampler` getters (the casters + importance-sized tile rects are the published `Compute.buffers.get("pointShadows")` / `("pointTileRects")` — a consumer binds both; the compacted lights + grid are `render/core`'s `LightCull`); and the sun path — `SUN_SHADOW_STRUCT_WGSL` + `SAMPLE_SUN_SHADOW_WGSL` (`sampleSunShadow(worldPos, normal)`, `shadowMap`/`shadowSamp`/`sunShadow` by name) + the `sunShadowView` / `sunShadowParams` getters + `SHADOW_PARAMS_BYTES` (read-only exposure of sear's sun shadow — sear still owns the map/render/params, not a writable seam). Also the **backdrop seam**: the `Backgrounds` registry (`Backgrounds.register({ name, bindings?, preamble?, fs })` — the `Surfaces` analogue, a view-ray → HDR color fill on un-rendered pixels) + the `Background` recipe type; the `Backdrop` per-camera selector component rides the main barrel. Contract: `render.md` "Camera passes". (`surfaceCode` / `backgroundCode` — the surface + backdrop WGSL codegen — are internal to `sear/forward.ts`, imported directly by their structural tests, not re-exported.)
  - `glaze` — the default postfx composite: `GlazePlugin` + `Glaze` (also on the barrel) + `GlazeSystem` (the `before: [GlazeSystem]` ordering anchor) + the `Tonemap` operator enum (default Neutral, also on the barrel) + `TONEMAP_WGSL` (the WGSL chunk, glaze subpath only). All live in `standard/glaze/`; `Tonemap` + `TONEMAP_WGSL` in `standard/glaze/tonemap.ts`. Contract: `render.md` "glaze".
  - `fog/core` — the opt-in volumetric atmosphere's extension surface (`Fog` / `FogPlugin` on the barrel): the march WGSL twins — extinction (`FOG_MARCH_WGSL` / `FOG_STRUCT_WGSL`) + in-scatter (`FOG_INSCATTER_WGSL`: `henyeyGreenstein` + `inScatterContribution` clustered + `sunInScatter` directional) — the `Fog` uniform packer `packFog` + the `FogSystem` anchor, and the TS oracles the GPU twins are pinned to (`fogTransmittance` extinction + `fogInScatter` single-light + `fogSunInScatter` directional-sun in-scatter, over `FogLight` / `FogScatter` / `FogSun`). A screen-space `sceneTransform` consumer (in the post-color seam) that binds `render/core`'s `LightCull` + `Lighting` + `sear/core`'s point + sun shadow service for the shafts.
  - `bvh/core` — the LBVH GPU builder, a rendering-unaware library: `createBvh`, the per-stage factories, the ray-AABB traverser `BVH_TRAVERSE_WGSL`, node constants. GPU-count contract (`bvh.count`).
  - `physics/core` — `PhysicsStep` + the GPU/SAT constants, CPU raycast (`raycast`, `screenToRay`, `qRotate`) + the pick utilities (`bodyCandidates`, `grabHit`, `forwardRay`, `cursorRay`, `worldToLocal`), hull registry. Contract: `physics.md`.
  - `character/core` — the eid-keyed kinematic drive (`move`/`jump`/`pose`/`grounded`). Contract: `physics.md` "Authoring".
  - `audio/core` — the audio voice contract + DSP-substrate authoring. Contract: `audio.md`.
  - `gltf/core` — the glTF import pipeline for tooling + custom async loading: the deviceless `decode` + content-keyed cache (`ensureDecoded` / `register`), off-thread `decodeInWorker`, union-staging progress (`unionPending`) + cache management (`invalidate` / `clearGltfCache` / `gltfCacheStats`), the baseColor bucket names, and the raw parsed-glTF types. The author happy path (`loadGltf` / `placeScene` / `placeGltf` / `GltfPlugin`) rides `extras`.
  - `tween/core` — the WAAPI timing atom (pure fns over numbers + the easing surface). The happy path (`Tween` / `tween()`, `Sequence` / `sequence()`) rides `extras`.
  - `ecs/core` — registry (`register`, `getComponent`, `getTraits`, `entries`, `clear`, `idOf` — the stable component id keying membership/queries), reflection (`schema`, `inspect`, `snapshot`, `dump`, `readFields`), scheduler seams (`addSystem`, `record`, `fenceWait`) on `State`.
  - `scene/core` — the scene text↔state codec for the editor + scene tooling: `parseFields` / `formatFields` (attribute string ↔ field record), `readComponent` / `setFieldValue` (live component ↔ field), `normalizeAttr` (canonicalize, the scene formatter's), and `findNodeById` / `findParent` (parsed-tree navigation). The author load/save path (`parse` / `load` / `serialize` / `stringify` / `diagnose`) rides the main barrel.
  - `utils/core` — the GPU codec extension surface an extender splices: the relocatable WGSL chunks (`XFORM_WGSL`, `OCT_ENCODE_WGSL`, `POS_QUANT_WGSL` / `POS_QUANT_PACK_WGSL`, `LDR_COLOR_UNPACK_WGSL`, the OkLab pair `LINEAR_TO_OKLAB_WGSL` / `OKLAB_TO_LINEAR_WGSL`) + their bit-identical CPU pack/unpack twins (`octEncodeNormal` / `octDecodeNormal`, `pack2x16unorm` / `unpack2x16unorm`, `packColor`, `packLdrColor`). The author math + color + trait-authoring helpers (`lerp` / `quat` / `unpackColor` / `eulerAlias` / `units` / `Registry`, …) ride the main barrel. Definitions stay in `engine/utils/{color,encode}.ts`; internal cross-module consumers import the subpath, never the deep file (`check-imports`).

## Distribution layers

Two layers, drawn on purpose; the boundary is load-bearing.

- **The tool (canonical, inner).** `bun install @dylanebert/shallot` + the `shallot` CLI, used inside a standard web dev project with any framework. Shallot is a tool/package, not a framework — the editor reads/writes shallot files (scene, manifest, plugins) in any layout; the user's own bundler/framework owns preview. First-class; never traded away for the convenience layer.
- **The convenience bundle (outer).** Bundles the canonical path into a frictionless experience (the standalone app, the later live editor — one bundle). Its own folder (not `src/`), depends **purely inward** on the tool + engine, and **composes** the tool — never duplicates or reimplements it. Inward core: the headless project-resolution substrate (read manifest → resolve plugins → load scene, no editor/vite/browser).

For engine/core code under `src/`: stay unaware of the convenience layer. Dependencies point inward, never outward; a convenience-only concern reaching into `src/` is the inversion to reject.

## Barrel rules

Each plugin barrel (`index.ts`) exports its public API:

- **Barrel:** components, singletons, registration functions, types that game developers use.
- **`*/core` subpath:** WGSL constants, shader compilation, GPU buffer packing, internal mesh/surface queries.
- **Neither (internal):** implementation details consumed only within the module. A module-internal export
  shared across sibling files (or a test seam) lives in a sibling file imported directly, never re-exported
  from the barrel — orbit's `OrbitSmooth` (`smooth.ts`) is the shape.

The generated docs reference renders the barrels and component schemas verbatim, so an export or field
decision **is** a docs decision — see `hardening.md` "The API is the docs".

Cross-module imports within `standard/` use relative paths. Producers (the `extras/` viz plugins, external plugins) import from `./render/core`.

**`sideEffects` (tree-shaking).** `package.json` lists the only side-effect modules so the barrel shakes — import `Part`, ship no `audio`/`gltf`. A module with an import-time side effect (today just `standard/defaults.ts`: the `DEFAULT_PLUGINS` + default-loading injection) must be listed, **and so must any barrel that re-exports it via a bare `import` for that effect** (`standard/index.ts`) — named imports resolve straight to source modules and skip the barrel body, so an unlisted side-effect host is silently dropped (a `defaults: true` build then renders nothing). Exact paths only; bun <1.3 deopts on wildcard globs.

## Registry pattern

`Registry<T>` (`engine/utils/registry.ts`, on the barrel) — name-keyed, auto-assigns a stable numeric ID. `.register(spec)` returns the ID; `.delete(name)`, `.clear()` (the reload seam — wipes all entries + the ID space for a clean rebuild), `.get(name)`, `.id(name)`, `.name(id)`, iteration via `Symbol.iterator`. The renderer uses it for `Surfaces` / `Draws` / `Meshes`; producers call `.register(...)` directly.

`mesh()` wraps `Meshes.register(...)` because it does real work first (allocates GPU buffers from typed arrays). Plain pass-through wrappers don't exist — `.register()` / `.delete()` convey the intent.

## Type discipline

- No type aliases that just rename primitives (`type Foo = string`). Use the primitive directly.
- Interfaces earn their place by having >1 field or methods.
- Don't export types used only within one file.

## Naming

Exports should not commonly conflict (no `as` renames at import sites). If a name is too generic (e.g. `readBuffer`), it belongs in a subpath or stays internal — not the main barrel.
