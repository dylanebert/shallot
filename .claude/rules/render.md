---
paths:
    - "packages/shallot/src/standard/render/**/*.ts"
    - "packages/shallot/src/standard/sear/**/*.ts"
    - "packages/shallot/src/standard/glaze/**/*.ts"
    - "packages/shallot/src/standard/part/**/*.ts"
---

# Render Contract

`exports.md`: exports; `gpu.md`: limits; `render/contract.ts`: types.

## Layering

Render owns frame/view/camera/lighting/registries, never iterates draws. Part produces instances, no shading. Sear/Part never import each other; Render imports neither. Glaze imports only `render/core`. Camera markers select peer renderers; custom producers are peers.

Render owns froxels/GPU light compact/cull; overflow warns, depth-only views never bin lights. Sear owns explicit shading/shadows: half-Lambert diffuse, physical specular; bare dielectric zero, glTF standard.

Renderers write HDR `view.framebuffer`, never `view.present`; glaze/custom compute presents, Sear forces neither. Glaze encodes linear→sRGB for storage swapchain; Neutral is zero-config tonemap. Grade→tonemap→saturation→perceptual posterize/dither→display vignette; no post-AA.

`sceneTransform` gives distinct ping-pong read/write; third in-frame consumer triggers ring growth. Scene effects precede `OverlaySystem`, overlays follow, both before glaze regardless of registration order. Overlays use compute: scratch may be storage-only. Color-feeding effects order prepass→color; resolved-color effects follow color. No effect registry/coupling.

## Registration

Stable registry IDs, State-owned surfaces; delete draws by name. Happy path: data/plugins/`mesh()`. Surfaces final at warm; mesh/view growth never recompiles pack.

Part registers all instanced-surface × mesh pairs, unused pairs zero-instance, no entity query. Grow buffers preserving mesh offsets; reseed/repoint draws, free old buffers after submit fence. Register draws at setup or later, never parallel warm.

## Surface authoring

Use `surfaceLayout`, TGSL via `layout.$`, State-owned registration/helper graphs; no preamble/strings. Surfaces shade any mesh. `eids` + `transforms` opts into instancing before custom vertex work; inverse-transpose normals handle nonuniform scale. Screen surfaces own clip position/no face culling; world surfaces project after displacement.

Renderer owns decode/uniforms/entries/helpers. Integer varyings flat; prune unread built-ins. Four custom slots before justification, 16 total; plain interpolated normals renormalized, never oct. Fragment color returns verbatim; lighting is opt-in. Vertex lighting gets neither clustered points nor sun shadows; fragment lighting gets both, ambient unshadowed, absent shadows fully lit.

Sear owns default/vertex/unlit; Part defaults default/cube. Only `Surface.tag(ctx, defaultTag)` authors tags with full context, no color shading/extra draw. Default eid if instanced, `TAG_NONE` otherwise; meaning is consumer-owned.

## Draw shape and ordering

Always 20-byte indexed-indirect, CPU-known draws too. Absolute indices, mesh count/base, zero base vertex; offset adds slot × `viewStride` (default zero). Storage-pull attributes, never `setVertexBuffer`; hardware indices preserve vertex reuse. Draw carries no resources.

Producer/render work follows `BeginFrameSystem`; position writes (geometry/transforms/eids) also precede `PrepassSystem`, the geometry anchor even without lanes. Fragment-only writes need only frame opening. Registration order isn't ordering. Render owns unique terminal submit after first/normal/last work; custom renderers may be last, never terminal.

## Binding resolution

Geometry resolves via Meshes; named resources via `Compute.typed`, raw twins for raw consumers. Texture/sampler type selects view. Missing resources skip draws. Only registered fields' named slabs publish; standalone slabs don't allocate. Cache identities, not contents; allocate stable resources once.

One Draw/pair, never entity; cache pipelines/surface, bind groups/shared buffers. Multi-draw changes internals, not contract.

## Camera passes

Reverse-Z near 1, far/clear 0, greater+writes. Color owns depth and fuses opaque→backdrop→alpha into one HDR `rg11b10ufloat` output, never MRT. Default 4× MSAA toggles per-camera to 1×; resolve once, no blur/TAA. Prepass/shadows/effects stay 1×, separate depth. Shared vertex math across variants; prepass shadow stubs prevent sampling their own depth.

Bare cameras have no prepass depth round-trip. Requested lanes form one engine-closed prepass, not consumer registry/pass-per-output. Tag: u32, opaque/clip only, never blended/MSAA/color MRT. Store depth only on request, otherwise discard; handles null until produced. Future normals/motion join that prepass as named lanes, never speculative fields/generic texture maps.

Backdrop fuses view-ray→HDR between opaque/blend; far-plane test/no depth write, pixel-derived ray. Otherwise clear color; sky is plugin-owned. Storage scratch stays `rgba16float`; scene precision option waits for banding.

## Transparency: `blend` modes

Per surface, not Part. Opaque writes depth; alpha straight-alpha linear blend, greater-equal test, no writes/prepass/tag/casting. Clip discards in color/depth/shadows so holes cast. Pack/cull is mode-independent. First overlapping alpha consumer owes GPU back-to-front camera-depth sort; nonoverlapping layers need none.

## Culling lives in the producer, not the consumer

Cull inside compaction per view, shadows included; Sear ignores instance bounds. No-view keeps all; uncull is correct but slower. Extract shared cull/count/scan/scatter only at second many-instance producer, never move into Sear. Hi-Z waits for real target high occluded geometry complexity; prefer generation-time visibility/authored streaming even then.

## Shadows: sear-internal, `Shadow`-gated

Sun shadows and Point-light shadows are presence-gated, no separate plugin/visibility seam. No casters means no atlas/combo allocation, fully lit. Perspective sun cascades; ortho fits one ground footprint. Extend toward light for occluders; texel-snap, blend cascade boundaries, clamp PCF to tiles.

Point/spot share importance-sized tiles (six/one), highest importance not query order; hysteresis prevents flicker. Atlas/caster limits fix before build; fit/slot overflow drops least important with warnings. Cull every combo through shared views; regather one indirect draw/casting mesh without extra storage binding. Only instanced surfaces cast. Hardware clipping/projection matches receivers; perspective bias uses linear depth. Shading/depth-only slots stay separate.

## Producers

Typed main/position/quant share indexed families; static meshes pack at warm, procedural buffers producer-owned. Correct index/storage/indirect usage, pre-shift atlas indices. Typed overrides validate schema/usage; raw unchecked. Mutate contents, not identities.

Touched shadows: `bun bench --scenario render --param mode=<m>`. Product/CPU/display/golden/release-demo laws: `testing.md`.
