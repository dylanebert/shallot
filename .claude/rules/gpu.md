---
paths:
  - "src/engine/runtime/**/*.ts"
  - "src/engine/utils/encode.ts"
  - "src/standard/render/**/*.ts"
  - "src/standard/sear/**/*.ts"
  - "src/standard/part/**/*.ts"
  - "src/standard/slab/**/*.ts"
  - "src/standard/bvh/**/*.ts"
  - "src/extras/{cells,gltf,lines,outline,profile,skin,sky,sprite,text}/**/*.ts"
---

# GPU

`render.md`; capacity fixed at build. No bindless/async compute; bounds checked.

## Types

Shallot owns commands; TypeGPU schemas/shaders. One named owner schema derives types/stride/layout; typed identity crosses boundaries, adopt wrapped/unwrap raw commands only. TGSL; raw resolve interop, WGSL leaves only inexpressible primitives; whole raw only diagnostics/reference/output/no-walls.

Packed uploads, not objects. JS packer mirrors only for needed TGSL CPU arms; differential-test both. TGSL `idiv`/typed integers, lint use-gpu. Handle completion rejection/counters. Root/device memo; maps/build reset: republish typed/raw hits.

Base only default-plugin needs: indirect-first-instance/bgra8unorm-storage/rg11b10ufloat-renderable. Required fails before load; only preferred falls back. BVH subgroups→LDS/glTF BC/ETC2/ASTC preferred; profiler timestamp-query required; shader-f16 opt-in. Unrequested=false. Adoption: host unions/users guard requirements. Keep gates lazy; audit removed guarantees.

KTX2 gate per image; standalone requests nothing. Compressed size/format buckets; no batch fallback for outliers.

## Binding limits

10 storage/stage across groups, read-only too; count before adding. **Reuse over add** at 8+: spare lanes. Extend glTF ceiling audit; uniforms/textures separate.

### Consolidation

GPU scans, no CPU gathers. Slab authored entities; bulk non-entities owner-managed; derived GPU-only. Mirror/consumer compaction. Small-int slabs CPU-only, pack u32; half declarations need feature, packed storage doesn't.

Cols-buffer ≥3-field SoA; merge same-pass/header data, bulk-concatenate near-limit uploads, split last. Don't merge different-pass/mixed CPU-GPU/exclusive owners.

## Layout rules

1. SoA default; AoS ≥4 hot fields/measured thrash. Hot≤64 B, split cold; 32-element tiles after measured limits. Avoid power-two strides≥128 B; pad. Vec3 stride 16 B; schema size checks.

6. Audit emitters' worst range/tolerance. Bounded fixed-point: unit unorm8/scaled position unorm16/directions snorm-8/oct. LDR sRGB u32; HDR r11g11b10ufloat/half intermediates; packed IDs/flags, smallest-3 quats. Mixed HDR half; storage encode/f32 math.

World position/velocity/acceleration/instance transforms f32; never quantize instance rotation/scale. Sentinel/unbounded/time-cumulative/finite-differenced state f32; saturation can't change classifiers. Audit fields, not categories/library defaults.

9. No oct interpolate/filter: plain normals/renormalize, VAT too. Flat IDs fetch constants; pixel reconstruction, no redundant spaces; pack/prune.

8. Quantize only if production pass ≥5× off bandwidth floor AND per-field audit clean AND encoding microbench-proved free/near-free. Isolate branch-optimization proof, not unified paths. Affine first; log decode costs.

## Atomics

Low-contention counters unbatched. Subgroup reductions: one global atomic/slot/workgroup, ordered float extrema; LDS fallback. LDS≥3 reuse/load, ≤8 KiB unless occupancy proved; tiles 33. Runtime widths 32/64; software isn't proof.

Uniformity opt-out: runtime-uniform/proven fixed cap. Separate-data cross-workgroup handoff crosses dispatch; relaxed atomics/barriers insufficient. Scan alone: value+flag one atomic, stable prior-input fallback, no progress spin; barrier exits `workgroupUniformLoad`.

## Dispatch count is a first-class cost

Minimize synchronized passes; one workgroup, one dispatch. **Indirect ≈ 2× direct**: CPU-known direct, indirect zero validates. GPU-count branches pay both encodes: reduce unconditionally or specialize static caps.

Mask cheap ALU; bench memory/sample/traversal early-outs. Uniform/tail branches fine. Verify unrolling/repeat small bodies. FMA needs unfused-code/win proof; prefer dot.

## Bandwidth ceiling check

Strip empty→reads→atomics→math→suspects; restore. Active set (solver color slice) sets cache-tier floor. ≥5× off: stride first. Sub-0.1 ms: isolated stress; desktop cache/noise can't refute portable DRAM gains. Varyings no cache escape.

## Native targets and webview backends

Default wry webviews; portable CEF. Mac LDS/no subgroups; Linux needs portable. Admission by features/limits, not browser name: a floor-meeting configuration owes rendered proof, and Firefox/Windows is unqualified. Intel Mac unaudited.

## DXC shader compilation

No large dynamic-loop functions/dead duplicates; split pipelines. Constant bounds/breaks fine; repeat cache isn't compile proof.

## Render passes on TBDR

No default depth prepass; requested lanes stay. Fuse fullscreen compute/present once; fold compatible outputs, never MSAA color/discrete id. Audit load/clear/discard; discard transient depth. Auto DPR 1–2, apps may lower.

## NaN policy

Propagate NaN/Inf; fix sources. Branch only for required finite outputs (centroids/atomic sentinels).

## GPU debugging

Exact claim: CPU→WGSL/hash/compile→API/errors→shader→submission probes→verify output. Stop first failed rung; no neighbour proof. One fix, same boundary; probes before capture.

Safe TypeGPU logs over bespoke buffers: perturbing/no vertices/overflow loses evidence. Owned draws drain; replay mixes backlog, order isn't time/causality. No frame logs; console errors fail verify. Typed reads/`probeBuffer`/`probeTexture` causal; Mirror isn't. No allocating telemetry polls.

### Runtime Inspector

Isolated aid, not app/host proof. Declare features; no scaffolding; unavailable→verify.

## GPU profiling

Timestamps every pass, optional; names accumulate/profiler first/absent no-op. Spans omit barriers/copies/present/validation: no subtracted bubbles. Command counts (recorded bundles) predict GPU floor, not fence addition. GPU-bound no-op calibration, not approximate direct/readback; ablate before capture.

Timing-grown pools: lazy descriptors, tested at growth sites, not label exclusions. `testing.md`: Freezing a golden.

## Labels

Stable shader/sync/async labels per pipeline; typed names/functions/schemas. Join diagnostics/WGSL/hash/profile/verify/capture. Compose detail, not hidden causes; extend creation check only.

## WebGPU-specific traps

Consolidate indirect buffers AND commands: distinct costs. Bundles save CPU not validation; static lists may bundle, Sear collapses shadows. Cache bindings. Metal dynamic private spills: keep small/global-stream/scalar-unroll; pointer/private/vec4 isn't fix. Real multi-lane Metal gate, not NVIDIA alone.
