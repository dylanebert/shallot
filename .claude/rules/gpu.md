---
paths:
    - "packages/shallot/src/engine/runtime/**/*.ts"
    - "packages/shallot/src/engine/utils/{encode,readback}.ts"
    - "packages/shallot/src/standard/render/**/*.ts"
    - "packages/shallot/src/standard/sear/**/*.ts"
    - "packages/shallot/src/standard/part/**/*.ts"
    - "packages/shallot/src/standard/slab/**/*.ts"
    - "packages/shallot/src/extras/**/*.ts"
---

# GPU

Reference: `render.md` for the rendering pipeline + mesh format. Capacity is fixed at app construction (`build({ capacity })`, default 65536) and exposed as a module-level `capacity` export; component storage is SoA typed columns (`ecs.md`).

## Framing — WebGPU isn't Vulkan

WebGPU enforces bounds-check semantics on storage accesses (via hardware robustness on most discrete GPUs, injected clamps where unavailable — measured 17% perf cost on AMD, 34% on NVIDIA 1080 Ti, up to 70% on Intel integrated, per [gpuweb#1202](https://github.com/gpuweb/gpuweb/issues/1202)). It has no bindless ([proposal in draft](https://github.com/gpuweb/gpuweb/blob/main/proposals/bindless.md), no shipping date), single queue (no graphics/compute concurrency, [gpuweb#1065](https://github.com/gpuweb/gpuweb/issues/1065)), and validates every indirect dispatch. Our integrated-GPU floor is what bites, not the desktop ceiling. When porting from a Vulkan reference, be more aggressive on layout, precision, and interpolator count than the reference suggests — not less.

## Types and constants

Engine code uses raw WebGPU types directly (`GPUDevice`, `GPUBuffer`, `GPUBufferUsage`, etc.). Type declarations from `@webgpu/types` in tsconfig.

- **Device wrapper:** `requestGPU(device?, features?)` in `engine/runtime/gpu.ts` populates the `Compute` singleton with the device, frame counter, the `sync()`/`pending()` frames-in-flight fence (loop backpressure), and the `buffers`/`textures`/`samplers` publish maps — **reset to empty on every call, so each `build()` wipes them**. A producer re-publishes its named GPU resources every build (the GPU analog of the registry clear-then-rebuild, `ecs.md`); a cross-build memo must still re-publish into the wiped map, never skip the publish on a hit (skipping left a textured glTF's union unbound → black on the first rebuild). Pass an external device to adopt it without floor enforcement, else the call acquires from `navigator.gpu` and enforces the floor: required = `BASE_FEATURES` (`indirect-first-instance`, `bgra8unorm-storage`, `rg11b10ufloat-renderable` — no `subgroups`, no `timestamp-query`, no texture compression) ∪ plugin `features` (a missing one throws), plus `preferred` features requested only where the adapter has them. **A `BASE_FEATURES` entry earns its place only by being a `DEFAULT_PLUGINS` need** — the floor gates acquisition for every app before a plugin loads, so an entry an opt-in plugin needs rejects hardware that would have run the scene fine; it belongs on that plugin instead (`timestamp-query` → `ProfilePlugin.features`, the BC/ETC2/ASTC families → `GltfPlugin.preferredFeatures`). `build()` passes the union of the active plugins' `Plugin.features` + `Plugin.preferredFeatures`, so `subgroups` rides in as *preferred* only when a BVH-building plugin (physics) is loaded — never gating the device, so a no-subgroup device runs physics on the LDS arm and a physics-free app requests it not at all. **`device.features.has(x)` is false for a feature the device never requested**, even on hardware that has it — so a plugin that runtime-branches over a family (the glTF KTX2 transcode's compression pick) must list *every* member in `preferredFeatures`, or the branch reads false and its fallback throw fires on a capable device. **Adoption bypasses feature resolution entirely** — an adopted device never sees a plugin's `features` / `preferredFeatures`, so a plugin with a *required* feature also guards at its use site (declare at the plugin, enforce where the feature is used — `ProfileImpl.attach` before its `createQuerySet` is the worked case), and a host that hand-acquires a shared device unions the stack's declarations itself (the gym's `gltf` scenario is the pattern). **A gate's evaluation site is part of its contract:** passing a lazy gate's result as an eager argument converts it into an eager one (`loadGltf` once passed `pickTargets(device)` eagerly into an optional param whose per-image consumers were already lazy — the throw fired on every import, masked only while the floor's compression check made it unreachable). When removing a floor entry, audit the consumers of what it guaranteed, not just the entry — every dormant assumption it propped up goes live. Nothing else; memory and profile concerns are opt-in via `ProfilePlugin`.
- **Profile hooks:** `Compute.span?: (name) => GPU...TimestampWrites | undefined` is an optional slot installed by `ProfilePlugin` in `extras/profile`. Without the plugin, every call site is a `?.` no-op. Pass authors call `ctx.timestampWrites?.("pass-name")`; encoder owners (viewport, physics) call `_profile.resolve(device)` indirectly via the profile system. `ProfilePlugin` also patches `device.createBuffer` / `createTexture` / `createComputePipelineAsync` / `createRenderPipelineAsync` for allocation + compile timing — register `ProfilePlugin` first so the patches catch every call.

## Binding limits

WebGPU `maxStorageBuffersPerShaderStage`: 8 (spec min), **10 is our hard ceiling**. 99.6% of devices support 10. Only 64% support 16 (31% on Mac). Requesting above 10 rejects `requestDevice()` on a third of users. Plan for 10, not higher.

**Per shader stage across all bind groups** — splitting into groups doesn't help reduce the count.

**Count storage buffers per stage before adding any binding.** Both `storage` and `read-only-storage` count toward the same limit.

**Reuse over add.** When a shader is at 8+/10, look for a free lane in an existing cols-buffer column before adding a new binding. Cheaper than any consolidation strategy below — no refactor needed.

Current counts (audit when touching): sear color 5 storage shared per stage (vertices/pointLights/lightGrid/lightIndices/meshQuant; indices is a hardware index buffer — `setIndexBuffer` — not a storage binding; frame/view/lighting + the group-1 shadow params are uniforms) plus the surface's own bindings — the `default`/`vertex` materials add `eids`/`transforms`/`color`/`material` = 9, the glTF importer adds `eids`/`transforms`/`color`/`materialIndex`/`materialData` = 10 (the ceiling, **zero headroom** — the next shared color binding breaks it; reuse a cols-buffer lane). The prepass/shadow pipelines bind the 8 B position-only stream at the vertices slot (no extra binding) and a separate group, so they sit at the same count, not above. `meshQuant` is the per-mesh unorm16 dequant table (gpu.md rule 6); the glTF five PBR `texture_2d_array`s + sampler are a separate limit, not storage, light compact 9 storage (membership/transforms/color/intensity/range/radius/spotInner/spotOuter read + the compacted lights write — the Spot cone bakes here), light cull 5 storage, physics collide 7 (6 storage + 1 uniform — one layout shared by the 4 shape-class pipelines, box/rounded/hull/rounded-poly), physics inertial/velocity 2 (1 storage + 1 uniform), physics primal 9 (7 storage + 2 uniform — step + dynamic-offset color; csrOffset+csrCount merged into one `csr` binding, Phase 4.9), physics coloring 9 (8 storage + 1 uniform — same `csr` merge, + the `colorCount` atomic for the readback-bounded color loop, Phase 4.9), physics compose 4 (3 storage + 1 uniform), physics compact 9 (8 storage + 1 uniform — the membership-gated dense-buffer builder), part pack 3+1u (post-`inputCols` consolidation), part compact 2+1u. The glTF surface family (the ceiling-critical one — textured / skin / skin-live at 9–10 own storage + sear's shared 5) is now **machine-audited**, not hand-counted: `extras/gltf/live.test.ts` (the storage-ceiling `describe`) registers every gltf surface device-free and asserts each own-storage + 5 ≤ 10, so a binding that breaks the ceiling goes red in `bun test` instead of failing pipeline creation with no diagnostic. Extend that audit when adding a surface rather than trusting this hand list.

### Consolidation strategies

Two orthogonal levers — don't conflate them:

**GPU-side iteration over sparse data: compute-pack, don't gather on CPU.** A JS loop over `bodyCount` / `entityCount` to scan membership and produce a dense work list is the anti-pattern. Dispatch over capacity and branch on membership (or write a tiny compute pre-pass that emits a packed index buffer + indirect args). Validated (gpu-sparse and gpu-pack both beat CPU-pack at production sparsity); the `render` gym scenario exercises this draw path every frame (the `part:pack` span). This is the firehose-not-dirty-tracking rule, *only for GPU iteration*.

**Pick the primitive by direction and shape:**

- **CPU→GPU per-entity authored data → `standard/slab`.** Declare a component field as `slab(f32)` / `slab(i32)` / `slab(u32)` / `slab(f16)` / `slab(vec2)` / `slab(vec4)`; `.set(eid, v)` writes the CPU array and marks a dirty bit. Multi-lane factories (`vec2`, `vec4`) return a Slab augmented with `.x/.y/.z/.w` lane Field handles; the master's `.gpu` is the canonical buffer for binding. `slab(u8)` / `slab(u16)` warn and stay CPU-only — WGSL has no native storage for sub-32-bit integers; pack into u32 manually. `slab(f16)` binds the native `f16` type, so its scatter shader carries an `enable f16`: the app's plugin must declare `shader-f16` in `Plugin.features` (it is not on the base floor). For four half lanes with no feature, use `slab(f16x4)` — `vec2<u32>` + `unpack2x16float`, rule 6. Slab is write-only by design — CPU is the source of truth, GPU is a synced mirror, `SlabSystem` flushes dirty slots per frame using Toji's persistent-staging + scatter compute (~2× faster than `writeBuffer` at production K, a settled measurement; the `render` gym scenario exercises the flush via its `slab:flush` span + a transport round-trip assert). Non-entity-indexed buffers (frame uniforms, event lists, indirect args, particle ring buffers) stay outside slab — consumers manage their own `GPUBuffer` + `writeBuffer` directly.
- **GPU→CPU buffer-level readback → `standard/mirror`.** Construct `mirror(buffer)`; `MirrorSystem` copies into a staging ring slot + maps it each frame, populating `instance.snapshot = { fixedTick, frame, bytes }` once the map resolves. Mirror has no opinion about what the bytes mean — it owns the staging ring + map orchestration + tick stamp only. Compaction is a consumer concern: write a smaller GPU-only buffer in your compute graph (physics' compact pos+quat pass is the pattern) and point Mirror at *that*, not at the full source.
- **GPU-only per-entity derived data → bare `capacity`-sized buffer at the consumer level.** No CPU side, no Slab, no Mirror — just allocate a `STORAGE` buffer in `warm()` and read/write it from your compute passes.

1. **Cols-buffer pattern for SoA mirrors.** Default for any AoS→SoA refactor on structures with ≥3 fields. N logical columns share **one physical binding** of `array<vec4<u32>>`, indexed `cols[col * capacity + idx]`. Each column read by a warp coalesces to a single cache line. Bitcast at the access site for heterogeneous types. Column constants live in WGSL alongside the binding. Examples: physics `bodies` (12 f32 columns, the dense solver — pos/quat/inertial/initial/vel/prevVel + moment·mass + halfExtents·friction + shape·radius, indexed `bodies[col*eidCap + i]` via `step.eidCap`; f32-first, quantization deferred per rule 8), physics `contacts` (7 f32 columns — meta/normal/rA/rB/c0/penalty·friction/lambda, indexed `contacts[col*contactCap + ci]`), `rayCols` (4 columns, RT wavefront rays + shadow rays). This is the WebGPU-portable way to do SoA at scale — bindless isn't on the WebGPU short-term roadmap.

2. **Interleave GPU-generated buffers written in the same shader.** If the same compute pass already writes both buffers per entity, merge them into one struct. Zero overhead. Example: `instances: array<InstanceData>` (`{ blasMeta: vec4<u32>, invMatrix: mat4x4<f32> }` — merged the per-entity entityBlasMeta and instanceInverses outputs of `bvh/instance.ts` into one binding consumed by closest-hit + any-hit traversal).

3. **Fold scalars into an existing buffer's header.** Counter u32s, entity counts — store at offset 0 of a related buffer. Example: `entityCount` → `shapeData[0]`.

4. **Block-concatenate CPU uploads (mixed-stride).** Two `writeBuffer` calls to different offsets of one GPU buffer. Shader accesses both regions via offset arithmetic from one binding. No iteration — each write is a bulk memcpy. Tradeoff: shader loses direct typed access (e.g., `mat4x4` becomes 4 `vec4<u32>` reads + bitcast). Only worth it when the pipeline is near the limit. Example: `entities` (96B/entity) — bytes `[0, capacity*64)` matrices region (`mat4x4<f32>` per entity), bytes `[capacity*64, ...)` cols region (`COL_DATA` + `COL_SIZE_SHAPE`); helpers `matrixOf(eid)` / `dataOf(eid)` / `sizeOf(eid)` / `shapeOf(eid)` / `hasShadows(eid)` in `ENTITY_COLS_WGSL` (exported from `render/core`); capacity in `scene.capacity` for offset arithmetic.

5. **Split the pass.** When a shader genuinely needs >10 buffers, break it into two compute passes with intermediate results. Last resort — adds latency.

**Don't consolidate:** GPU buffers generated in different passes (sortedIds from radix sort + leafAABBs from AABB compute), mixed CPU/GPU buffers (entityIds from batch shader + shapes from CPU upload), or buffers whose ownership is shared with utilities that require exclusive access (radix sort ping-pong).

### When you hit the limit

Never silently exceed 10 — Chrome fails with zero diagnostics. Don't add a separate bind group thinking it solves the problem (it doesn't — same per-stage limit). The cols-buffer pattern (above) is the first thing to reach for on SoA work; pass split is last resort.

### The cols-buffer pattern in WGSL

```wgsl
const COL_ORIGIN: u32 = 0u;
const COL_DIRECTION: u32 = 1u;
const COL_PIXEL_FLAGS: u32 = 2u;
const COL_THROUGHPUT: u32 = 3u;
@group(0) @binding(0) var<storage, read_write> rayCols: array<vec4<u32>>;

fn readOrigin(idx: u32) -> vec3<f32> {
    let v = rayCols[COL_ORIGIN * params.capacity + idx];
    return vec3<f32>(bitcast<f32>(v.x), bitcast<f32>(v.y), bitcast<f32>(v.z));
}
```

A warp of 32 threads reading sequential `idx` per column accesses 32 sequential vec4u = 4 sectors = 1 cache line. Same coalescing as bodyCols. Total bindings unchanged regardless of column count — the limit is on physical bindings, not logical columns.

## On-GPU data layout & access patterns

Bandwidth, not compute, is the dominant cost in most of our compute passes (RT shade, physics solver, render light loops). Layout is the lever. Diagnose layout debt by comparing measured pass time to its theoretical bandwidth floor (`bytes_moved / peak_BW`). **≥5× off peak means stride-induced sector waste, not arithmetic.** See "Bandwidth ceiling check" below.

### Coalescing, in numbers

Modern GPUs fetch global memory in **32-byte sectors**, four sectors per 128-byte cache line. A warp issues one memory request; hardware translates to N sector transactions.

| Access pattern (warp of 32, 4B each) | Sectors fetched | Bandwidth used |
|---|---|---|
| Contiguous, aligned | 4 | 100% |
| Misaligned by 1 sector | 5 | 80% |
| Stride 8 (32B AoS) | ~32 | ~12% |
| Stride 32 (128B AoS) | 32 | **3%** |

A 48B-stride AoS read by a warp touches **32 cache lines** to read 32 floats. 5–8× sector waste is the signature when each warp reads only one or two of the struct's fields. When the warp reads ALL fields, AoS coalesces nearly as well as SoA (microbench: AoS read-all-N at 1.03× floor on Ada), so SoA only wins on partial reads.

### Layout rules

1. **Default to SoA for buffers read by many parallel threads.** Adjacent threads → adjacent memory → coalesced. AoS is acceptable only when the hot loop reads ≥4 fields per entity AND a profile shows SoA is cache-thrashing (4 separate strided streams beat the AoS request count). Use the **cols-buffer pattern** (Binding limits → Consolidation strategies #1) for the SoA storage shape — N logical columns share one physical binding, so SoA refactors don't blow the per-stage binding ceiling.
2. **Hot struct cap = 64B.** Above that, partition fields by access frequency: hot (read every tick by the bandwidth-bound pass) and cold (debug, structural, rare branches). Physics already does this — `Body` is 208B AoS, primal solver reads from a 64B `bodyCols` SoA mirror (4 cols × 16B), ~32% perf delta. Same pattern applies engine-wide via the cols-buffer convention.
3. **AoSoA tile = 32 elements.** Earns its place only after SoA proves cache-bound on a measured workload. Tile = warp size (NV 32, AMD 64; pick 32 for portability — under-utilizing AMD wavefronts is cheaper than splitting NV warps). Don't lead with it.
4. **`vec3` in storage = 16B stride, always.** Pack as three scalars or `vec4` explicitly. `array<vec3<f32>>` has stride 16, not 12 — silent killer. Verify `byteLength` matches `count × expected_stride` at write time.
5. **Avoid power-of-2 strides ≥128B.** They correlate with DRAM channel hashing and L2 set indexing. Pad row pitch by one element when 2D buffer width is a power of 2 (e.g., 1024-wide row → pad to 1056 or 1025).
6. **Format choice: range × consumer tolerance. f32 is the exception, not the default.**

   **Fixed-point (unorm / snorm) beats f16 in any bounded range.** unorm16 in [0,1] has uniform 1.5e-5 precision; f16 in the same range has 0.05% relative — coarse near unit values, useless near the high end. Reach for fixed-point first; use float formats only when range is genuinely unbounded.

   | range | best 16-bit format | precision |
   |---|---|---|
   | [0, 1] | **unorm16** | 1.5e-5 uniform |
   | [-1, 1] | **snorm16** | 3e-5 uniform |
   | [0, scale] (known scale) | **unorm16 × scale** | 1.5e-5 × scale uniform |
   | unbounded HDR magnitude | **f16** | ~0.05% relative (varies absolutely with magnitude) |

   **Storage format defaults by category:**
   - **LDR colors** → 8-bit-per-channel `u32`, sRGB-encoded at storage. Storage: `LDR_COLOR_PACK_WGSL` from `engine/utils/encode.ts`. Read sites: `LDR_COLOR_UNPACK_WGSL` (sRGB→linear in unpack). Never `vec4<f32>`.
   - **HDR colors** → r11g11b10ufloat manual u32 pack (`rgb9e5ufloat` is read-only in WebGPU per [gpuweb#957](https://github.com/gpuweb/gpuweb/issues/957)). Storage: `HDR_COLOR_PACK_WGSL` from `engine/utils/encode.ts`. Read sites: `HDR_COLOR_UNPACK_WGSL`. ~3% relative precision per channel, range [0, 65024].
   - **Normals** → oct-encoded `u32` (12B → 4B) for *storage* only (decoded once per vertex). `OCT_ENCODE_WGSL` from `engine/utils/encode.ts`, Cigolle et al. 2014. **Never oct for an interpolated/filtered normal** (VS→FS varying, VAT texture) — the seam breaks under interpolation (rule 9): cross those as a plain `vec3` and renormalize.
   - **Quaternions** → smallest-3 (10-10-10-2 packed u32, 16B → 4B). Storage: `SMALLEST3_WGSL` from `engine/utils/encode.ts`. ~0.1° max error, free decode.
   - **Object-space positions** → unorm16 with per-mesh AABB scale/offset uniform.
   - **World-space positions, velocities, accelerations** → f32. Range exceeds f16 at planet scale or fast motion.
   - **Instance transforms** (the `transforms` firehose: per-entity world `{pos, quat, scale}`, `XFORM_WGSL`) → **f32-decomposed, never quantized.** Decided on first principles (2026-06-14, not gated on a measurement): the per-instance read is cache-amortized (one 48 B AoS line per *instance*, the VS reads it flat-per-instance — not bandwidth-bound on any GPU), translation is world-scale (f32 regardless), and no engine quantizes instance transforms (niagara / Bevy / Unreal all f32). Quantizing only the rotation + scale lanes saves ~22 B/entity behind that cache, against precision risk to the frustum-cull bounds, the normal reconstruct, and physics interpolation — a bad trade. Decompose-on-read is the final form; don't re-propose snorm16/f16 here.
   - **Bounded [0,1] scalars** (roughness, reflectivity, opacity, AO, attenuation) → unorm8 via `pack4x8unorm` — 1 byte, uniform 1/255 spacing (vs f16's 2 bytes, non-uniform). Use `f16x4` instead when a [0,1] lane shares a 4-lane component with an HDR lane (`Material`: emissive is an unbounded glow strength, so the whole vec4 mirrors as f16). **Half-precision *storage* needs no feature — only half-precision *declarations and arithmetic* do.** `shader-f16` gates the WGSL `f16` type; `pack2x16float` / `unpack2x16float` are core WGSL and predate it, so `f16x4` binds `vec2<u32>` and unpacks (`Material` is the worked case) — same 8 bytes, bit-identical, and `shader-f16` stays off the floor. Reach for the `f16` type only when the arithmetic itself must be half.
   - **Bounded scaled scalars** (friction [0,2], halfExtents within scene scale, mass with scale uniform) → unorm16 with scale.
   - **Tangent frames / bounded directions** → snorm8 via `pack4x8snorm` (~0.008 cosine error) or oct.
   - **Lighting / postfx intermediates** (NdotL, falloff, soft shadow factors) → f16. HDR-shaped, range varies, fixed-point doesn't fit.
   - **Flag enums, IDs, small integers** → bit-packed u32. Don't allocate a u32 per boolean.

   **Quantization is a storage-boundary concern, not a register concern.** Math runs at f32 in registers (default WGSL); quantized fields pay one decode at load + one encode at store, not pervasive precision loss through the inner loop. Solver state errors don't compound through f16 storage when reads happen once per iteration and the field range fits.

   **Use f32 storage when ANY of these apply:**
   - **Sentinel-laden** — value compared against thresholds where saturation flips the comparison (`stiffness >= 1e30` hard-vs-soft classifier; `fmin/fmax = ±1e30` joint markers; `penalty` clamped to 1e10). f16 saturation past 65504 → +inf trips these silently. Validated 2026-05-08: f16 packing of constraint `C_init_*` broke ball-joint stabilization in the bridge scenario.
   - **Unbounded accumulator** — Lagrange multipliers, penalty accumulators that grow across iterations with no renormalization.
   - **World-scale unbounded** — world positions, velocities, accelerations. f16 has 0.001 precision at unit 1, 64 at unit 65000.
   - **Time-cumulative** — values that compound across substeps without renormalization.
   - **Iter-mutated state read by a downstream pass that finite-differences against a lossless reference** — e.g. body `quat` mutated each primal iter, then read by `computeVelocities` to compute `angVel = 2 * (quat * inv(initialQuat)).xyz / dt`. The precision of the storage encoding feeds *directly* into spurious angVel for a body at rest. Validated 2026-05-09: smallest-3 (~0.1° per round-trip) → ~0.12 rad/s spurious → visible body slide; snorm16x4 (~3e-5 per component, ~0.005° angle) → ~0.013 rad/s spurious (9× tighter, 37× under solver's recoverable threshold) — bridge / box / stack / pile-10k all clean, shipped in `bodyCols.QUAT_OWNED`. Smallest-3 stays fine for narrowphase + per-pair neighbor reads (no compounding) in `bodyCols.QUATS`. Microbench in `tests/encode.test.ts` "snorm16x4 quaternion".

   **Per-field precision audit, not per-category.** A field shared between callers with different value ranges (e.g. constraint `C_init_n/t1/t2` is a contact penetration in some paths and a 3D joint anchor offset in others) must be packed for the worst-case caller, not the typical one. Audit every emit site before packing; never assume "bounded by halfExtents" without checking.

   **Industry-default-f32 in physics** (Bullet, PhysX, Havok, AVBD reference) reflects portable-library / CPU-first conservatism, not a first-principles "f16 fails in physics" finding. Bounded fields with range that fits tolerate quantization fine; we are not bound by their defaults.

   **"Lossless to the eye/sim" beats "lossless to the bit."** Storage bytes are the bandwidth bill — f32 quadruples it vs packed u8.

7. **Decode intrinsics are bandwidth-free in DRAM-bound regimes.** Validated 2026-05-08 against a measured 4B/elem bandwidth floor: `unpack2x16float`, `unpack2x16snorm`, `unpack4x8unorm`, `unpack4x8snorm`, oct decode, smallest-3 quat unpack (sqrt + 4-arm switch), and the constraint solver's tangentBasis derivation all coincide with the 4B/elem floor. The decode pipelines into memory-fetch latency. **Exception: `exp2` (log-domain decode) is NOT free** — measured +52% wall time over plain f16x2. Use log-domain only when range demands it; affine encode (linear in a bounded range with scale uniform) is the default.

8. **Bench in isolation before applying.** This governs *quantization* specifically — a storage-format change whose win is working-set- and hardware-dependent — not unified-path design. The principled, optimal *path* (e.g. culling every view through one spine, SoA layout, "Instance transforms") is decided up front on first principles, never gated on a profile; only a quantization choice or other **branching** optimization waits for the measurement (coding.md "Complete over incremental"). Measure each quantization variant against a computed bandwidth floor in a throwaway `.lab.ts` harness — the floor-ratio diagnostic batches re-submits vs a peak bandwidth, a one-shot derivation, not a living surface (it never mapped onto a per-frame GPU span). `bun bench` — the gym — is the standing real-GPU surface, for per-frame spans + correctness. Apply to production code only when (a) the production pass is diagnosed bandwidth-bound (≥5× off floor) AND (b) the per-field audit is clean AND (c) the encoding is microbench-validated to be free or near-free. Historical: a solve-stress harness (git-only) confirmed the constraint column-drop (10→8) wins -24% wall time at 2M-constraint working set (DRAM-bound) and is null in the L2-resident regime — committing to the principle is correct even when the production bench shows null.

9. **VS→FS interpolators are per-pixel bandwidth with no L2 escape.** Each `@location` gets interpolated and consumed every fragment; working set scales with shaded fragment count and bypasses the per-color tiny-working-set cache behavior physics enjoys. Quantization wins on interpolators land on 4090 directly (Phase 1c VertexOutput compaction shipped a measured -11% on `raster-forward` at lorenz 10k). Budget:
   - Pass `@interpolate(flat) instanceIndex` (or eid) and read constants from an SSBO indexed by it. Cheaper than re-emitting per-vertex.
   - Don't pass anything reconstructible from `@builtin(position)` + a uniform. World-space view ray, NDC, screen UV — all derivable.
   - Don't pass both world- and object-space when one transform recovers the other.
   - Pack multiple scalars into vec4 slots. **Never oct-encode a normal (or any direction) for *interpolation*** — octahedral encoding is discontinuous across the seam, and linearly interpolating the 2D oct coords across a triangle whose vertex normals straddle it decodes to garbage (symptom: a jagged normal zigzag on curved/draped geometry oriented to straddle the seam — wrong-direction normals that read white under specular and black under diffuse). Interpolate the **plain `vec3` normal and renormalize in the FS**. A `vec2` fills a whole `@location` slot anyway (locations are vec4-granular), so oct saves *nothing* on an interpolator — it's pure downside. Oct stays correct for per-vertex *storage* (`OCT_ENCODE_WGSL`, decoded once per vertex, never interpolated); a *filtered* oct texture has the same seam hazard, which is why the VAT normal map is stored as a plain `rgba16float` vec3 (renormalized in the VS), not oct.
   - Hard budget: 4 custom interpolator slots per surface before justification. Sear's built-ins are a **per-surface** prefix (the `vec3` worldNormal + `eid` + `world` always; `uv` / `localPos` only when the fs reads them); custom interpolators pack after them, capped at 16 total (render.md "Surface authoring"). An unread built-in doesn't cross — sear prunes it, so don't reference `uv` / `localPos` in an fs that doesn't need them.

### Atomic and shared-memory rules

- **Counter atomicAdd with low contention is fast.** L2-cached `atomicAdd` against a single counter scales well even at ~1M ops/frame (RT shade shadow-ray emit ≈0.01 ms). Don't preemptively batch.
- **Reductions to a global atomic — reduce inside the workgroup first, subgroup-first.** atomicMin/Max/Add called from every thread to a single global slot serializes. The floor has `subgroups`, so the reduce is a subgroup op, not an LDS tree: each lane folds its slice → `subgroupMin`/`subgroupMax`/`subgroupAdd` across the subgroup → lane 0 publishes the per-subgroup partial to a small `var<workgroup>` array → a second subgroup op folds the partials (they fit one subgroup when `numSub <= sgsize`, true for sgsize ≥ 16) → `if (localId == 0u)` does ONE atomic per slot to the global. WebGPU has integer atomics only, so f32 extremes reduce as order-preserving u32 (sign-bit flip on positives, all bits on negatives) and decode after. `standard/bvh/bounds.ts` is the canonical reference. Per-thread atomic reductions, and an LDS tree where a subgroup op suffices, are the antipatterns — fix them when you find them. The one deliberate exception: a subgroup-free LDS reduce as the fallback for the no-`subgroups` (WebKit) tier — `bvh/bounds.ts` ships both arms behind a flag, so don't delete the LDS one on sight.
- **`var<workgroup>` (LDS) only when reuse ≥3 per global load.** Stencil-shaped kernels (blur, SDF, Jacobi) qualify; one-pass copies/reductions don't. Cap LDS per workgroup at ≤8KB unless occupancy is measured fine — high LDS use is the most common cause of "many warps, low utilization."
- **Pad LDS tile widths to (warp_size + 1) = 33** for column accesses. Avoids 32-way bank conflicts. Shared memory has 32 banks × 4 bytes; same address = broadcast (free), different addresses on same bank = serialized.
- **Subgroup width is 32 or 64 — design for both, never assume a value.** Read `subgroup_size` at runtime; size thresholds off it (`subgroupSize / 2`, etc.). 32 = NVIDIA/Intel, 64 = AMD/RDNA (Steam Deck). Software rasterizers (llvmpipe/lavapipe, the WSL `bun test` adapter at width 8) are **out of scope** — don't gate on them, don't tune for them. Real-GPU validation is `bun bench` (the gym) via Playwright → native Chrome; `bun test` is for CPU-reference and ECS logic, not subgroup correctness.
- **Subgroup ops in data-dependent loops: filter the uniformity diagnostic, cap the loop.** WGSL's uniformity analysis rejects a subgroup op (`subgroupShuffle`/`subgroupBallot`/…) inside any loop or `if` whose condition derives from a subgroup-reduction result — Tint treats *every* such result as non-uniform for control flow, and there is no in-analysis workaround (fixed-count + `if (gate)` taints on the gate; routing the count through a function return taints on the function's other non-uniform effects). When the loop *is* subgroup-uniform at runtime (its condition is a reduction result, so all lanes iterate in lockstep and every lane reaches the op), suppress the false positive with a global `diagnostic(off, subgroup_uniformity);` — the sanctioned opt-out (WGSL subgroups proposal, honored on Dawn). **Always pair it with a fixed iteration cap** proven ≥ the worst case, so a logic error degrades to a wrong result the gate catches, never a GPU-watchdog hang. The alternative (fixed-count with no early-out, no-op tail) is correct but catastrophically slow — every no-op step still runs the shuffles (measured ~140× in a subgroup-merge build). This is a uniformity-*analysis* limit, not a missing op — WGSL has the full subgroup set.
- **Cross-workgroup ordering: WGSL has no device-scope fence.** `storageBarrier()` is workgroup-scope; there is no `__threadfence()` equivalent. A producer/consumer handoff across workgroups (write data, signal a flag, another workgroup observes the flag then reads the data) is **not** spec-guaranteed even with atomics — WGSL atomics are relaxed, with no acquire/release across locations. **Cross a dispatch boundary — the only correct option.** Writes from one dispatch are visible to a later dispatch; structure any cross-workgroup producer/consumer as multi-dispatch (`standard/bvh/build.ts`'s LBVH bounds relaxation is the worked example — a bottom-up fit where each sweep is its own dispatch). The tempting alternative — same-kernel atomic flag + atomic data relying on hardware L2 coherence — is **not reliable, even on NVIDIA**: the single-kernel H-PLOC BVH build used it (a non-spin-waiting climb) and intermittently produced wrong trees on the dev Lovelace GPU (the forest shadow flicker), forcing the rewrite to multi-dispatch. Relaxed atomics give per-location coherence but no acquire/release, so observing a flag does not publish the separate data writes — gpuweb#2229 (no acquire semantics in WGSL), #3935 (barriers can't cross workgroups). Same wall the naive decoupled-lookback scan hits. Don't reach for the single-kernel handoff; pay the dispatch boundary.
- **The decoupled-scan exception — when a single-kernel cross-workgroup handoff IS correct.** The wall above is a flag and its *separate* data: observing the flag doesn't publish the data. A chained scan sidesteps it by packing `value << 2 | flag` into **one** atomic word — per-location coherence (which WGSL relaxed atomics *do* give) makes the read see both consistently, no cross-location ordering needed. Pair it with a **fallback** (a stalled workgroup recomputes the predecessor from stable prior-dispatch input instead of spinning on forward progress) and the scan is correct without device-scope sync: Decoupled Fallback (`reference/GPUPrefixSums` csdldf.wgsl; shallot `standard/bvh/sort.ts`). Its early-exit loop — a `workgroupBarrier` inside a loop that exits on a shared flag — is illegal if the flag is read by `atomicLoad` (Tint flags non-uniform control flow), so gate the loop on **`workgroupUniformLoad(&flag)`**: a control barrier whose result the uniformity analysis treats as uniform, making the in-loop barriers legal. (A fixed-count loop is the wrong workaround — every workgroup then runs to full length.) The dispatch-boundary rule still governs everything else; this is the one structure engineered around it.

### Dispatch count is a first-class cost

Per-dispatch overhead is fixed and paid at CPU **encode** time (submit, command validation, pass setup) plus a GPU-side per-pass launch and the memory barrier WGSL forces between dependent passes, not GPU compute. Measured over a 64–8192 dependent-dispatch sweep: a primal-shaped dependent dispatch is **~1.1 µs to encode / ~5.9 µs all-in** (encode→GPU-done) on desktop D3D12 (4090) — the page-side encode is only a floor (Chrome's GPU-process Dawn/D3D12 build is async, invisible to a page timer; the all-in is the honest figure), and a weak-CPU integrated / Steam-Deck part is multiples higher. A structure that fans into many small dispatches pays this *per dispatch, regardless of N*, so it dominates at small N and is binding against a tight frame budget. Signature: a pass's wall-clock flat across a wide N range, far above its bandwidth floor.

- **Minimize passes, not just bytes.** Collapse to the fewest dispatches still correct under "Cross-workgroup ordering". A job whose data fits one workgroup's LDS is **one** dispatch (`workgroupBarrier`); spread across blocks it needs one dispatch per cross-workgroup sync point. A 16-dispatch sort over 12 keys is overhead for parallelism that doesn't exist.
- **Indirect ≈ 2× direct.** `dispatchWorkgroupsIndirect` triggers Dawn's injected validation pass per call (mechanism in "WebGPU-specific traps" below); measured ~2× the direct per-dispatch cost. Prefer direct dispatch when the count is CPU-known; reserve indirect for the GPU-count path.
- **Count-gated path selection fights the GPU-count contract.** A "fast path when N is small" must commit at *encode* time, but a GPU producer's count lives only on the GPU. Zeroing a path's workgroups via indirect args still pays its encode + validation, so there's no cheap runtime branch. Dispatch reduction for a dynamic producer must be **unconditional** (fewer passes for all N); only a static, small *declared cap* can commit to a one-workgroup path.

### Branch discipline

Hardware uniforms-out branches it can prove are convergent (params, constants, dispatch-index bounds checks where all threads in the warp agree). Anything data-dependent that varies within a warp serializes both paths.

- **`continue`, `break`, `return` mid-loop in inner kernels diverge.** Default: compute both paths, mask with `select` / multiply, **for ALU-bounded loop bodies**. For memory-/sample-/traversal-heavy bodies (BVH inner loop, ray-march steps, expensive texture taps), early-out can win — predication forces both paths to load. The lensflare ghost loop (`extras/lensflare/index.ts:72-85`) is ALU-bounded enough that masking wins. BVH traversal is memory-bound and stays branched. When in doubt, bench both. Reference: [aschrein on branches on GPU](https://aschrein.github.io/jekyll/update/2019/06/13/whatsup-with-my-branches-on-gpu.html).
- **`if (x) { ... } else { ... }` with both sides cheap → `select(b, a, cond)`** or `mix(a, b, f32(cond))`. WGSL emits a single instruction; no branch.
- **Early-return from out-of-bounds dispatch threads is fine** — the warp tail has uniform agreement, hardware skips them cleanly.
- **Uniform branches over uniform values are fine** too — the same constant for every thread folds.
- **Loops with constant upper bounds and dynamic `break`** are usually fine on Tint/Naga, but in hot inner loops verify the unroll happened (read the compiled HLSL/MSL or bench an explicitly-unrolled variant). WGSL has no `[[unroll]]` — for small fixed counts, write the body N times.
- **Compilers can re-order `a*b + c` away from FMA, but most of the time they already emit it.** Bench before sprinkling `fma(a, b, c)` calls — measured 2026-05-08 on the AVBD primal solver: ~30 fma conversions in solve6 + C/f sites showed null runtime delta on `phys:solve` and a 33% DXC compile-time hit on `solvePrimal`. The rule survives only for cases where you can verify the compiler is *not* fusing (read the compiled HLSL/MSL or bench an explicit-fma variant against the original); blind conversion is a tax with no payoff. Where many small `a*b+c` sums sit in scalar form (e.g. `j.x*F.x + j.y*F.y + j.z*F.z`), prefer `dot()` — clarity win and the compiler reliably fuses it.

### Diagnostic methodology

When a compute pass is slow, **don't theorize — strip it.** Progressively remove pieces from the kernel and bench each variant. Each delta isolates one cost. RT shade's strip-down found 70% of cost was input read, not compute or atomics — opposite to intuition.

Pattern:
1. Strip to dispatch + early return. Floor.
2. + Read inputs. Adds bandwidth.
3. + Atomic write to outputs. Adds atomic-write contribution.
4. + Compute. Adds arithmetic.
5. Layer back specific suspects (shadow ray emission, branch divergence, etc.) one at a time.

Restore the kernel before committing — the variant scaffolding is for diagnostics only.

### Bandwidth ceiling check

Before declaring a pass "as fast as it gets," compute the bandwidth floor:
```
bytes_moved = total_reads + total_writes per frame
peak_BW = device peak (4090: ~1 TB/s; integrated: ~50 GB/s)
theoretical_min_ms = bytes_moved / peak_BW
```
If measured ≥5× theoretical_min, the pass is layout-bound, not compute-bound. Look at stride and coalescing before optimizing arithmetic.

**Absolute wall time gates desktop measurability.** A pass at 12× off floor is a real bandwidth candidate, but if its absolute time on the 4090 is sub-0.1 ms, the available saving (e.g. 4× → 0.05 ms on a 0.07 ms pass) is below the bench's run-to-run variance floor. The diagnosis isn't wrong; the desktop bench just can't see it. Validate sub-0.1 ms candidates via a synthetic isolated-stress `.lab.ts` harness — pump fixed inputs through both layouts and isolate per-element fetch cost from dispatch / atomic / emission overhead. Don't declare "no win" from the desktop production bench alone — the diagnosis-vs-bench mismatch is the signal that the calibration bench is the right tier. Validated 2026-05-08: narrowphase cols-buffer SoA migration (12× off floor) showed null on 4090 pile because the pass was 0.07 ms; saving was bench-noise.

**The L2-cache caveat. The bytes_moved/peak_BW floor uses DRAM bandwidth — it under-estimates when working set fits in L2.** Ada AD102 (4090) has **98 MB L2**; Ampere GA102 (3080/3090) has 6 MB; Apple M-series shared cache is ~12 MB; integrated GPUs are typically 2–8 MB. If `working_set ≤ L2_size`, the production pass isn't waiting on DRAM — it's waiting on L2 latency, which is much faster but finite.

**For diagnosis, compute working-set size first, then pick the cache tier whose bandwidth defines the floor.** `working_set = bytes per logical work unit × work units active in flight per pass`. For graph-colored solvers, "in flight" is the per-color slice, not the whole buffer — even a 100k-body pile has ~1 MB active at any moment. For scattered scene reads (raster forward FS, RT shade), working set is per-frame consumed bytes.

**For application:** quantization on a desktop discrete GPU often shows null/regressive on small-scene production benches because the working set fits in L2 — but the principle still applies for deployed portability (integrated GPUs and bigger scenes hit DRAM). A synthetic isolated-stress `.lab.ts` harness is the calibration bench; the production pile / lorenz 10k is the secondary validation. Don't bench-revert correct quantization on desktop alone. Per-fragment data is the exception — no L2 escape, wins land on 4090 directly (rule 9).

### Sources for deeper reading

- [Toji's WebGPU Best Practices](https://toji.dev/webgpu-best-practices/) — bind groups, indirect draws, render bundles, buffer uploads. The WebGPU-specific reference.
- [Linebender GPU sorting wiki](https://linebender.org/wiki/gpu/sorting/) — the WebGPU sorting reference (Vello/Levien). Why plain decoupled-lookback/Onesweep deadlocks without a forward-progress guarantee (Apple) — the grounding for "Cross-workgroup ordering" + the "decoupled-scan exception" (Decoupled Fallback is the portable fix) above.
- [Modal GPU Glossary](https://modal.com/gpu-glossary) — hardware/memory hierarchy, roofline model, warp divergence. Domain-agnostic systems reference.
- [NVIDIA blog — Unlock GPU Performance: Global Memory Access](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)
- [WebGPU Data Memory Layout — webgpufundamentals](https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html)
- [Aaltonen-Haar GPU-driven rendering 2015](https://advances.realtimerendering.com/s2015/) — indirect-draw + cluster-cull pipeline, the reference for GPU-driven design.
- [zeux/niagara](https://github.com/zeux/niagara) — modern Vulkan reference renderer with mesh shaders, Hi-Z, GPU-driven pipeline.
- [Aras Pranckevičius — Compact normal storage](https://aras-p.info/texts/CompactNormalStorage.html) — octahedral encoding survey.
- [aschrein on branches on GPU](https://aschrein.github.io/jekyll/update/2019/06/13/whatsup-with-my-branches-on-gpu.html) — predication vs divergence empirical writeup.
- [Unity DOTS chunk layout](https://www.youtube.com/watch?v=_VsRD392YIM) (pseudo-SoA-within-chunk pattern)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — coalescing, occupancy, shared memory
- [NVIDIA Ada GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) — L1 (128 KB/SM combined), L2 (98 MB on AD102), 128B L1 transactions, 32B L2 transactions. Authoritative source for the L2-cache caveat above.

## DXC shader compilation

DXC (Chrome on Windows) is the compilation bottleneck:

- **Never call large functions inside dynamic loops.** BVH traversal in a dynamic loop = 10x+ slowdown
- **Dead code isn't free.** DXC doesn't DCE. Duplicated traversal doubles compile cost
- **Dynamic loop bounds hurt.** Constant upper bounds with dynamic `break` are fine
- **Chrome shader cache handles repeat visits.** Optimize via pipeline splits (separate shader per code path)

## Render passes on TBDR (Apple Silicon, mobile)

Apple Silicon GPUs are tile-based deferred renderers; `beginRenderPass` pays a tile-memory load/store per attachment, charged at main-memory bandwidth. Immediate-mode renderers (Windows D3D12, Linux Vulkan) pay no such cost. A fragment workload that reads 0.2 ms on a 4090 reads 3–6 ms on an M-series Mac at retina, and the gap closes by *removing passes*, not optimizing within them. Measured on a consumer-app bench at simulated 1440×900 logical retina: 8 sear+post passes total ~6 ms; per-pass tile setup is ~50–150 µs of that, the rest is fragment fillrate × Apple's ~3 TFLOPs vs a 4090's ~36 TFLOPs.

- **Depth prepass is counter-productive on TBDR** — [Imagination/PowerVR docs](https://blog.imaginationtech.com/powervr-performance-tips-for-unreal-engine-4/) explicit: TBDR HSR culls overdraw in tile memory before the FS runs; the prepass forces depth through main RAM for no benefit. On a discrete GPU it buys ~1–2% overdraw reduction — fair trade, not a load-bearing optimization. A renderer that ships to both drops the prepass and lets the color pass write depth directly.
- **Fuse fullscreen post-FX into one compute pass.** Each render pass for a screen-space effect costs one tile load + one tile store. Chaining N effects costs N × that. WebGPU doesn't expose Metal's programmable blending / tile shaders ([gpuweb proposal](https://github.com/gpuweb/gpuweb/issues/64), no shipping date) — the portable substitute is one compute dispatch reading every input texture and writing the swapchain once. Halves bandwidth, removes per-pass setup.
- **MRT-fold what's morally one pass.** A second attachment in tile memory is cheaper than a second render pass with a second fragment shader. Color + tag from one surface → MRT, not two passes.
- **DPR scales fragment cost linearly.** shallot's canvas defaults to `pixelRatio: "auto"`, which clamps `devicePixelRatio` to `[1, 2]` (`render/view.ts`) — the react-three-fiber standard. At native DPR 2 each fullscreen pass is 4× the cost of DPR 1; the cap doesn't save a DPR-2 Mac (it renders native), so a fullscreen app that feels the cost opts down with a fixed `build({ pixelRatio: 1 })` (CSS resolution, the old default).
- **`loadOp: "load"` is the explicit cost; `loadOp: "clear"` is half-price; `storeOp: "discard"` skips the writeback.** Audit every render pass's load/store actions when touching them — a transient target (e.g. the depth attachment that's only read within the same pass) wants `discard` to keep it in tile memory.

References:

- [Imagination — PowerVR UE4 tips](https://blog.imaginationtech.com/powervr-performance-tips-for-unreal-engine-4/) — depth prepass anti-pattern, explicit
- [Apple — Tailor your apps for Apple GPUs and TBDR](https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering)
- [Crosley — Apple Silicon TBDR: What App Developers Actually Get](https://blakecrosley.com/blog/apple-silicon-tbdr)
- [Heinäpurola — Engine Internals: Optimizing for Metal](https://medium.com/@heinapurola/engine-internals-optimizing-our-renderer-for-metal-and-ios-77aeff5faba)

## NaN policy

Don't branch around NaN/Inf unless a downstream consumer requires deterministic finite values (BVH centroid, atomic reduction sentinels). Compute through; let arithmetic propagate. The `if (hasNaN(...)) { sentinel } else { value }` shape costs more than the rare NaN it guards against. Fix the source if NaNs are happening; don't wallpaper the consumer.

## GPU debugging

GPU shaders can't print, log, or break. The **only** way to know what a shader computed is to write values to a buffer and read them back.

### Methodology

1. **Pick the exact value you're uncertain about.** Not "is broadphase working" but "what is `body.pos.y` for entity 0 after the primal pass"
2. **Write it to a debug slot.** `atomicStore(&debug[SLOT], bitcast<u32>(value))` in solver WGSL
3. **Read it back.** `gpu.debug[SLOT]` — the debug buffer is read back every frame as a `Float32Array`
4. **Compare actual vs expected.** Fix assumptions or trace upstream
5. **Repeat upstream** until actual matches expected
6. **Clean up.** Remove debug writes

### Discipline

- Verify actual values via readback before changing shader code. Off-by-one, wrong binding, stale bind group all look correct in source.
- Apply one fix at a time. Confirm the bug, then change.

### Existing infrastructure

- **Physics readback:** `requestReadback` dispatches a compact compute shader (extracts pos+quat → 7 floats/body), then copies counters + compact buffer to staging. `processReadback` warns on all overflow/saturation conditions and syncs transforms to ECS. Tick is captured at dispatch time
- **General:** `readBuffer`, `readFloat32`, `readUint32` in `engine/utils/readback.ts`
- **Profiling:** opt-in via `ProfilePlugin` (`extras/profile/index.ts`); see the "Types and constants" section above

## GPU profiling

Every compute/render pass must include `timestampWrites`. Compute passes use `beginComputePass(encoder, ctx.timestampWrites?.("pass-name"))`. Same-named entries accumulate.

`"timestamp-query"` is **not** on the platform-support floor — it's `ProfilePlugin.features` (required, so an explicitly added profiler fails loud on a device that can't time). Without `ProfilePlugin`, no device requests it, `Compute.span` is undefined, and every `ctx.timestampWrites?.()` is a `?.` no-op.

Timestamps bracket **pass begin→end only** — there is no `writeTimestamp` (removed from the spec), so time *between* passes (barriers, buffer copies, swapchain acquire, present pacing, **Dawn's injected indirect-draw validation**) is invisible to them: a frame whose passes sum to ~1ms can still cost several ms of GPU wall time, and that gap surfaces as fence wait, attributable to no pass. **Don't try to recover it with timestamps** — a `frame-span − Σpasses` "bubble" conflates real GPU stalls with present/pacing idle and misleads (it also depends on submit order: a separate early submit like the slab flush skews the span). The per-pass numbers are the accurate signal; the frame total is the fence wait. One untimed cost is *predictable* (not the same as recoverable-from-fence) — Dawn's indirect-draw validation floor — by **counting commands, not subtracting timestamps**: `ProfilePlugin` installs `Compute.indirect?.(name, count)` (a `?.` no-op without the plugin, parallel to `span`), each indirect-issuing pass reports the draws it issues (a bundle reports its *recorded* count — the replay validates the same), and the profiler derives `count × INDIRECT_FLOOR_US`, shown as a named per-pass + Σ line **in the gpu section** (it's a GPU cost). It does **not** sit under fence wait: fence is a pipelined residual (GPU frame time minus CPU/GPU overlap, beating against pacing), so it swings frame to frame and can read *below* the floor when the frame isn't GPU-bound — the floor is a deterministic GPU-timeline cost that only *surfaces* in fence when GPU-bound. So the floor is read against fence's trend (or the ablation below), never summed into a single frame's fence. Its real jobs: quantify the no-`multi-draw-indirect` tax and act as a regression tripwire (a ballooning draw count jumps the line, attributed, instead of an unexplained fence spike). Calibrated via the `floor` lab scenario (`INDIRECT_FLOOR_US` ≈ 1 µs/draw): the floor surfaces in fence **only above the GPU-bound threshold** — below it the validation hides in rAF/vsync idle, so a non-GPU-bound box can't read it from a real scene's fence; `floor` forces GPU-bound by cranking the probe count past the frame interval, no special hardware. The **direct-vs-indirect swap** is **not** the calibration tool on a GPU-driven renderer — the per-draw counts are GPU-computed, so a direct path needs a fence-polluting readback or an inflating approximate count; the redundant no-op draw is the faithful instrument.

**Localize the untimed cost in-repo by ablating its trigger against fence wait — a native capture is the last resort, not the first.** The fence (`pendingFenceWaitMs`, on the bench/`__benchmark` measure) is GPU-completion latency, so a controlled swap that changes one suspected cause and re-measures fence attributes the gap without a capture. The injected indirect-draw validation (the common case) is isolated by a **direct-vs-indirect** swap (issue the same draws as `drawIndexed` instead of `drawIndexedIndirect`) or a **buffer-count** swap (consolidate vs split args); the point-shadow regression was attributed this way 2026-06-14 (the ~3 ms collapsed to ~0.2 ms under direct draws — "WebGPU-specific traps"). A native capture (RenderDoc / PIX / Nsight on the GPU process, or webgpu_inspector) is for *reading the absolute breakdown* once the ablation has named the cause — not for the localization itself.

## Pipeline labels

Every `createComputePipelineAsync` and `createRenderPipelineAsync` call must include a `label` property. Labels appear in the stats overlay startup section and bench output. Use the entry point name or a short descriptive name (`"narrowphase"`, `"forward"`, `"bvh-tree"`).

## WebGPU-specific traps

- **Chrome's D3D12 backend injects indirect-draw validation, and it has two distinct costs — only one is fixed by consolidating.** (1) *Per indirect buffer:* splitting args across many small `GPUBuffer`s vs one consolidated buffer was measured 300× slower ([Toji's bundle-culling benchmark](https://toji.dev/webgpu-best-practices/indirect-draws.html)). Consolidate — this collapses the dispatch *count* across buffers. (2) *Per indirect draw command, ~1 µs, which consolidation does NOT remove.* Measured 2026-06-14 (lovelace, `bun bench --scenario render --param mode=gltf-model`): the point-shadow pass issues ~124 distinct meshes × 6 faces × 4 casters ≈ 2976 `drawIndexedIndirect`/frame against the **one** consolidated `Parts.drawArgs` buffer, and the validation runs *before* the render pass (untimed by `timestampWrites`), surfacing as **~3 ms of fence wait** while the timed pass is 0.46 ms. Swapping the identical commands to non-indirect `drawIndexed` collapses the untimed cost to ~0.2 ms — proving the cost is the *indirect* path, ~1 µs × the command count, not bandwidth/raster. So the floor is `#drawIndexedIndirect × ~1 µs`, untimed: **reduce the command count** (instancing-collapse the per-(view × mesh) fan-out, fewer passes), don't just consolidate buffers. Sear's point-shadow atlas does exactly this (render.md "Point-light shadows") — the per-(caster, face) fan-out collapsed by a per-combo cull + re-gather to one indirect draw per casting mesh (2976 → ~124), fence wait 3.89 → 0.55 ms on lovelace. This is the cost the absent `multi-draw-indirect` ([gpuweb#1354](https://github.com/gpuweb/gpuweb/issues/1354)) would erase. Localize it with a direct-vs-indirect ablation on fence wait (below), not a native capture.
- **Render bundles cut CPU-side encode, NOT the GPU-side indirect-draw validation above.** Static draw lists (UI, fixed scene-graph segments) belong in `GPURenderBundle` ([Toji on render bundles](https://toji.dev/webgpu-best-practices/render-bundles.html)) — the record-time validation + encoding isn't repeated *on the CPU* at execute. But the GPU-injected indirect validation runs the same from a bundle replay or inline: measured 2026-06-14, the (then-bundled) sun shadow pass's fence wait was **identical** for `executeBundles` (3.89 ms) vs inline indirect (4.42 ms). Bundles are a **CPU** win, not a fix for the per-command floor — so sear ships none: **both** shadow atlases (the sun cascades + the point atlas) take the *command-count collapse* instead (the re-gather to one `drawIndexedIndirect` per casting mesh, render.md "Point-light shadows" / "Sun shadows"), the floor fix the bundle CPU-win can't reach.
- **Single queue, no async compute.** Cannot overlap compute and graphics; the compute graph must serialize what Vulkan would parallelize. Don't design as if they're independent.
- **`createBindGroup` is per-draw-frequency cheap, but cache it when bindings are stable.** Recreating a bind group per frame for static resources is pure waste.
- **Texture compression.** WebGPU exposes BC (`texture-compression-bc`), ETC2 (`texture-compression-etc2`), ASTC (`texture-compression-astc`) as features. None is on the base floor — `GltfPlugin` declares all three as `preferredFeatures`, which is what keeps the `device.features` read below true (an unrequested feature reads false). That holds only where the plugin is in the app's list: `loadGltf` standalone requests nothing, so every family reads false. So the read is total (`pickTargets` returns no targets rather than throwing) and the gate sits per-image in `gltf/assets.ts`, firing only on a KTX2 image with nowhere to transcode to. BC7 for desktop, ASTC for mobile. Check `device.features` and prefer compressed at load time — compression is the largest single bandwidth lever for textured scenes, dwarfing most shader-level optimizations. A `texture_2d_array` (the firehose multi-texture binding — WebGPU has no bindless) forces every layer to one size + format, so varied-size sources can't all stay compressed in one array. Don't resolve that by decoding the whole set to uncompressed: one outlier then throws the lever away for every texture — a silent 4× bandwidth + VRAM regression. Bucket by (size, format) into one compressed array per bucket, selecting the bucket per-instance in the FS (`textureSampleGrad` on a per-draw-uniform index — the no-bindless array-select; `extras/gltf` is the worked case). The trap generalizes — a fallback that drops a batch optimization for all N on a single mismatch is the anti-pattern, in any array-packing context.
- **Metal miscompiles large dynamically-indexed function-private arrays under multi-lane SIMD.** When a kernel's per-invocation `function`-space footprint grows enough to spill registers, Apple/Metal computes the wrong per-lane offset for the spilled, dynamically-indexed data once >1 SIMD lane is active (wg ≥ 64, or wg32 multi-lane); single-lane runs and nvidia/D3D12 stay correct, so it hides from one-pair probes. It's a *threshold*: shrinking the footprint relocates the corruption, doesn't remove it. Keep function-private working sets small; if an accumulator must be large, stream to a global buffer (immune) or unroll to scalars (no dynamic-indexed private array). The AVBD narrowphase hit this: `collide.ts`'s `Poly` `MAX_POLY_VERTS` 16→8 (its exact clip bound) dropped `collideBoxBox` below the threshold. Dead dodges: `ptr<function>` out-param and `var<private>` (Tint lowers both to the same pointer-threaded private storage), vec3→vec4 (already 16B stride, layout #4). Gate such kernels on a real-GPU Metal run (`bun bench`), never just nvidia.
