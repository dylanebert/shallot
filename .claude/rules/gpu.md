---
paths:
    - "packages/shallot/src/gpu/**/*.ts"
    - "packages/shallot/src/standard/compute/**/*.ts"
    - "packages/shallot/src/standard/viewport/**/*.ts"
    - "packages/shallot/src/standard/physics/**/*.ts"
    - "packages/shallot/src/standard/render/**/*.ts"
    - "packages/shallot/src/standard/raster/**/*.ts"
    - "packages/shallot/src/extras/**/*.ts"
---

# GPU

Reference: `docs/standard/compute.md` for compute graph, dynamic capacity, and performance patterns. `docs/standard/render.md` for rendering pipeline and mesh format.

## Types and constants

Engine code uses raw WebGPU types directly (`GPUDevice`, `GPUBuffer`, `GPUBufferUsage`, etc.). Type declarations from `@webgpu/types` in tsconfig.

- **Device tracking:** `trackDevice()` in `compute/core` subpath — patches `createBuffer`/`createTexture` for memory tracking, `queue.submit` for pending submit counting, and async pipeline creation for compile timing. Applied by `StatsPlugin` in its `initialize`, not by compute. Call `registry.finalizeCompile()` before reading `compileTimings` (derives per-pipeline deltas from raw end timestamps)

## Binding limits

WebGPU `maxStorageBuffersPerShaderStage`: 8 (spec min), **10 is our hard ceiling**. 99.6% of devices support 10. Only 64% support 16 (31% on Mac). Requesting above 10 rejects `requestDevice()` on a third of users. Plan for 10, not higher.

**Per shader stage across all bind groups** — splitting into groups doesn't help reduce the count.

**Count storage buffers per stage before adding any binding.** Both `storage` and `read-only-storage` count toward the same limit.

Current counts (audit when touching): raster VS 8-9 (3 groups, instanceData conditional), raster FS 4-5 (instanceData conditional), physics solver 11 (10 storage + 1 uniform, solverBindGroupLayout — at limit), broadphase/narrowphase 10 (8 storage + 1 uniform + 2 pair group, bodyBindGroupLayout + pairBindGroupLayout — at limit), character sweep 8 (3 storage + 4 read-only-storage + 1 uniform, own layout), rebuild pass 4, pack 6+1u, compact 2+1u, BVH instance 9 (at limit), TLAS <6, RT closest-hit 10 (3 groups, at limit), RT any-hit 10 (3 groups, same layout as closest-hit), RT shade 9-10 (2 groups, instanceData conditional).

### Consolidation strategies

Never add per-entity CPU iteration to save a binding.

1. **Interleave GPU-generated buffers written in the same shader.** If the same compute pass already writes both buffers per entity, merge them into one struct. Zero overhead. Example: `entityBlasMeta` + `instanceInverses` (both written by instance.ts per-entity workgroup).

2. **Fold scalars into an existing buffer's header.** Counter u32s, entity counts — store at offset 0 of a related buffer. Example: `entityCount` → `shapeData[0]`.

3. **Block-concatenate CPU uploads.** Two `writeBuffer` calls to different offsets of one GPU buffer. Shader accesses both regions via offset arithmetic from one binding. No iteration — each write is a bulk memcpy. Tradeoff: shader loses direct typed access (e.g., `mat4x4` becomes 4 `vec4` reads + manual construction). Only worth it when the pipeline is near the limit. Example: matrices at offset 0, sizes at offset `capacity * 64`.

4. **Split the pass.** When a shader genuinely needs >10 buffers, break it into two compute passes with intermediate results. Last resort — adds latency.

**Don't consolidate:** GPU buffers generated in different passes (sortedIds from radix sort + leafAABBs from AABB compute), mixed CPU/GPU buffers (entityIds from batch shader + shapes from CPU upload), or buffers whose ownership is shared with utilities that require exclusive access (radix sort ping-pong).

### When you hit the limit

Never silently exceed 10 — Chrome fails with zero diagnostics. Don't add a separate bind group thinking it solves the problem (it doesn't — same per-stage limit). Consolidate first, split passes second.

## DXC shader compilation

DXC (Chrome on Windows) is the compilation bottleneck:

- **Never call large functions inside dynamic loops.** BVH traversal in a dynamic loop = 10x+ slowdown
- **Dead code isn't free.** DXC doesn't DCE. Duplicated traversal doubles compile cost
- **Dynamic loop bounds hurt.** Constant upper bounds with dynamic `break` are fine
- **Chrome shader cache handles repeat visits.** Optimize via pipeline splits (separate shader per code path)

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
- **General:** `readBuffer`, `readFloat32`, `readUint32` in `standard/compute/readback.ts`
- **Profiling:** timestamp queries via `standard/compute/profile.ts`

## GPU profiling

Every compute/render pass must include `timestampWrites`. Compute passes use `beginComputePass(encoder, ctx.timestampWrites?.("pass-name"))`. Same-named entries accumulate.

Timestamp queries require `"timestamp-query"` feature. Viewport and physics each create their own `ProfileState` — `null` when unavailable. Profile lives in the `compute/core` subpath as opt-in tooling.

## Pipeline labels

Every `createComputePipelineAsync` and `createRenderPipelineAsync` call must include a `label` property. Labels appear in the stats overlay startup section and bench output. Use the entry point name or a short descriptive name (`"narrowphase"`, `"forward"`, `"bvh-tree"`).
