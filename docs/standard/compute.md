---
title: Compute
description: running work on the GPU
source: standard/compute
icon: cpu
---

# Compute

<!-- tabs -->
<!-- tab: UI -->

coming soon

<!-- tab: Code -->

Shallot's GPU layer is a dependency-driven compute graph — order emerges from resource dependencies, not manual pass ordering. If two nodes need ordering but share no resource dependency, create a sequencing resource.

### Compute graph

Nodes declare `inputs` and `outputs` as string arrays naming shared resources. The graph topologically sorts nodes so producers run before consumers. Plans are cached and invalidated on node add/remove.

A node with the same name in `inputs` and `outputs` is a **transformer** — it redirects that resource for downstream consumers. Forward writes `color`; an outline transformer reads `color` + `eid`, writes a new `color` intermediate, and calls `ctx.setTextureView("color", view)` so postprocess sees the transformed version. Multiple fresh writers of one name are allowed; consumers wait for all of them.

Two scopes control when nodes execute:

- **`"frame"` scope** — runs once per frame, before any view. The scene uniform has previous-frame camera data at this point, so frame-scoped shaders must not depend on camera position/orientation
- **`"view"` scope** (default) — runs once per view, after scene uniform upload

Sub-graphs (`graph.subGraph("raster")`) hold pipeline-specific nodes. Only the active sub-graph's nodes are included in compilation. Shared nodes (added to the graph directly) run for all sub-graphs.

Use `sync: true` when the CPU needs GPU results before continuing. Use `ctx.afterSubmit(fn)` to defer work like `mapAsync` to after `queue.submit`.

### Dynamic capacity

Entity capacity starts at 1024 and grows by doubling. Growth happens in `addEntity()` between frames, never mid-render.

- **`gbuf()`** creates GPU buffers sized to capacity — lazy-recreates on `.buffer` access when capacity changes
- **`binding()`** creates lazy bind groups that invalidate on capacity change and expose manual `invalidate()` for external triggers
- **`view()`** creates sub-regions of a gbuf

Never bake capacity into WGSL as a compile-time constant — use uniforms. Fixed generous bounds are acceptable for tree depth (32 covers 4B entities).

### Profiling

GPU timestamp profiling and memory tracking live in a separate subpath (`@dylanebert/shallot/compute/core`), not in the main compute barrel. Viewport and physics each own their own `ProfileState`. `StatsPlugin` applies device tracking in its `initialize`.

## Examples

### Compute nodes

Register nodes on the compute graph with resource dependencies. The graph resolves execution order automatically.

### GPU readback with back-pressure

Readback sites must use a pending flag to skip both copy and `mapAsync` when a previous readback is in flight:

```typescript
if (!readbackPending) {
    readbackPending = true;
    ctx.encoder.copyBufferToBuffer(src, 0, staging, 0, size);
    ctx.afterSubmit(() => {
        staging.mapAsync(MapMode.READ).then(
            () => { /* read, unmap */ readbackPending = false; },
            () => { readbackPending = false; },
        );
    });
}
```

Always include a rejection handler on `mapAsync` to clear the flag on device loss.

### Performance patterns

- **Single indirect buffer** — separate buffers cause per-draw validation overhead (measured 300x worse)
- **`writeBuffer` over mapped staging** — one large write beats many small ones (40% JS savings)
- **Upload only what changed** — `writeBuffer` supports sub-range writes
- **Dispatch actual work** — use entity count, not `capacity()`. When GPU-resident, use `dispatchWorkgroupsIndirect`


<!-- tab: Reference -->

<!-- API:standard/compute -->

<!-- CORE:compute -->

<!-- /tabs -->
