import { Compute, type State } from "@dylanebert/shallot";
import type { MeshBinding } from "@dylanebert/shallot/render/core";
import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { documentDirtyTiles, type StrokeDocument } from "./document";
import { allocate, drain } from "./queue";
import { rasterizeTile, warm as rasterizeWarm } from "./rasterize";
import {
    ALBEDO_FORMAT,
    ATLAS_LAYERS,
    DIST_FORMAT,
    THROTTLE,
    TILE_COUNT,
    TILE_RES,
    tileCoordOf,
} from "./tiles";

// The overlay atlas's GPU state: two texture-2d-arrays (albedo + boundary distance, `tiles.ts`'s formats)
// sized to {@link ATLAS_LAYERS} — smaller than {@link TILE_COUNT}, the "small indirection" the spec's
// Approach names — plus the indirection storage buffer the terrain fs (`terrain/terrain.ts`) looks tile id
// up in. A tile's layer is allocated the first time it's marked dirty (never evicted — out of scope, see
// tiles.ts); `redraw` drains the dirty queue at a fixed per-frame throttle, so a burst of edits (a hand-
// authored stroke, or a future full network regeneration) never stalls one frame with every tile's
// rasterize dispatch at once. Stage 5 moved the per-tile content from a CPU `TilePacker` (`writeTexture`
// on a JS-computed `Uint8Array`) to `rasterize.ts`'s GPU compute dispatch (`copyBufferToTexture` off a
// packed storage buffer) — `redraw` below is the seam that changed; `queue.ts`'s `drain`/`allocate` are
// untouched.

/** the indirection buffer's declared element schema: `array<i32, TILE_COUNT>` — negative = unallocated
 *  (`terrain.ts`'s fs reads `< 0` as "no overlay here"). Exported so the surface layout and this module's
 *  own typed buffer agree on one schema, never two independently-constructed `d.arrayOf` calls. */
export const Indirection = d.arrayOf(d.i32, TILE_COUNT);

let albedoTex: GPUTexture | null = null;
let distTex: GPUTexture | null = null;
let sampler: GPUSampler | null = null;
let indirectionRaw: GPUBuffer | null = null;
let indirectionTyped: (TgpuBuffer<typeof Indirection> & StorageFlag) | null = null;

// the indirection table's CPU mirror (so allocateLayer can read "already resident?" without a GPU
// readback) + the free-running layer counter. Reset at warm.
const indirectionCpu = new Int32Array(TILE_COUNT).fill(-1);
let nextLayer = 0;

// the dirty queue: an array (FIFO order — earliest marks redraw first) + a parallel Set for O(1)
// de-duplication (`markDirty` called twice for the same tile before it drains must not double-queue it).
const pending: number[] = [];
const pendingSet = new Set<number>();

function teardown(): void {
    albedoTex?.destroy();
    distTex?.destroy();
    indirectionRaw?.destroy();
    albedoTex = null;
    distTex = null;
    sampler = null;
    indirectionRaw = null;
    indirectionTyped = null;
    indirectionCpu.fill(-1);
    nextLayer = 0;
    pending.length = 0;
    pendingSet.clear();
}

/** allocate the atlas's GPU resources — textures, sampler, indirection buffer, all cleared to "no tile
 *  resident". Ties cleanup to `state.onDispose` (the same in-place-rebuild-safe pattern `terrain.ts` uses). */
export function warm(state: State): void {
    teardown();
    state.onDispose(teardown);
    rasterizeWarm(state); // its own teardown/onDispose registration, the same pattern
    const { device, root } = Compute;

    albedoTex = device.createTexture({
        label: "overlay-albedo",
        size: { width: TILE_RES, height: TILE_RES, depthOrArrayLayers: ATLAS_LAYERS },
        format: ALBEDO_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    distTex = device.createTexture({
        label: "overlay-dist",
        size: { width: TILE_RES, height: TILE_RES, depthOrArrayLayers: ATLAS_LAYERS },
        format: DIST_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    // clamp-to-edge: an indirection miss reads coverage 0 (see terrain.ts), but a resident tile's own edge
    // texels must never sample a neighbour layer's content — array layers don't share address space, so
    // this only guards the tile's own uv range degenerating past [0,1] at the antialiased boundary.
    sampler = device.createSampler({
        label: "overlay-samp",
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
    });
    indirectionRaw = device.createBuffer({
        label: "overlay-indirection",
        size: TILE_COUNT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    indirectionCpu.fill(-1);
    device.queue.writeBuffer(indirectionRaw, 0, indirectionCpu as Int32Array<ArrayBuffer>);
    indirectionTyped = root.createBuffer(Indirection, indirectionRaw).$usage("storage");
    nextLayer = 0;
    pending.length = 0;
    pendingSet.clear();
}

/** the terrain mesh's `bindings` override for the four overlay entries the surface layout declares
 *  (`terrain.ts`). Reads live GPU resource identities — the renderer resolves a texture binding to a view
 *  each frame, so no bind group here needs manual (re)construction. */
export function bindings(): Record<string, MeshBinding> {
    if (!albedoTex || !distTex || !sampler || !indirectionTyped) {
        throw new Error("overlay atlas: bindings() read before warm()");
    }
    return {
        albedo: albedoTex,
        dist: distTex,
        overlaySamp: sampler,
        indirection: indirectionTyped,
    };
}

/** mark every tile `doc` touches dirty (`document.ts`'s exact per-primitive union, not one bounding rect
 *  over the whole document) — idempotent: a tile already pending isn't re-queued. */
export function markDirty(doc: StrokeDocument): void {
    for (const id of documentDirtyTiles(doc)) {
        if (pendingSet.has(id)) continue;
        pendingSet.add(id);
        pending.push(id);
    }
}

/**
 * drain up to {@link THROTTLE} pending dirty tiles: allocate each one's atlas layer, rasterize `doc`'s
 * content into it via `rasterize.ts`'s GPU dispatch, and flush the changed indirection entries. Returns
 * the number of tiles redrawn — 0 once the queue is empty, so a caller (the per-frame `OverlaySystem`,
 * `terrain.ts`) can poll it every frame with no extra bookkeeping.
 */
export function redraw(doc: StrokeDocument): number {
    if (!albedoTex || !distTex || !indirectionRaw) return 0;
    const ids = drain(pending, pendingSet, THROTTLE);
    if (ids.length === 0) return 0;
    const { device } = Compute;
    for (const id of ids) {
        const alloc = allocate(indirectionCpu, id, nextLayer, ATLAS_LAYERS);
        nextLayer = alloc.nextLayer;
        const layer = alloc.layer;
        const [tx, tz] = tileCoordOf(id);
        rasterizeTile(doc, tx, tz, albedoTex, distTex, layer);
        device.queue.writeBuffer(indirectionRaw, id * 4, new Int32Array([layer]));
    }
    return ids.length;
}

/** whether every marked tile has drained through {@link redraw} — the boot path polls this so the
 *  hand-authored stroke is fully resident before the device gate's capture reads the frame. */
export function idle(): boolean {
    return pending.length === 0;
}
