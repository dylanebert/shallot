import { Compute, type State } from "@dylanebert/shallot";
import type { MeshBinding } from "@dylanebert/shallot/render/core";
import type { StorageFlag, TgpuBuffer, UniformFlag } from "typegpu";
import * as d from "typegpu/data";
import { documentDirtyTiles, type StrokeDocument } from "./document";
import { chordOf } from "./network";
import { allocate, drain, invalidate as invalidateQueue, release } from "./queue";
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

// Why a tiled overlay atlas, sampled inside the terrain surface, rather than projected decals:
//
// Cost model. A decal is its own geometry (a box or a screen-space quad) rendered over the terrain; its
// per-frame cost is proportional to how many decal volumes overlap a given pixel — cheap for a handful of
// bullet holes or footprints, expensive once decals cover most of the visible ground, since a dense
// overlapping set means every covered pixel re-evaluates every decal that reaches it. This atlas inverts
// that: the terrain's own fragment shader pays one indirection lookup into {@link Indirection} plus one
// atlas sample (`terrain/terrain.ts`'s composite) every frame, *regardless* of how many roads or carparks
// exist — a network of 5 primitives and one of 500 cost the same per-frame sample. The part of the cost
// that scales with primitive count is the rasterization into dirty tiles below, and that's paid once per
// edit, throttled ({@link THROTTLE} tiles/frame), never once per frame per pixel.
//
// Same-surface sampling. A decal is a second piece of geometry positioned just above the terrain, which
// needs a depth bias to keep it from z-fighting the surface underneath it — flickering where the two
// depths are close enough that which one wins a given pixel becomes numerically unstable. That bias is a
// tuned constant with no value that's simultaneously right at every camera distance and every terrain
// slope. Reading this atlas from inside the terrain fs, instead of drawing a second surface, has nothing
// to z-fight: there's exactly one depth value written per pixel, the terrain's own, so the failure mode
// isn't tuned away, it has no surface to occur on.
//
// The overlay atlas's GPU state: two texture-2d-arrays (albedo + boundary distance, `tiles.ts`'s formats)
// sized to {@link ATLAS_LAYERS} — 64 layers (measured worst-case swath 46 + headroom, stage 4d), so
// the indirection table maps 256 tile ids into 64 atlas layers with compaction — plus the indirection
// storage buffer the terrain fs (`terrain/terrain.ts`) looks tile id up in. The indirection is retained
// (the fs still reads it as "which layer holds this tile?"), and it packs a larger tile space into a
// smaller atlas: a tile's layer is allocated the first time it's marked dirty and never evicted within
// one document's lifetime (per-tile eviction/paging is out of scope, see tiles.ts) — but a document *swap*
// (`terrain.ts`'s `regenerate`, the F9 reseed control) is a coarser event: {@link invalidate} releases
// every resident layer at once, so the incoming document allocates into a fresh atlas rather than
// accreting on top of the outgoing one. `redraw` drains the dirty queue at a fixed per-frame throttle, so
// a burst of edits (a hand-authored stroke, or a full network regeneration) never stalls one frame with
// every tile's rasterize dispatch at once. Stage 5 moved the per-tile content from a CPU `TilePacker` (`writeTexture`
// on a JS-computed `Uint8Array`) to `rasterize.ts`'s GPU compute dispatch (`copyBufferToTexture` off a
// packed storage buffer) — `redraw` below is the seam that changed; `queue.ts`'s `drain`/`allocate` are
// untouched.

/** the indirection buffer's declared element schema: `array<i32, TILE_COUNT>` — negative = unallocated
 *  (`terrain.ts`'s fs reads `< 0` as "no overlay here"). Exported so the surface layout and this module's
 *  own typed buffer agree on one schema, never two independently-constructed `d.arrayOf` calls. */
export const Indirection = d.arrayOf(d.i32, TILE_COUNT);

/** the chord uniform's schema: two endpoints (vec2f, world x/z) + halfWidth — the analytic fs's marking
 *  geometry input (stage 8). Exported so `terrain.ts`'s surface layout and this module's typed buffer
 *  agree on one struct, never two independently-constructed `d.struct` calls. */
export const ChordUniform = d.struct({
    a: d.vec2f,
    b: d.vec2f,
    halfWidth: d.f32,
});

let albedoTex: GPUTexture | null = null;
let distTex: GPUTexture | null = null;
let sampler: GPUSampler | null = null;
let indirectionRaw: GPUBuffer | null = null;
let indirectionTyped: (TgpuBuffer<typeof Indirection> & StorageFlag) | null = null;
let chordRaw: GPUBuffer | null = null;
let chordTyped: (TgpuBuffer<typeof ChordUniform> & UniformFlag) | null = null;

// the indirection table's CPU mirror (so allocate can read "already resident?" without a GPU
// readback) + the free list of available atlas layers. Reset to full capacity at warm.
const indirectionCpu = new Int32Array(TILE_COUNT).fill(-1);
const free: number[] = [];

// the dirty queue: an array (FIFO order — earliest marks redraw first) + a parallel Set for O(1)
// de-duplication (`markDirty` called twice for the same tile before it drains must not double-queue it).
const pending: number[] = [];
const pendingSet = new Set<number>();

function teardown(): void {
    albedoTex?.destroy();
    distTex?.destroy();
    indirectionRaw?.destroy();
    chordRaw?.destroy();
    albedoTex = null;
    distTex = null;
    sampler = null;
    indirectionRaw = null;
    indirectionTyped = null;
    chordRaw = null;
    chordTyped = null;
    invalidateQueue(indirectionCpu, free, ATLAS_LAYERS, pending, pendingSet);
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
    invalidateQueue(indirectionCpu, free, ATLAS_LAYERS, pending, pendingSet);
    device.queue.writeBuffer(indirectionRaw, 0, indirectionCpu as Int32Array<ArrayBuffer>);
    indirectionTyped = root.createBuffer(Indirection, indirectionRaw).$usage("storage");

    // the chord uniform — the analytic fs's marking geometry input (stage 8). Written once at warm
    // and re-written on every document change via updateChord.
    chordRaw = device.createBuffer({
        label: "overlay-chord",
        size: d.sizeOf(ChordUniform),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    chordTyped = root.createBuffer(ChordUniform, chordRaw).$usage("uniform");
}

/**
 * invalidate every resident tile on a document swap (`terrain.ts`'s `regenerate`, the F9 reseed control):
 * releases every resident layer back to the free list, clears the indirection CPU mirror, and drops any
 * redraw still queued from the outgoing document — then flushes the reset indirection straight to the GPU
 * buffer the terrain fs reads (`terrain.ts`'s composite treats `layer < 0` as "no overlay here"), so a
 * tile the old document touched but the new one doesn't reads terrain, not stale road, the very next frame
 * rather than whenever its old layer happens to get reused. The atlas *textures* keep their stale texels —
 * inert until a fresh `redraw` overwrites them, since nothing samples an unindexed layer — so this only
 * needs to touch indirection, not `writeTexture` every layer. Call before {@link markDirty} marks the
 * swapped-in document's own tiles, or they'd pack onto the tail of a still-full atlas instead of a freshly
 * emptied one — the ordering `queue.ts`'s `invalidate` doc comment names.
 */
export function invalidate(): void {
    if (!indirectionRaw) return; // warm() hasn't run yet — nothing resident to release
    invalidateQueue(indirectionCpu, free, ATLAS_LAYERS, pending, pendingSet);
    Compute.device.queue.writeBuffer(indirectionRaw, 0, indirectionCpu as Int32Array<ArrayBuffer>);
}

/** the terrain mesh's `bindings` override for the four overlay entries the surface layout declares
 *  (`terrain.ts`). Reads live GPU resource identities — the renderer resolves a texture binding to a view
 *  each frame, so no bind group here needs manual (re)construction. */
export function bindings(): Record<string, MeshBinding> {
    if (!albedoTex || !distTex || !sampler || !indirectionTyped || !chordTyped) {
        throw new Error("overlay atlas: bindings() read before warm()");
    }
    return {
        albedo: albedoTex,
        dist: distTex,
        overlaySamp: sampler,
        indirection: indirectionTyped,
        chord: chordTyped,
    };
}

/** write the chord uniform from `doc`'s one road — call wherever the live document changes
 *  (`terrain.ts`'s `warm`, `regenerate`, `editDocument`), or the fs renders markings for the old chord
 *  and the drag's markings lag behind the handle. */
export function updateChord(doc: StrokeDocument): void {
    if (!chordTyped || !chordRaw) return;
    const { a, b, halfWidth } = chordOf(doc);
    chordTyped.write({
        a: d.vec2f(a[0], a[1]),
        b: d.vec2f(b[0], b[1]),
        halfWidth,
    });
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
 * re-tile the atlas for a document edit: mark `tiles(old) ∪ tiles(new)` dirty (so every tile whose content
 * changed or appeared is redrawn) and release `tiles(old) − tiles(new)` (so tiles the old document touched
 * but the new one doesn't have their indirection cleared to −1 and their layer pushed back to the free
 * list). The mark-dirty runs before the release — `release` removes a released id from the pending queue,
 * so a tile in the difference is first marked dirty then unmarked, never redrawn. After `retile` the
 * pending queue holds exactly `tiles(new)` (the union minus the difference), and the resident set is the
 * previous resident minus the difference; once {@link redraw} drains, the resident set equals
 * `tiles(new)` and the free list has absorbed every layer the difference freed. Flushes the changed
 * indirection entries to the GPU buffer so a released tile reads terrain, not stale road, the very next
 * frame. This is the edit path; {@link invalidate} is the reseed path — both share `release`.
 */
export function retile(oldDoc: StrokeDocument, newDoc: StrokeDocument): void {
    if (!indirectionRaw) return; // warm() hasn't run yet — nothing to retile
    const oldTiles = new Set(documentDirtyTiles(oldDoc));
    const newTiles = new Set(documentDirtyTiles(newDoc));
    markDirty(oldDoc);
    markDirty(newDoc);
    const toRelease: number[] = [];
    for (const id of oldTiles) {
        if (!newTiles.has(id)) toRelease.push(id);
    }
    release(indirectionCpu, toRelease, free, pending, pendingSet);
    Compute.device.queue.writeBuffer(indirectionRaw, 0, indirectionCpu as Int32Array<ArrayBuffer>);
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
        const layer = allocate(indirectionCpu, id, free, ATLAS_LAYERS);
        const [tx, tz] = tileCoordOf(id);
        rasterizeTile(doc, tx, tz, albedoTex, distTex, layer);
        device.queue.writeBuffer(indirectionRaw, id * 4, new Int32Array([layer]));
    }
    return ids.length;
}

/** whether every marked tile has drained through {@link redraw} — the boot path polls this so the
 *  procedural network is fully resident before the device gate's capture reads the frame. */
export function idle(): boolean {
    return pending.length === 0;
}
