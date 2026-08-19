// The terrain mesh + surface: registers the fixed W×H grid (grid.ts) as a `Meshes`/`Draws` entry whose
// vertex content the height kernel (generate.ts) fills, and a custom sear surface that shades the result
// from slope + height alone, then composites one overlay sample over that base by coverage (the road
// network's substrate, `../overlay/` — same-surface sampling, no second draw, no depth bias: the spec's
// Locked decision).
//
// Unlike voxel's mesher, the index list here never changes (grid.ts's `gridIndices` is a pure function
// of the fixed cell count) — so it's authored once on the CPU and uploaded once, and the mesh is
// registered once at warm, not re-emitted every frame. `warm`/`dispose` still exist (not a one-shot
// `initialize`) because GPU buffers need a live device, which only exists once RenderPlugin has warmed.

import { Compute, type Plugin, RenderPlugin, type State, type System } from "@dylanebert/shallot";
import { DrawIndexedIndirect, Draws, type Mesh, Meshes } from "@dylanebert/shallot/render/core";
import { fsCtxSchema, lit, registerSurface, surfaceLayout } from "@dylanebert/shallot/sear/core";
import { MeshQuant } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { envFlag } from "../env";
import * as overlayAtlas from "../overlay/atlas";
import type { StrokeDocument } from "../overlay/document";
import { generateNetwork } from "../overlay/network";
import { COVERAGE_BAND_PX, DIST_RANGE, TILE_SIZE, TILES_PER_SIDE } from "../overlay/tiles";
import {
    DEFAULT_FALLOFF_SCALE,
    MAX_FALLOFF_SCALE,
    MIN_FALLOFF_SCALE,
    setNetwork,
    warmNetwork,
} from "./flatten";
import { bindTerrainKernel, generate, TERRAIN_QUANT } from "./generate";
import {
    GridIndices,
    GridPosition,
    GridVertices,
    gridIndices,
    INDEX_COUNT,
    VERTEX_COUNT,
    WORLD_HALF,
} from "./grid";
import { DEFAULT_SMOOTH_RADIUS, MAX_SMOOTH_RADIUS, MIN_SMOOTH_RADIUS } from "./profile";

export { generate } from "./generate";

export const SEED = 1337;

// slope/height blend thresholds — grass on flat low ground, rock where it steepens, snow above the
// relief's upper band. `slope` is `dot(worldNormal, up)`: 1 = flat, 0 = vertical. Fractions of RELIEF
// (generate.ts / noise.ts), not absolute metres, so the bands scale automatically if RELIEF changes.
const ROCK_SLOPE_LO = 0.7;
const ROCK_SLOPE_HI = 0.92;
const SNOW_HEIGHT_LO = 0.55; // fraction of the [GROUND_LEVEL, GROUND_LEVEL + RELIEF] band
const SNOW_HEIGHT_HI = 0.8;

const terrainSurfaceLayout = surfaceLayout({
    albedo: { type: "texture-2d-array" },
    dist: { type: "texture-2d-array" },
    overlaySamp: { type: "sampler" },
    indirection: { type: "storage", element: d.i32 },
});

const terrainFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const grass = d.vec3f(0.09, 0.16, 0.05);
    const rock = d.vec3f(0.12, 0.11, 0.1);
    const snow = d.vec3f(0.55, 0.58, 0.62);

    const slope = std.dot(ctx.worldNormal, d.vec3f(0, 1, 0));
    const rockBlend = 1 - std.smoothstep(d.f32(ROCK_SLOPE_LO), d.f32(ROCK_SLOPE_HI), slope);
    const base = std.mix(grass, rock, rockBlend);

    const heightT = std.clamp(ctx.world.y / d.f32(TERRAIN_QUANT.posScale.y * 0.5), 0, 1);
    const snowBlend = std.smoothstep(d.f32(SNOW_HEIGHT_LO), d.f32(SNOW_HEIGHT_HI), heightT);
    let color = std.mix(base, snow, snowBlend);

    // the overlay composite: world (x, z) → tile column → indirection lookup → one atlas sample, coverage
    // thresholded from the stored signed distance with fwidth (Green, SIGGRAPH 2007 — the sub-texel-crisp
    // edge the spec's Locked decision names). Same-surface: no second draw, no depth bias.
    const tx = std.clamp(
        d.i32(std.floor((ctx.world.x + d.f32(WORLD_HALF)) / d.f32(TILE_SIZE))),
        0,
        d.i32(TILES_PER_SIDE - 1),
    );
    const tz = std.clamp(
        d.i32(std.floor((ctx.world.z + d.f32(WORLD_HALF)) / d.f32(TILE_SIZE))),
        0,
        d.i32(TILES_PER_SIDE - 1),
    );
    const tileIdx = tz * d.i32(TILES_PER_SIDE) + tx;
    const layer = terrainSurfaceLayout.$.indirection[d.u32(tileIdx)];
    // `textureSample` computes screen-space derivatives (mip/anisotropy), so WGSL requires it run in
    // *uniform* control flow — a per-fragment `if (layer >= 0)` branch (layer varies with world position,
    // non-uniform across the wavefront) makes the compiler reject the call outright ("must only be called
    // from uniform control flow", caught by the device gate's real-GPU pipeline compile). The fix is the
    // standard one: always sample, at a clamped-valid layer index, then zero the *contribution* via
    // `select` instead of skipping the sample.
    const layerU = d.u32(std.max(layer, 0));
    const localX = ctx.world.x + d.f32(WORLD_HALF) - d.f32(tx) * d.f32(TILE_SIZE);
    const localZ = ctx.world.z + d.f32(WORLD_HALF) - d.f32(tz) * d.f32(TILE_SIZE);
    const uv = d.vec2f(localX / d.f32(TILE_SIZE), localZ / d.f32(TILE_SIZE));
    const distSample = std.textureSample(
        terrainSurfaceLayout.$.dist,
        terrainSurfaceLayout.$.overlaySamp,
        uv,
        layerU,
    ).x;
    const overlay = std.textureSample(
        terrainSurfaceLayout.$.albedo,
        terrainSurfaceLayout.$.overlaySamp,
        uv,
        layerU,
    ).xyz;
    // decode the r8unorm distance channel back to a signed world-metre distance — the fs's half of the
    // codec overlay/tiles.ts's encodeDist writes (kept in sync by hand: DIST_RANGE is the one shared
    // constant both sides read; TGSL can't call a plain JS helper, so this stays inline).
    const dist = (distSample - d.f32(0.5)) * d.f32(2) * d.f32(DIST_RANGE);
    const fw = std.fwidth(dist) * d.f32(COVERAGE_BAND_PX) + d.f32(1e-5);
    const resident = std.select(d.f32(0), d.f32(1), layer >= 0);
    const coverage = std.clamp(d.f32(0.5) - dist / fw, 0, 1) * resident;
    color = std.mix(color, overlay, coverage);

    return d.vec4f(lit(color, ctx.worldNormal), 1);
});

/** the emitted terrain fs WGSL — the device-free structural seam the overlay composite tests resolve
 *  (`terrain.test.ts`, the `generate.test.ts`/`noise.test.ts` idiom). */
export function terrainFsWgsl(): string {
    return tgpu.resolve([terrainFs], { names: "strict" });
}

/** the terrain mesh's local-space bounding sphere: the grid's XZ half-extent + the relief's half-height,
 *  centred at the mesh centroid — an analytic AABB→sphere, not a scan (the grid isn't uploaded to the
 *  CPU to compute one). */
function terrainBounds(): [number, number, number, number] {
    const cx = TERRAIN_QUANT.posOffset.x + TERRAIN_QUANT.posScale.x * 0.5;
    const cy = TERRAIN_QUANT.posOffset.y + TERRAIN_QUANT.posScale.y * 0.5;
    const cz = TERRAIN_QUANT.posOffset.z + TERRAIN_QUANT.posScale.z * 0.5;
    const hx = TERRAIN_QUANT.posScale.x * 0.5;
    const hy = TERRAIN_QUANT.posScale.y * 0.5;
    const hz = TERRAIN_QUANT.posScale.z * 0.5;
    const radius = Math.sqrt(hx * hx + hy * hy + hz * hz);
    return [cx, cy, cz, radius];
}

let buffers: {
    verticesRaw: GPUBuffer;
    positionRaw: GPUBuffer;
    quantRaw: GPUBuffer;
    indicesRaw: GPUBuffer;
    indirectRaw: GPUBuffer;
} | null = null;

function teardown(): void {
    if (!buffers) return;
    Meshes.delete("terrain");
    Draws.delete("terrain");
    buffers.verticesRaw.destroy();
    buffers.positionRaw.destroy();
    buffers.quantRaw.destroy();
    buffers.indicesRaw.destroy();
    buffers.indirectRaw.destroy();
    buffers = null;
}

// The boot document is the seeded procedural network (`overlay/network.ts`'s `generateNetwork`) at this
// module's own {@link SEED} — the same pinned seed the height kernel boots with, so a fresh load and a
// capture both see one fixed network. `capture.ts`'s `capturePoints` derives its on/off-road probe points
// from this exact `generateNetwork(SEED)` call (`overlay/network.ts`'s `captureProbePoints`), so the two
// can't drift apart. `overlay/stroke.ts`'s hand-authored stroke stays live only as the CPU-vs-GPU
// differential's known pattern (`document.test.ts`), no longer what boots on screen. {@link regenerate}
// swaps to a fresh random seed on demand (the seed control, `boot.ts`'s F9 handler), exercising the exact
// same `markDirty`/`redraw`/flatten path this boot document already takes.
let liveDocument: StrokeDocument = generateNetwork(SEED);
let currentSeed = SEED;
// the straightness instrument's no-cut control: `liveDocument` still drives the overlay and the height-
// axis anchors (the analytic road edge stays a real, checkable position), only the *flatten* kernel gets
// an empty network — `flatten.ts`'s own documented fallback ("empty network... degrades to plain
// heightAt") — so the rendered mesh is genuinely undeformed terrain rather than a cut against zeroed
// RELIEF. `VITE_ROADS_NO_CUT`-gated (`env.ts`); paired with `VITE_ROADS_RELIEF=0` (`noise.ts`) for the
// spec's "zeroed-RELIEF, no-cut control" arm.
const NO_CUT = envFlag("VITE_ROADS_NO_CUT");
const EMPTY_DOCUMENT: StrokeDocument = { polylines: [], polygons: [] };
// the longitudinal smoothing strength (`terrain/profile.ts`'s box-filter radius, samples each side) —
// the spec's taste handover: a live control (`boot.ts`'s bracket-key idiom), not a value baked in and
// declared good. `setSmoothRadius` below is the control's own entry point.
let smoothRadius = DEFAULT_SMOOTH_RADIUS;
// stage 11's live handover, the other half of the same session (`boot.ts`'s bracket-key idiom again):
// the falloff-width multiplier (`terrain/flatten.ts`'s `computeFalloff` `scale` argument). `setFalloffScale`
// below is the control's own entry point.
let falloffScale = DEFAULT_FALLOFF_SCALE;

async function warm(state: State): Promise<void> {
    teardown(); // a rebuild (HMR) re-warms — clear the prior generation's buffers first
    // `Plugin.dispose` only fires on `App.dispose`, never on a direct `state.dispose()` (the path an
    // in-place rebuild takes) — tie cleanup to the State itself so both paths clear these buffers.
    state.onDispose(teardown);
    overlayAtlas.warm(state); // its own teardown/onDispose registration, the same pattern
    const { device, root } = Compute;

    const verticesRaw = device.createBuffer({
        label: "terrain-main",
        // COPY_SRC so the device gate's seed-determinism check (gate.ts) can read the generated stream back.
        size: VERTEX_COUNT * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const positionRaw = device.createBuffer({
        label: "terrain-pos",
        size: VERTEX_COUNT * 8,
        usage: GPUBufferUsage.STORAGE,
    });
    const quantRaw = device.createBuffer({
        label: "terrain-quant",
        size: d.sizeOf(MeshQuant),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const indicesRaw = device.createBuffer({
        label: "terrain-indices",
        size: INDEX_COUNT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    const indirectRaw = device.createBuffer({
        label: "terrain-indirect",
        size: 20,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    buffers = { verticesRaw, positionRaw, quantRaw, indicesRaw, indirectRaw };

    device.queue.writeBuffer(indicesRaw, 0, gridIndices() as Uint32Array<ArrayBuffer>);

    const vertices = root.createBuffer(GridVertices, verticesRaw).$usage("storage");
    const position = root.createBuffer(GridPosition, positionRaw).$usage("storage");
    const quant = root.createBuffer(d.arrayOf(MeshQuant, 1), quantRaw).$usage("storage");
    quant.write([TERRAIN_QUANT]);
    const indices = root.createBuffer(GridIndices, indicesRaw).$usage("storage", "index");
    const indirect = root
        .createBuffer(DrawIndexedIndirect, indirectRaw)
        .$usage("storage", "indirect");
    indirect.write({
        indexCount: INDEX_COUNT,
        instanceCount: 1,
        firstIndex: 0,
        baseVertex: 0,
        firstInstance: 0,
    });

    const mesh: Mesh = {
        name: "terrain",
        vertices,
        position,
        quant,
        indices,
        indexBase: 0,
        indexCount: INDEX_COUNT,
        bounds: terrainBounds(),
        dynamic: true, // content (not topology) is compute-rewritten on reseed — a stale BLAS would cast wrong
        bindings: overlayAtlas.bindings(), // the four overlay entries terrainSurfaceLayout declares
    };
    Meshes.register(mesh);
    Draws.register({ name: "terrain", surface: "terrain", mesh: "terrain", args: { indirect } });

    // the boot document (the procedural network at this module's SEED, above), rasterized by
    // overlay/rasterize.ts's GPU kernel. Marking it dirty here (not per-frame) means one redraw burst at
    // warm, not a repeated mark.
    overlayAtlas.markDirty(liveDocument);

    bindTerrainKernel(vertices, position);
    warmNetwork(state); // its own onDispose registration, the same pattern
    setNetwork(NO_CUT ? EMPTY_DOCUMENT : liveDocument, currentSeed, smoothRadius, falloffScale); // the flatten kernel's
    // geometry input, kept in sync with the overlay's own document and the height kernel's own
    // permutation seed — except under the no-cut control arm, above
    if (state.signal.aborted) return;
    await generate(SEED);
}

/** drain the overlay's per-frame dirty-tile throttle (`overlay/atlas.ts`'s `redraw`), rasterizing
 *  {@link liveDocument}'s content into every drained tile via `overlay/rasterize.ts`'s GPU kernel. Exported
 *  so the device gate can assert draining actually happened. */
const OverlayRedrawSystem: System = {
    name: "roads-overlay-redraw",
    group: "simulation",
    annotations: { mode: "always" },
    update() {
        overlayAtlas.redraw(liveDocument);
    },
};

/**
 * swap the live document to a freshly seeded procedural network (`overlay/network.ts`'s
 * `generateNetwork`) and re-dispatch the height kernel — the seed control's live reseed (`boot.ts`'s F9
 * handler). Marks the new network's tiles dirty (`overlay/atlas.ts`'s `markDirty`, the exact-set oracle
 * `document.test.ts` pins) and rebinds the flatten kernel's geometry (`flatten.ts`'s `setNetwork`) before
 * `generate` re-dispatches, so the next-drawn frame's heights and overlay both reflect the new network in
 * one call — the "affected-region remesh" the spec's Approach names. A tile the previous document touched
 * but the new one doesn't keeps its stale content (`tiles.ts`'s ATLAS_LAYERS never evicts, a stage-4
 * decision this stage inherits, not one it revisits).
 */
export async function regenerate(seed: number): Promise<void> {
    currentSeed = seed;
    liveDocument = generateNetwork(seed);
    setNetwork(liveDocument, seed, smoothRadius, falloffScale);
    overlayAtlas.markDirty(liveDocument);
    await generate(seed);
}

/**
 * the longitudinal smoothing-strength live control (`boot.ts`'s bracket-key handler): re-derives the
 * network's flatten geometry at the new radius and re-dispatches the height kernel, without touching the
 * overlay (2D coverage/albedo is unaffected by a change that only moves target *heights*, so no
 * `markDirty`/redraw is needed here — unlike {@link regenerate}, which also gets a new document). Clamped
 * to `[MIN_SMOOTH_RADIUS, MAX_SMOOTH_RADIUS]` (`terrain/profile.ts`).
 */
export async function setSmoothRadius(radius: number): Promise<void> {
    smoothRadius = Math.min(MAX_SMOOTH_RADIUS, Math.max(MIN_SMOOTH_RADIUS, radius));
    setNetwork(liveDocument, currentSeed, smoothRadius, falloffScale);
    await generate(currentSeed);
}

/** the live smoothing-strength radius — read by the boot's key handler to step relative to the current
 *  value, and by tests. */
export function getSmoothRadius(): number {
    return smoothRadius;
}

/**
 * the falloff-width live control (`boot.ts`'s `-`/`=` handler, stage 11's own half of this session's two-
 * knob handover): re-derives the network's flatten geometry at the new scale and re-dispatches the height
 * kernel — the same shape as {@link setSmoothRadius}, no overlay `markDirty` needed for the same reason
 * (only target heights move). Clamped to `[MIN_FALLOFF_SCALE, MAX_FALLOFF_SCALE]` (`terrain/flatten.ts`) —
 * the lower bound is 1, never narrowing the transition below either of `computeFalloff`'s two derived
 * constraints.
 */
export async function setFalloffScale(scale: number): Promise<void> {
    falloffScale = Math.min(MAX_FALLOFF_SCALE, Math.max(MIN_FALLOFF_SCALE, scale));
    setNetwork(liveDocument, currentSeed, smoothRadius, falloffScale);
    await generate(currentSeed);
}

/** the live falloff-width scale — read by the boot's key handler to step relative to the current value,
 *  and by tests. */
export function getFalloffScale(): number {
    return falloffScale;
}

/**
 * re-bake {@link liveDocument}'s flatten geometry for `seed` without swapping the document or touching the
 * overlay — `gate.ts`'s own seed-determinism probe dispatches `generate` at seeds other than the live
 * one, and the flatten targets are baked CPU-side against a specific seed's permutation (`setNetwork`'s own
 * doc comment): left unsynced, the gate's alternate-seed dispatch would flatten toward a stale seed's
 * terrain under a different seed's natural surface — invisible to today's gate checks (none read the
 * flatten boundary), but wrong regardless. Call before every `generate(seed)` whose seed isn't
 * {@link currentSeed}.
 *
 * Deliberately doesn't touch {@link currentSeed} itself — `gate.ts` always restores to the boot `SEED`
 * at the end regardless of any live F9 reseed (its own pre-existing "restore the boot seed's terrain for
 * the live view" contract), so this only re-syncs the baked geometry, not the module's live-seed
 * bookkeeping. Residue: if `__roadsGate()` is ever invoked *after* a live F9 reseed, `currentSeed` stays
 * at the F9 seed even though the gate has just forced the displayed terrain back to `SEED` — a later
 * `setSmoothRadius` call would then bake against the wrong permutation until the next `regenerate`. Not
 * reachable by today's single-fresh-page-load gate flow, so left as documented residue rather than a
 * bigger restructure (`gate.ts` resetting `liveDocument` itself is a separate, pre-existing design choice
 * this stage didn't touch).
 */
export function syncNetworkForSeed(seed: number): void {
    setNetwork(liveDocument, seed, smoothRadius, falloffScale);
}

/** whether every marked overlay tile has drained through the atlas — the device gate polls this before
 *  its capture reads the frame (`gate.ts`). */
export function overlayIdle(): boolean {
    return overlayAtlas.idle();
}

/** one-shot GPU→CPU readback of the whole main vertex stream — the device gate's own arm (gate.ts): a
 *  regenerate-and-compare check needs the actual GPU output, not a re-derivation of it. Mirrors voxel's
 *  `readGrid()`; an assert-only bridge, never a per-frame readback. */
export async function readVertices(): Promise<Uint32Array> {
    if (!buffers) throw new Error("terrain: readVertices before the mesh buffers exist");
    const { device } = Compute;
    const size = VERTEX_COUNT * 16;
    const staging = device.createBuffer({
        label: "terrain-readback",
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder({ label: "terrain-readback" });
    enc.copyBufferToBuffer(buffers.verticesRaw, 0, staging, 0, size);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
}

const TerrainPlugin: Plugin = {
    name: "Terrain",
    dependencies: [RenderPlugin],
    systems: [OverlayRedrawSystem],
    initialize(state) {
        registerSurface(state, { name: "terrain", layout: terrainSurfaceLayout, fs: terrainFs });
    },
    warm,
    dispose: teardown,
};

export default TerrainPlugin;
