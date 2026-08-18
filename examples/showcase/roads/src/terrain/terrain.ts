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
import * as overlayAtlas from "../overlay/atlas";
import { packStrokeTile, strokeRect } from "../overlay/stroke";
import { DIST_RANGE, TILE_SIZE, TILES_PER_SIDE } from "../overlay/tiles";
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
    const fw = std.fwidth(dist) * d.f32(0.5) + d.f32(1e-5);
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

    // the stage's hand-authored known pattern (overlay/stroke.ts) — stands in for a real stroke document
    // (stage 5). Marking it dirty here (not per-frame) means one redraw burst at warm, not a repeated mark.
    overlayAtlas.markDirty(strokeRect());

    bindTerrainKernel(vertices, position);
    if (state.signal.aborted) return;
    await generate(SEED);
}

/** drain the overlay's per-frame dirty-tile throttle (`overlay/atlas.ts`'s `redraw`) — this stage's only
 *  producer is the hand-authored stroke, so `packStrokeTile` stands in for the real stroke-document
 *  rasterizer (stage 5). Exported so the device gate can assert draining actually happened. */
const OverlayRedrawSystem: System = {
    name: "roads-overlay-redraw",
    group: "simulation",
    annotations: { mode: "always" },
    update() {
        overlayAtlas.redraw(packStrokeTile);
    },
};

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
