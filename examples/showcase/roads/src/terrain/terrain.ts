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
import type { StrokeDocument } from "../overlay/document";
import { generateNetwork } from "../overlay/network";
import {
    CENTRE_ALBEDO,
    COVERAGE_BAND_PX,
    DASH_DUTY,
    DASH_OFFSET,
    DASH_PERIOD,
    DASH_SEGMENT,
    DIST_RANGE,
    EDGE_ALBEDO,
    EDGE_INSET,
    LINE_HALF_WIDTH,
    TILE_SIZE,
    TILES_PER_SIDE,
} from "../overlay/tiles";
import { dispatchPosts, warmPosts } from "../posts";
import { setNetwork, warmNetwork } from "./flatten";
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
    chord: { type: "uniform", struct: overlayAtlas.ChordUniform },
});

/** two-edge analytic pixel coverage — the fraction of a pixel a stripe of half-width `h` covers
 *  on signed cross-coordinate `x` with pixel footprint `p`. `cover(e) = clamp((e - x)/p + 0.5, 0, 1)`,
 *  `alpha = cover(h) - cover(-h)`. Correct through the sub-pixel regime by construction: when both
 *  edges land inside one pixel the terms subtract to the stripe's true fractional coverage, so the line
 *  fades proportionally instead of snapping to fully-opaque or vanishing. CPU-callable so `bun test`
 *  exercises this exact math with no device. */
export const coverFn = tgpu.fn(
    [d.f32, d.f32, d.f32],
    d.f32,
)((e, x, p) => {
    "use gpu";
    return std.clamp((e - x) / p + d.f32(0.5), 0, 1);
});

/** the signed distance to the nearest road marking from the chord — the GPU twin of `document.ts`'s
 *  `markingDistanceForSegment`, independently derived via the clamped-projection form (same split as
 *  `segmentDistanceGpu` vs `segmentDistance`). Two solid edge lines inset from the road edge (centred
 *  at d = −EDGE_INSET on the existing edge distance) and a broken centreline (perpendicular distance via
 *  the cross product, station along the chord via fract(s / DASH_PERIOD) < DASH_DUTY). CPU-callable so
 *  `bun test` exercises this exact math with no device — the GPU side of the marking differential oracle.
 *  `terrain.ts` never imports `markingDistance` — the two derivations are independent by contract.
 * @example markingDistanceFromChord(0, 0, -100, 0, 100, 0, 4) // on the centreline, in a gap → positive */
export const markingDistanceFromChord = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((px, pz, ax, az, bx, bz, halfWidth) => {
    "use gpu";
    const abx = bx - ax;
    const abz = bz - az;
    const apx = px - ax;
    const apz = pz - az;
    const dd = abx * abx + abz * abz;
    let t = d.f32(0);
    let tRaw = d.f32(0);
    if (dd > 0) {
        tRaw = (apx * abx + apz * abz) / dd;
        t = std.clamp(tRaw, 0, 1);
    }
    const cx = ax + t * abx;
    const cz = az + t * abz;
    const dist = std.distance(d.vec2f(px, pz), d.vec2f(cx, cz));
    const edgeDist = dist - halfWidth;

    // edge line: solid, centred at d = −EDGE_INSET, width LINE_WIDTH
    const edgeLineDist = std.abs(edgeDist + d.f32(EDGE_INSET)) - d.f32(LINE_HALF_WIDTH);

    // centreline: broken, lateral from the cross product, longitudinal from the dash phase
    const len = std.sqrt(dd);
    const cross = len > 0 ? (apx * abz - apz * abx) / len : 0;
    const lateral = std.abs(cross) - d.f32(LINE_HALF_WIDTH);
    const along = t * len;
    let longitudinal = d.f32(0);
    if (tRaw < 0) {
        longitudinal = -tRaw * len;
    } else if (tRaw > 1) {
        longitudinal = (tRaw - 1) * len;
    } else {
        const phase = std.fract((along + d.f32(DASH_OFFSET)) / d.f32(DASH_PERIOD));
        if (phase < d.f32(DASH_DUTY)) {
            longitudinal = -std.min(phase, d.f32(DASH_DUTY) - phase) * d.f32(DASH_PERIOD);
        } else {
            longitudinal = std.min(phase - d.f32(DASH_DUTY), d.f32(1) - phase) * d.f32(DASH_PERIOD);
        }
    }
    const centreDist = std.max(lateral, longitudinal);

    return std.min(edgeLineDist, centreDist);
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
    const albedoSample = std.textureSample(
        terrainSurfaceLayout.$.albedo,
        terrainSurfaceLayout.$.overlaySamp,
        uv,
        layerU,
    );
    const overlay = albedoSample.xyz;
    // decode the r8unorm distance channel back to a signed world-metre distance — the fs's half of the
    // codec overlay/tiles.ts's encodeDist writes (kept in sync by hand: DIST_RANGE is the one shared
    // constant both sides read; TGSL can't call a plain JS helper, so this stays inline).
    const dist = (distSample - d.f32(0.5)) * d.f32(2) * d.f32(DIST_RANGE);
    const fw = std.fwidth(dist) * d.f32(COVERAGE_BAND_PX) + d.f32(1e-5);
    const resident = std.select(d.f32(0), d.f32(1), layer >= 0);
    const coverage = std.clamp(d.f32(0.5) - dist / fw, 0, 1) * resident;

    // the marking channel: analytic in the fs from the chord uniform, not baked into a texel.
    // Two-edge pixel coverage — the fraction of the pixel the marking covers — not a smoothstep threshold.
    // For a stripe of half-width h on signed cross-coordinate x with p = max(fwidth(x), 1e-6):
    //   cover(e) = clamp((e - x)/p + 0.5, 0, 1),  alpha = cover(h) - cover(-h)
    // This is correct through the sub-pixel regime by construction.
    const chord = terrainSurfaceLayout.$.chord;
    const ax = chord.a.x;
    const az = chord.a.y;
    const bx = chord.b.x;
    const bz = chord.b.y;
    const halfWidth = chord.halfWidth;

    // chord projection
    const abx = bx - ax;
    const abz = bz - az;
    const apx = ctx.world.x - ax;
    const apz = ctx.world.z - az;
    const dd = abx * abx + abz * abz;
    let tRaw = d.f32(0);
    if (dd > 0) tRaw = (apx * abx + apz * abz) / dd;
    const t = std.clamp(tRaw, 0, 1);
    const cx = ax + t * abx;
    const cz = az + t * abz;
    const distToSegment = std.distance(d.vec2f(ctx.world.x, ctx.world.z), d.vec2f(cx, cz));
    const len = std.sqrt(dd);
    const cross = len > 0 ? (apx * abz - apz * abx) / len : 0;

    // edge line: solid, two-edge coverage on the edge cross-coordinate
    // (centred at d = −EDGE_INSET on the edge distance, half-width LINE_HALF_WIDTH)
    const edgeX = distToSegment - halfWidth + d.f32(EDGE_INSET);
    const edgeP = std.max(std.fwidth(edgeX), d.f32(1e-6));
    const edgeAlpha =
        coverFn(d.f32(LINE_HALF_WIDTH), edgeX, edgeP) -
        coverFn(-d.f32(LINE_HALF_WIDTH), edgeX, edgeP);

    // centreline: broken, lateral coverage * dash coverage * endpoint coverage
    // lateral: two-edge coverage on the signed perpendicular distance (cross)
    const centreP = std.max(std.fwidth(cross), d.f32(1e-6));
    const lateralAlpha =
        coverFn(d.f32(LINE_HALF_WIDTH), cross, centreP) -
        coverFn(-d.f32(LINE_HALF_WIDTH), cross, centreP);

    // dash: two-edge coverage along the station axis, converging to DASH_DUTY past Nyquist
    const s = tRaw * len;
    const sP = std.max(std.fwidth(s), d.f32(1e-6));
    const sRel =
        s +
        d.f32(DASH_OFFSET) -
        std.floor((s + d.f32(DASH_OFFSET)) / d.f32(DASH_PERIOD)) * d.f32(DASH_PERIOD);
    const dashCov = coverFn(d.f32(DASH_SEGMENT), sRel, sP) - coverFn(d.f32(0), sRel, sP);
    const dashCovWrap =
        coverFn(d.f32(DASH_PERIOD + DASH_SEGMENT), sRel, sP) -
        coverFn(d.f32(DASH_PERIOD), sRel, sP);
    let dashCoverage = std.clamp(dashCov + dashCovWrap, 0, 1);
    // Nyquist blend: aliasing begins when the sample footprint passes half the dash period — the
    // highest frequency a signal can represent is 2 samples per cycle, so below half a period the dash
    // is still fully resolvable and the blend must be inactive (otherwise the pattern washes toward
    // continuous inside the regime where it still reads crisp). The blend starts at half a period and
    // reaches full convergence at one period: a far dashed line reads as a faint continuous line at
    // DASH_DUTY opacity, never as resolved flickering dots.
    const nyquist = std.clamp(
        (sP - d.f32(DASH_PERIOD) * d.f32(0.5)) / (d.f32(DASH_PERIOD) * d.f32(0.5)),
        0,
        1,
    );
    dashCoverage = std.mix(dashCoverage, d.f32(DASH_DUTY), nyquist);

    // endpoint: the centreline exists only for station in [0, len]
    const endpointAlpha = coverFn(d.f32(len), s, sP) - coverFn(d.f32(0), s, sP);

    const centreAlpha = lateralAlpha * dashCoverage * endpointAlpha;

    // select the marking class: the nearest marking (edge vs centre) determines the albedo
    const markingDist = markingDistanceFromChord(
        ctx.world.x,
        ctx.world.z,
        ax,
        az,
        bx,
        bz,
        halfWidth,
    );
    const edgeLineDist =
        std.abs(distToSegment - halfWidth + d.f32(EDGE_INSET)) - d.f32(LINE_HALF_WIDTH);
    const isCentre = markingDist < edgeLineDist;
    const markingAlbedo = std.select(
        d.vec3f(d.f32(EDGE_ALBEDO[0]), d.f32(EDGE_ALBEDO[1]), d.f32(EDGE_ALBEDO[2])),
        d.vec3f(d.f32(CENTRE_ALBEDO[0]), d.f32(CENTRE_ALBEDO[1]), d.f32(CENTRE_ALBEDO[2])),
        isCentre,
    );
    const markingCoverage = std.max(edgeAlpha, centreAlpha);

    // mix the marking albedo over the road albedo before the terrain composite
    const overlayWithMarkings = std.mix(overlay, markingAlbedo, markingCoverage);
    color = std.mix(color, overlayWithMarkings, coverage);

    return d.vec4f(lit(color, ctx.worldNormal), 1);
});

/** the emitted terrain fs WGSL — the device-free structural seam the overlay composite tests resolve
 *  (`terrain.test.ts`, the `generate.test.ts`/`noise.test.ts` idiom). */
export function terrainFsWgsl(): string {
    return tgpu.resolve([coverFn, markingDistanceFromChord, terrainFs], { names: "strict" });
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

// The boot document is the network's one fixed standard chord (`overlay/network.ts`'s
// `generateNetwork`, no seed). `capture.ts`'s `capturePoints` derives its on/off-road probe points
// from this exact `generateNetwork()` call (`overlay/network.ts`'s `captureProbePoints`), so the two
// can't drift apart. `overlay/stroke.ts`'s hand-authored stroke stays live only as the CPU-vs-GPU
// differential's known pattern (`document.test.ts`), not what boots on screen. {@link regenerate} draws a
// fresh terrain seed on demand (the seed control, `boot.ts`'s F9 handler) and resets the road to the
// standard chord — the seed drives the noise permutation alone, never the road's placement.
let liveDocument: StrokeDocument = generateNetwork();
let currentSeed = SEED;

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
        bindings: overlayAtlas.bindings(), // the overlay entries terrainSurfaceLayout declares
    };
    Meshes.register(mesh);
    Draws.register({ name: "terrain", surface: "terrain", mesh: "terrain", args: { indirect } });

    // the boot document (the procedural network at this module's SEED, above), rasterized by
    // overlay/rasterize.ts's GPU kernel. Marking it dirty here (not per-frame) means one redraw burst at
    // warm, not a repeated mark.
    overlayAtlas.markDirty(liveDocument);

    bindTerrainKernel(vertices, position);
    warmNetwork(state); // its own onDispose registration, the same pattern
    warmPosts(state); // posts buffer + surface + registered draw
    setNetwork(liveDocument, currentSeed); // the flatten kernel's geometry input, kept in sync with
    // the overlay's own document and the height kernel's own permutation seed
    overlayAtlas.updateChord(liveDocument); // the fs's marking geometry input
    await dispatchPosts(currentSeed); // write post positions for the boot chord
    if (state.signal.aborted) return;
    await generate(SEED);
}

/** drain the overlay's per-frame dirty-tile throttle (`overlay/atlas.ts`'s `redraw`), rasterizing
 *  {@link liveDocument}'s content into every drained tile via `overlay/rasterize.ts`'s GPU kernel. Exported
 *  so the device gate can assert draining actually happened. */
const OverlayRedrawSystem: System = {
    name: "roads-overlay-redraw",
    group: "simulation",
    update() {
        overlayAtlas.redraw(liveDocument);
    },
};

/**
 * reseed the terrain's noise permutation and re-dispatch the height kernel, resetting the live document
 * to the standard chord — the seed control's live reseed (`boot.ts`'s F9 handler). The seed drives the
 * noise permutation alone ("the road is not seeded"), never the road's
 * placement — a person's drag is the only thing that ever moves it. `overlayAtlas.invalidate()`
 * releases every tile the outgoing document left resident (the indirection mirror reset to unallocated,
 * the layer counter restarted, any still-pending redraw dropped) *before* the new network's tiles are
 * marked dirty (`overlay/atlas.ts`'s `markDirty`, the exact-set oracle `document.test.ts` pins) —
 * otherwise a tile the old document touched but the new one doesn't would keep its stale content forever,
 * and each reseed's fresh handful of layers would pile onto the last's until the fixed-size atlas ran out
 * (`overlay/queue.test.ts` drives the fixed order against real reseeds; the pre-invalidation overflow has
 * no live demonstration, the road not moving with the seed — see that file's note).
 * Rebinds the flatten kernel's geometry (`flatten.ts`'s `setNetwork`) before `generate` re-dispatches, so
 * the next-drawn frame's heights and overlay both reflect the new terrain in one call — the
 * "affected-region remesh" the spec's Approach names.
 */
export async function regenerate(seed: number): Promise<void> {
    currentSeed = seed;
    liveDocument = generateNetwork();
    setNetwork(liveDocument, seed);
    await dispatchPosts(seed); // re-write post positions for the reset chord
    overlayAtlas.updateChord(liveDocument);
    overlayAtlas.invalidate();
    overlayAtlas.markDirty(liveDocument);
    await generate(seed);
}

/**
 * apply an edited road document to the live terrain: re-bake the flatten kernel's geometry
 * (`flatten.ts`'s `setNetwork`), re-tile the overlay atlas (`overlay/atlas.ts`'s `retile` — marks
 * `tiles(old) ∪ tiles(new)` dirty and releases `tiles(old) − tiles(new)`), and re-dispatch the height
 * kernel (`generate(currentSeed)`). The seed is unchanged — a person's drag moves the road, not the
 * terrain permutation ("the seed owns the terrain; the road is not seeded"). The edit system's per-frame
 * system (`edit.ts`) calls this on an admissible drag; the
 * handle re-placement is the edit system's own job (it reads {@link getDocument} each frame and sets the
 * handle Transform positions to the endpoints at `y = heightAtCpu`).
 */
export async function editDocument(doc: StrokeDocument): Promise<void> {
    const oldDoc = liveDocument;
    liveDocument = doc;
    setNetwork(doc, currentSeed);
    await dispatchPosts(currentSeed); // re-write post positions for the edited chord
    overlayAtlas.updateChord(doc);
    overlayAtlas.retile(oldDoc, doc);
    await generate(currentSeed);
}

/** the live road document — the edit system reads this each frame to place the handles at the endpoints
 *  and to clamp a drag to the `ROAD_MIN_LENGTH` floor. A reseed (`regenerate`) resets it to the standard
 *  chord. */
export function getDocument(): StrokeDocument {
    return liveDocument;
}

/** the seed the height kernel was last dispatched with — the live permutation seed, not the boot
 *  constant `SEED`. After an F9 reseed (`regenerate`) this differs from `SEED`, and a CPU twin that
 *  bakes against `SEED` instead of this accessor (e.g. `checkPosts`'s `makePermutation`/
 *  `buildNetworkGeometry`) diverges from the GPU output — the permutation
 *  and the falloff change with the seed. An accessor rather than exporting `currentSeed` directly so the
 *  mutable module state stays write-only from outside this module. */
export function getCurrentSeed(): number {
    return currentSeed;
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
 * `setNetwork` call would then bake against the wrong permutation until the next `regenerate`. Not
 * reachable by today's single-fresh-page-load gate flow, so left as documented residue rather than a
 * bigger restructure (`gate.ts` resetting `liveDocument` itself is a separate, pre-existing design choice).
 */
export function syncNetworkForSeed(seed: number): void {
    setNetwork(liveDocument, seed);
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
