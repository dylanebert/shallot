import { computeViewProj, Views } from "@dylanebert/shallot/render/core";
import { decodePos } from "@dylanebert/shallot/utils/core";
import { markingDistance } from "./overlay/document";
import { captureProbePoints, generateNetwork } from "./overlay/network";
import { COVERAGE_BAND_PX, DASH_DUTY, DASH_OFFSET, DASH_PERIOD, EDGE_INSET } from "./overlay/tiles";
import { TERRAIN_QUANT } from "./terrain/generate";
import { CELLS, gridX, gridZ, SPACING, VERTS } from "./terrain/grid";
import { readVertices } from "./terrain/terrain";

// The device gate's world→screen bridge for the pixel-probe capture (the spec's flagged-risk validation —
// the fs composite is observable only in the rendered frame, never in the compute-write half `bun test`
// already covers). Rather than hand-deriving the orbit camera's trig in the Playwright driver, this reads
// the *real* live viewProj the renderer itself uses (`computeViewProj`, `@dylanebert/shallot/render/core`)
// — so the mapping can't drift from whatever the scene's orbit distance/pitch/fov actually are, and the
// driver never re-implements camera math it doesn't own.

/** a world point's screen position as a fraction of the canvas (0,0 = top-left, 1,1 = bottom-right) — the
 *  unit `test/roads.playwright.ts` multiplies by the captured screenshot's own pixel dimensions, so the mapping
 *  is DPR-independent. `depth` is the point's view-space depth (the clip w component, equal to view-space
 *  depth for this engine's perspective matrix, `engine/utils/math.ts`), carried for diagnostics — it plays
 *  no role in {@link TRANSITION_TOLERANCE_PX}, which is depth-independent by construction (see its doc). */
export interface ScreenPoint {
    x: number;
    y: number;
    depth: number;
}

const _vp = new Float32Array(16);

/** the eid of the canvas-presenting camera — never a plain `state.query([Camera])`/`state.only([Camera])`,
 *  since a casting light's pooled shadow camera (cascade/point-combo) also carries the `Camera` component
 *  (`sear/shadows.ts`'s `attachView`), so a scene with one shadow-casting light already has more than one
 *  `Camera`-tagged entity (measured: `state.only` warned "found multiple" the first time this ran against
 *  this scene's directional-light shadow). {@link Views} is the real disambiguator: only the display
 *  camera's entry carries a live `canvas` (a shadow camera's view is off-screen, `attachView`). */
function cameraEid(): number {
    for (const [eid, view] of Views) {
        if (view.canvas) return eid;
    }
    throw new Error("capture: no canvas-presenting camera in Views");
}

function canvasElement(): HTMLCanvasElement {
    const canvas = document.querySelector("canvas");
    if (!canvas) throw new Error("capture: no canvas element");
    return canvas;
}

/** project `points` (world x, y, z) through the scene's canvas-presenting camera into normalized screen
 *  fractions + view-space depth. Throws if no such camera or canvas exists — a capture with neither has
 *  nothing to probe. */
export function worldToScreen(
    points: ReadonlyArray<readonly [x: number, y: number, z: number]>,
): ScreenPoint[] {
    const eid = cameraEid();
    const canvas = canvasElement();
    const aspect = canvas.width / canvas.height;
    computeViewProj(eid, aspect, _vp);
    return points.map(([x, y, z]) => {
        // column-major mat4 × vec4(x, y, z, 1) (engine/utils/math.ts's convention)
        const cx = _vp[0] * x + _vp[4] * y + _vp[8] * z + _vp[12];
        const cy = _vp[1] * x + _vp[5] * y + _vp[9] * z + _vp[13];
        const cw = _vp[3] * x + _vp[7] * y + _vp[11] * z + _vp[15];
        const ndcX = cx / cw;
        const ndcY = cy / cw;
        // NDC is y-up in [-1,1]; screen space is y-down in [0,1] — flip.
        return { x: ndcX * 0.5 + 0.5, y: 1 - (ndcY * 0.5 + 0.5), depth: cw };
    });
}

/** one arbitrary world (x, z) → its full 3D world point, height read from the real generated vertex
 *  stream (`readVertices`, `terrain.ts`) at the *nearest* grid column, rather than re-deriving the noise
 *  function in JS — exact for a grid-aligned input (`overlay/network.ts`'s `captureProbePoints`), and a
 *  bounded-error approximation (at most half a grid cell's own height gradient) for an off-grid point,
 *  which only needs a point close enough to the real surface to project sensibly, not the exact flattened
 *  height. */
export async function withHeight(x: number, z: number): Promise<[x: number, y: number, z: number]> {
    const raw = await readVertices();
    const ix = gridX(x);
    const iz = gridZ(z);
    const idx = iz * VERTS + ix;
    const y = decodePos(raw[idx * 4], raw[idx * 4 + 1], TERRAIN_QUANT).y;
    return [x, y, z];
}

/** the real rendered surface height at an arbitrary off-grid world (x, z) — a two-triangle-per-quad linear
 *  reconstruction from the four surrounding vertices (`grid.ts`'s own split: `(i0,i2,i1)` and `(i1,i2,i3)`),
 *  the same plane the rasterizer actually draws. Unlike {@link withHeight}'s nearest-vertex read (a
 *  bounded approximation, fine for projecting a point sensibly), this is exact everywhere inside the
 *  mesh's footprint — `flatness.ts`'s corridor-flatness oracle needs the surface a viewer's eye actually
 *  sees at a point that generally doesn't land on a vertex, since the defect it measures *is* the gap
 *  between this reconstruction and the continuous analytic height at the same point. `raw` is one
 *  `readVertices()` readback, reused across every sample in a scan so the GPU is read back once, not once
 *  per probe step. `spacing`/`cells` default to the live mesh's own `grid.ts` constants —
 *  `flatness.ts`'s mesh-resolution discrimination arm (stage 12) is the one caller that overrides them,
 *  reconstructing a *different*-resolution lattice `raw` was built at without a second,
 *  independently-drifting copy of this same triangle-interpolation formula. */
export function meshHeightAt(
    raw: Uint32Array,
    x: number,
    z: number,
    spacing: number = SPACING,
    cells: number = CELLS,
): number {
    const half = cells / 2;
    const verts = cells + 1;
    const fx = x / spacing + half;
    const fz = z / spacing + half;
    const ix0 = Math.floor(fx);
    const iz0 = Math.floor(fz);
    const tx = fx - ix0;
    const tz = fz - iz0;
    const heightOf = (ix: number, iz: number): number => {
        const idx = iz * verts + ix;
        return decodePos(raw[idx * 4], raw[idx * 4 + 1], TERRAIN_QUANT).y;
    };
    const h00 = heightOf(ix0, iz0);
    const h10 = heightOf(ix0 + 1, iz0);
    const h01 = heightOf(ix0, iz0 + 1);
    const h11 = heightOf(ix0 + 1, iz0 + 1);
    // lower-left triangle (i0, i2, i1) covers tx + tz <= 1; upper-right (i1, i2, i3) the complement —
    // grid.ts's own winding, so this interpolates over exactly the triangle the renderer draws there.
    if (tx + tz <= 1) return h00 + tx * (h10 - h00) + tz * (h01 - h00);
    const utx = 1 - tx;
    const utz = 1 - tz;
    return h11 + utx * (h01 - h11) + utz * (h10 - h11);
}

/** the capture gate's on-road/off-road world probe points over the network's one fixed standard chord
 *  (`overlay/network.ts`'s `captureProbePoints`, `terrain.ts`'s live boot document — no seed,
 *  `roads-interactive.md` stage 1), heights read from the live generated terrain. */
export async function capturePoints(): Promise<{
    onRoad: [number, number, number];
    offRoad: [number, number, number];
}> {
    const { onRoad, offRoad } = captureProbePoints();
    const [onRoadPoint, offRoadPoint] = await Promise.all([
        withHeight(...onRoad),
        withHeight(...offRoad),
    ]);
    return { onRoad: onRoadPoint, offRoad: offRoadPoint };
}

/**
 * the capture gate's marking probe points — four world points derived from the document (never
 * hand-placed), each sitting on a distinct marking class the fs composites: one on an edge line
 * (solid white), one on the asphalt between the edge line and the centreline, one in a dash gap
 * (centreline absent → reads road), and one on a dash (broken yellow centreline). The device gate
 * asserts the luminance bands at these four points are DISJOINT — a probe that cannot discriminate
 * the classes it gates is not evidence (roads-interactive.md stage 3, Marking fidelity).
 *
 * Derivation: the standard chord's midpoint, offset along the road's normal by (halfWidth −
 * EDGE_INSET) for the edge line, by halfWidth/2 for the asphalt between, and along the centreline
 * at stations chosen by the dash phase (fract(s / DASH_PERIOD) < DASH_DUTY for a dash, >= DASH_DUTY
 * for a gap). Heights read from the live generated terrain.
 */
export async function markingProbePoints(): Promise<{
    edgeLine: [number, number, number];
    asphalt: [number, number, number];
    dashGap: [number, number, number];
    dash: [number, number, number];
}> {
    const doc = generateNetwork();
    const line = doc.polylines[0];
    const [[ax, az], [bx, bz]] = line.points;
    const mx = (ax + bx) / 2;
    const mz = (az + bz) / 2;
    const dx = bx - ax;
    const dz = bz - az;
    const len = Math.hypot(dx, dz) || 1;
    const nx = -dz / len; // unit normal (network.ts's convention)
    const nz = dx / len;
    const halfWidth = line.halfWidth;

    // edge line: midpoint, offset laterally by (halfWidth - EDGE_INSET) toward one edge
    const edgeOffset = halfWidth - EDGE_INSET;
    const edgeLineX = mx + nx * edgeOffset;
    const edgeLineZ = mz + nz * edgeOffset;

    // asphalt between: midpoint, offset laterally by halfWidth/2 (between centreline and edge line)
    const asphaltOffset = halfWidth / 2;
    const asphaltX = mx + nx * asphaltOffset;
    const asphaltZ = mz + nz * asphaltOffset;

    // dash / dash gap: scan along the centreline for the first station in a dash and the first in a gap.
    // Start at the midpoint (station = len/2) and step in both directions to find each class.
    const midStation = len / 2;
    let dashStation = -1;
    let gapStation = -1;
    const step = 0.5; // half-metre steps — finer than LINE_WIDTH, coarse enough to be fast
    for (let s = midStation; s < len && (dashStation < 0 || gapStation < 0); s += step) {
        const phase = ((s + DASH_OFFSET) / DASH_PERIOD) % 1;
        if (dashStation < 0 && phase < DASH_DUTY) dashStation = s;
        if (gapStation < 0 && phase >= DASH_DUTY) gapStation = s;
    }
    // fall back to scanning backward if forward didn't find both
    for (let s = midStation - step; s >= 0 && (dashStation < 0 || gapStation < 0); s -= step) {
        const phase = ((s + DASH_OFFSET) / DASH_PERIOD) % 1;
        if (dashStation < 0 && phase < DASH_DUTY) dashStation = s;
        if (gapStation < 0 && phase >= DASH_DUTY) gapStation = s;
    }
    if (dashStation < 0 || gapStation < 0) {
        throw new Error("markingProbePoints: could not find both a dash and a gap station");
    }

    const ux = dx / len;
    const uz = dz / len;
    const dashX = ax + ux * dashStation;
    const dashZ = az + uz * dashStation;
    const gapX = ax + ux * gapStation;
    const gapZ = az + uz * gapStation;

    // verify the probe points classify correctly against the document
    if (markingDistance(edgeLineX, edgeLineZ, doc) >= 0) {
        throw new Error("markingProbePoints: edge line probe is not inside a marking");
    }
    if (markingDistance(asphaltX, asphaltZ, doc) <= 0) {
        throw new Error("markingProbePoints: asphalt probe is inside a marking");
    }
    if (markingDistance(dashX, dashZ, doc) >= 0) {
        throw new Error("markingProbePoints: dash probe is not inside a marking");
    }
    if (markingDistance(gapX, gapZ, doc) <= 0) {
        throw new Error("markingProbePoints: dash gap probe is inside a marking");
    }

    const [edgeLinePt, asphaltPt, dashPt, gapPt] = await Promise.all([
        withHeight(edgeLineX, edgeLineZ),
        withHeight(asphaltX, asphaltZ),
        withHeight(dashX, dashZ),
        withHeight(gapX, gapZ),
    ]);
    return {
        edgeLine: edgeLinePt,
        asphalt: asphaltPt,
        dashGap: gapPt,
        dash: dashPt,
    };
}

/**
 * the derived antialiasing-band screen-pixel tolerance for the capture's boundary-transition probe.
 *
 * The governing mechanism is the fs's own coverage formula (`terrain.ts`): `fw = COVERAGE_BAND_PX·fwidth(dist) + ε`,
 * `coverage = clamp(0.5 − dist/fw, 0, 1)`. Coverage leaves `[0, 1]` exactly when `dist/fw` leaves
 * `[−0.5, 0.5]` — i.e. when `dist` moves by `fw ≈ 0.5·fwidth(dist)` — and `fwidth` is *defined* as the
 * change in its argument over one screen pixel (`|ddx| + |ddy|`), so the coverage band is designed to span
 * **about half a screen pixel**, independent of world scale, texel density, or camera distance (that
 * self-scaling property is the whole point of fwidth-thresholded AA, Green SIGGRAPH 2007 — a texel-size
 * bound converted through the camera's fov/depth would be the *wrong* quantity, and reads far too tight
 * whenever the atlas is minified, the common case at this showcase's camera distance). This constant
 * is that band ×4 — imported, not restated, so the two can't drift — allowing for what the formula doesn't model: the `ε` softening
 * term, the atlas sampler's own bilinear interpolation (one more texel-to-texel blend before `fwidth` ever
 * measures it), and discrete-pixel sampling slop in the probe itself — a fixed multiple of the mechanism's
 * own band, not a value fitted to one observed capture.
 */
export const TRANSITION_TOLERANCE_PX = COVERAGE_BAND_PX * 4;
