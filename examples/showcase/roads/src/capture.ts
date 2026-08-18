import { computeViewProj, Views } from "@dylanebert/shallot/render/core";
import { decodePos } from "@dylanebert/shallot/utils/core";
import { OFF_ROAD_POINT, ON_ROAD_POINT } from "./overlay/stroke";
import { TERRAIN_QUANT } from "./terrain/generate";
import { gridX, gridZ, VERTS } from "./terrain/grid";
import { readVertices } from "./terrain/terrain";

// The device gate's world→screen bridge for the pixel-probe capture (the spec's flagged-risk validation,
// stage 4's own arm — `checks.md`: the fs composite is observable only in the rendered frame). Rather than
// hand-deriving the orbit camera's trig in the Playwright driver, this reads the *real* live viewProj the
// renderer itself uses (`computeViewProj`, `@dylanebert/shallot/render/core`) — so the mapping can't drift
// from whatever the scene's orbit distance/pitch/fov actually are, and the driver never re-implements
// camera math it doesn't own.

/** a world point's screen position as a fraction of the canvas (0,0 = top-left, 1,1 = bottom-right) — the
 *  unit `test/roads.spec.ts` multiplies by the captured screenshot's own pixel dimensions, so the mapping
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

/** one grid-aligned world (x, z) → its full 3D world point, height read from the real generated vertex
 *  stream (`readVertices`, `terrain.ts`) rather than re-deriving the noise function in JS — {@link gridX}/
 *  {@link gridZ} require an exact grid column, which `overlay/stroke.ts`'s ON_ROAD_POINT/OFF_ROAD_POINT are
 *  built to be. */
async function withHeight(x: number, z: number): Promise<[x: number, y: number, z: number]> {
    const raw = await readVertices();
    const ix = gridX(x);
    const iz = gridZ(z);
    const idx = iz * VERTS + ix;
    const y = decodePos(raw[idx * 4], raw[idx * 4 + 1], TERRAIN_QUANT).y;
    return [x, y, z];
}

/** the capture gate's fixed on-road/off-road world probe points, heights read from the live generated
 *  terrain (`overlay/stroke.ts`'s ON_ROAD_POINT/OFF_ROAD_POINT). */
export async function capturePoints(): Promise<{
    onRoad: [number, number, number];
    offRoad: [number, number, number];
}> {
    const [onRoad, offRoad] = await Promise.all([
        withHeight(...ON_ROAD_POINT),
        withHeight(...OFF_ROAD_POINT),
    ]);
    return { onRoad, offRoad };
}

/**
 * the derived antialiasing-band screen-pixel tolerance for the capture's boundary-transition probe.
 *
 * The governing mechanism is the fs's own coverage formula (`terrain.ts`): `fw = 0.5·fwidth(dist) + ε`,
 * `coverage = clamp(0.5 − dist/fw, 0, 1)`. Coverage leaves `[0, 1]` exactly when `dist/fw` leaves
 * `[−0.5, 0.5]` — i.e. when `dist` moves by `fw ≈ 0.5·fwidth(dist)` — and `fwidth` is *defined* as the
 * change in its argument over one screen pixel (`|ddx| + |ddy|`), so the coverage band is designed to span
 * **about half a screen pixel**, independent of world scale, texel density, or camera distance (that
 * self-scaling property is the whole point of fwidth-thresholded AA, Green SIGGRAPH 2007 — a texel-size
 * bound converted through the camera's fov/depth would be the *wrong* quantity, and reads far too tight
 * whenever the atlas is minified, the common case at this showcase's camera distance). This constant
 * allows a further ×4 over that ~0.5 px formula band for what the formula doesn't model: the `ε` softening
 * term, the atlas sampler's own bilinear interpolation (one more texel-to-texel blend before `fwidth` ever
 * measures it), and discrete-pixel sampling slop in the probe itself — a fixed multiple of the mechanism's
 * own band, not a value fitted to one observed capture.
 */
export const TRANSITION_TOLERANCE_PX = 2;
