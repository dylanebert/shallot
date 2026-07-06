// The transform gizmo's geometry — pure math, no rendering, no engine state. Screen↔world projection, a
// cursor ray, hit-testing, and the constrained drags, all as transforms the `.svelte`/system wiring calls
// into (it supplies `viewProj`/`invViewProj` from `computeViewProj` + `invert`, the cursor, the canvas
// size, and the camera `eye`). Pure so the projection + drag math is `bun test`-covered without a mounted
// editor or a device, the way `lib/pick`'s `isClick` is. Matrices are column-major (WGSL convention, what
// `computeViewProj` emits); screen is CSS px, origin top-left. Editor-only, and physics-free by design —
// the camera-ray math that `physics/core` also has lives here too, so the editor never depends on physics.
//
// A handle is a tagged binding (axis / plane / screen / ring / uniform), not geometry. A manipulator
// declares its handle set; `glyphs()` projects that set to screen geometry once per frame, consumed by
// BOTH the render and the hit-test, so what's drawn is exactly what's grabbable for every kind.

import { Tool } from "./tool";

export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number];

/** a world-space ray: `origin` plus a unit `dir`. */
export interface Ray {
    origin: Vec3;
    dir: Vec3;
}

/** the gizmo's world-space axis frame — X, Y, Z unit vectors. The local frame ({@link localAxes}) is the
 * same three vectors rotated by the entity's orientation. */
export const WORLD_AXES: readonly [Vec3, Vec3, Vec3] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
];

/** the gizmo's on-screen size in CSS px — the axis-shaft length and ring radius. Render and the hit-test
 * both derive their world scale from it ({@link gizmoScale}), so what's drawn is what's grabbable. */
export const GIZMO_PX = 120;

// snap increments (hold-Ctrl): world units / radians / scale factor. Relative — the drag's delta quantizes,
// so an off-grid entity snaps its movement, not its absolute coordinate.
const GRID_SNAP = 1;
const ANGLE_SNAP = Math.PI / 12; // 15°
const SCALE_SNAP = 0.1;

function snapTo(v: number, step: number): number {
    return Math.round(v / step) * step;
}

// column-major mat4 × (x,y,z,w) → [X,Y,Z,W]; column c is m[c*4 .. c*4+3], row r picks m[c*4+r]
function apply(
    m: ArrayLike<number>,
    x: number,
    y: number,
    z: number,
    w: number,
): [number, number, number, number] {
    return [
        m[0] * x + m[4] * y + m[8] * z + m[12] * w,
        m[1] * x + m[5] * y + m[9] * z + m[13] * w,
        m[2] * x + m[6] * y + m[10] * z + m[14] * w,
        m[3] * x + m[7] * y + m[11] * z + m[15] * w,
    ];
}

function unproject(inv: ArrayLike<number>, nx: number, ny: number, nz: number): Vec3 {
    const [X, Y, Z, W] = apply(inv, nx, ny, nz, 1);
    return [X / W, Y / W, Z / W];
}

function sub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function norm(a: Vec3): Vec3 {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
}

// quaternion from a unit axis + angle (radians)
function axisAngle(axis: Vec3, a: number): Quat {
    const s = Math.sin(a / 2);
    return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(a / 2)];
}

// Hamilton product a∘b — the rotation that applies b first, then a. `qmul(delta, rot)` rotates `rot` by the
// world-space `delta`.
function qmul(a: Quat, b: Quat): Quat {
    return [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ];
}

function qnorm(q: Quat): Quat {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

// apply a quaternion to a vector (v' = q·v·q*)
export function qrotvec(q: Quat, v: Vec3): Vec3 {
    const t = cross([q[0], q[1], q[2]], v);
    const t2: Vec3 = [t[0] * 2, t[1] * 2, t[2] * 2];
    return add(add(v, [t2[0] * q[3], t2[1] * q[3], t2[2] * q[3]]), cross([q[0], q[1], q[2]], t2));
}

/** the entity's local axis frame — the world basis rotated by its orientation `q` (the columns of `q`'s
 * rotation matrix). Unit and orthogonal, so a {@link Manipulator} reads it exactly like {@link WORLD_AXES}.
 */
export function localAxes(q: Quat): [Vec3, Vec3, Vec3] {
    const [x, y, z, w] = q;
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    return [
        [1 - 2 * (yy + zz), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (xx + zz), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (xx + yy)],
    ];
}

// distance from point p to the segment a→b, all in 2D screen space
function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
    const vx = bx - ax;
    const vy = by - ay;
    const c1 = vx * (px - ax) + vy * (py - ay);
    if (c1 <= 0) return Math.hypot(px - ax, py - ay);
    const c2 = vx * vx + vy * vy;
    if (c2 <= c1) return Math.hypot(px - bx, py - by);
    const t = c1 / c2;
    return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
}

/** whether the screen point is inside the convex quad `pts` (4 flat corners `[x0,y0,…x3,y3]`). The cursor
 * is inside when it sits on the same side of every edge — the cross-product signs all agree. */
export function pointInQuad(px: number, py: number, pts: number[]): boolean {
    let sign = 0;
    for (let i = 0; i < 4; i++) {
        const ax = pts[i * 2];
        const ay = pts[i * 2 + 1];
        const bx = pts[((i + 1) % 4) * 2];
        const by = pts[((i + 1) % 4) * 2 + 1];
        const c = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        if (c !== 0) {
            const s = c > 0 ? 1 : -1;
            if (sign === 0) sign = s;
            else if (s !== sign) return false;
        }
    }
    return true;
}

/**
 * project a world point to screen px, or null when it's on the camera plane (`W ≈ 0`). `behind` flags a
 * point past the camera (`W ≤ 0`) — its screen position is a mirrored artifact, so the caller skips it.
 */
export function project(
    viewProj: ArrayLike<number>,
    p: Vec3,
    width: number,
    height: number,
): { x: number; y: number; behind: boolean } | null {
    const [X, Y, , W] = apply(viewProj, p[0], p[1], p[2], 1);
    if (Math.abs(W) < 1e-6) return null;
    return {
        x: (X / W) * 0.5 * width + 0.5 * width,
        y: 0.5 * height - (Y / W) * 0.5 * height,
        behind: W <= 0,
    };
}

/** a world-space line through a screen pixel, for ray-plane / closest-point math. Under the engine's
 * reverse-Z (near→1/far→0) `origin` lands on the *far* plane and `dir` points back toward the camera — a
 * valid infinite line, but `origin` is not the camera position and `dir` is not "into the scene". */
export function cursorRay(
    invViewProj: ArrayLike<number>,
    sx: number,
    sy: number,
    width: number,
    height: number,
): Ray {
    const nx = (sx / width) * 2 - 1;
    const ny = 1 - (sy / height) * 2;
    const near = unproject(invViewProj, nx, ny, 0);
    const far = unproject(invViewProj, nx, ny, 1);
    return { origin: near, dir: norm(sub(far, near)) };
}

/**
 * the parameter `t` along the axis line (`axisOrigin + t·axisDir`) at the point nearest the ray — the
 * standard closest-points-between-two-lines solve. NaN when the ray is parallel to the axis (no unique
 * nearest point). `axisDir` is assumed unit, so `t` is a world distance.
 */
export function closestAxisT(axisOrigin: Vec3, axisDir: Vec3, ray: Ray): number {
    const w0 = sub(axisOrigin, ray.origin);
    const b = dot(axisDir, ray.dir);
    const d = dot(axisDir, w0);
    const e = dot(ray.dir, w0);
    // a = dot(axisDir,axisDir) = 1, c = dot(ray.dir,ray.dir) = 1 (both unit), so denom = 1 - b²
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-9) return Number.NaN;
    return (b * e - d) / denom;
}

/**
 * the signed world distance to translate along a unit `axisDir` so the grabbed point tracks the cursor:
 * the change in nearest-point parameter between the drag-start ray and the current ray. 0 when either ray
 * is parallel to the axis. The caller writes `startPos + axisDrag(...)·axisDir`.
 */
export function axisDrag(axisOrigin: Vec3, axisDir: Vec3, start: Ray, now: Ray): number {
    const t0 = closestAxisT(axisOrigin, axisDir, start);
    const t1 = closestAxisT(axisOrigin, axisDir, now);
    if (!Number.isFinite(t0) || !Number.isFinite(t1)) return 0;
    return t1 - t0;
}

/** where a ray meets the plane through `p` with unit `normal`, or null when it's parallel to the plane. */
export function rayPlane(p: Vec3, normal: Vec3, ray: Ray): Vec3 | null {
    const denom = dot(ray.dir, normal);
    if (Math.abs(denom) < 1e-9) return null;
    const t = dot(sub(p, ray.origin), normal) / denom;
    return add(ray.origin, [ray.dir[0] * t, ray.dir[1] * t, ray.dir[2] * t]);
}

/** the in-plane translation as the cursor drags across the plane through `anchor` with unit `normal` — the
 * difference of the two rays' plane intersections. Zero when a ray is parallel to the plane. */
export function planeDrag(anchor: Vec3, normal: Vec3, start: Ray, now: Ray): Vec3 {
    const a = rayPlane(anchor, normal, start);
    const b = rayPlane(anchor, normal, now);
    if (!a || !b) return [0, 0, 0];
    return sub(b, a);
}

/** the signed angle (radians) the cursor sweeps about `axis` on the ring through `anchor`: the angle from
 * the start ray's plane-hit to the now ray's, measured in the ring plane. 0 when a ray is parallel. */
export function ringAngle(anchor: Vec3, axis: Vec3, start: Ray, now: Ray): number {
    const a = rayPlane(anchor, axis, start);
    const b = rayPlane(anchor, axis, now);
    if (!a || !b) return 0;
    const va = sub(a, anchor);
    const vb = sub(b, anchor);
    return Math.atan2(dot(cross(va, vb), axis), dot(va, vb));
}

function mul(a: Vec3, s: number): Vec3 {
    return [a[0] * s, a[1] * s, a[2] * s];
}

/**
 * the gizmo's world-space size so it projects to ~{@link GIZMO_PX} screen px — constant on-screen size, the
 * standard transform-gizmo behavior (three.js / PlayCanvas scale the gizmo by camera distance, then project
 * real 3D geometry). Handles built at this world size and projected as true world points stay stable and
 * correctly foreshortened at every angle — a screen-space extent collapses when an axis turns edge-on.
 * `dist` is the camera→origin distance (ignored for orthographic); `heightPx` the viewport height in the
 * same px space the projection maps to.
 */
export function gizmoScale(
    perspective: boolean,
    fovDegOrSize: number,
    dist: number,
    heightPx: number,
): number {
    const half = perspective ? dist * Math.tan((fovDegOrSize * Math.PI) / 360) : fovDegOrSize;
    return (GIZMO_PX * 2 * half) / heightPx;
}

/** what a handle constrains the drag to. `trackball` is the screen-facing rotation ring — a roll about
 * the view axis, the outer ring three.js / Unity draw. `free` is the interior disc inside that ring —
 * grabbed anywhere off a ring, it does free arcball rotation (two-axis tumble), three.js's XYZE handle. */
export type HandleKind = "axis" | "plane" | "ring" | "uniform" | "trackball" | "free";

/** one handle a manipulator exposes. `kind` selects the pick + drag math; `axis` binds it to a frame axis
 * (0/1/2 = X/Y/Z) for axis/ring, or to the plane's *normal* axis for plane (0 = the YZ plane); screen and
 * uniform ignore it (-1). `id` is the stable key the render, `Handles.active`/`hover`, and the drag driver
 * all speak — its value decodes back to `(kind, axis)` ({@link decodeHandle}). */
export interface Handle {
    id: number;
    kind: HandleKind;
    axis: number;
}

// id bases — axis ids stay 0/1/2 so today's `Handles.active` axis values are unchanged
const RING_BASE = 6;
const UNIFORM_ID = 10;
const SCREEN_RING_ID = 11;
const FREE_ID = 12;

/** the `(kind, axis)` a {@link Handle} id encodes — the one decode of the id scheme, shared by the drag
 * driver and the render's per-axis color. `axis` is -1 for the uniform/trackball/free handles. */
export function decodeHandle(id: number): { kind: HandleKind; axis: number } {
    if (id < 3) return { kind: "axis", axis: id };
    if (id < 6) return { kind: "plane", axis: id - 3 };
    if (id < 9) return { kind: "ring", axis: id - RING_BASE };
    if (id === UNIFORM_ID) return { kind: "uniform", axis: -1 };
    if (id === SCREEN_RING_ID) return { kind: "trackball", axis: -1 };
    return { kind: "free", axis: -1 };
}

function axisHandles(): Handle[] {
    return [0, 1, 2].map((axis) => ({ id: axis, kind: "axis", axis }));
}
function planeHandles(): Handle[] {
    return [0, 1, 2].map((axis) => ({ id: axis + 3, kind: "plane", axis }));
}
function ringHandles(): Handle[] {
    return [0, 1, 2].map((axis) => ({ id: axis + RING_BASE, kind: "ring", axis }));
}
const UNIFORM_HANDLE: Handle = { id: UNIFORM_ID, kind: "uniform", axis: -1 };
const SCREEN_RING_HANDLE: Handle = { id: SCREEN_RING_ID, kind: "trackball", axis: -1 };
const FREE_HANDLE: Handle = { id: FREE_ID, kind: "free", axis: -1 };

/**
 * tagged screen-space geometry for one handle, projected once per frame and consumed by BOTH the renderer
 * and the hit-test — so what's drawn is exactly what's grabbable, for every kind. Coords are CSS px. `id`
 * ties it back to its {@link Handle}.
 */
export type Glyph =
    | {
          id: number;
          kind: "axis";
          ox: number;
          oy: number;
          ex: number;
          ey: number;
          cap: "none" | "arrow" | "box";
      }
    | { id: number; kind: "quad"; pts: number[] } // 4 corners flat [x0,y0,…x3,y3]
    | { id: number; kind: "ring"; pts: number[] } // projected circle polyline, flat [x,y,…]
    | { id: number; kind: "point"; cx: number; cy: number }
    | { id: number; kind: "disc"; cx: number; cy: number; r: number }; // filled circle (free-rotate fill)

const PLANE_IN = 0; // plane-quad inner corner at the origin — its inner edges lie on the axes (flush)
const PLANE_OUT = 0.2; // plane-quad outer corner, fraction of the axis length
/** a plane handle below this |normal·eye| is dropped; the render fades it to nothing as it approaches, so
 * the drop lands when it's already invisible (no pop). Shared so draw + pick agree. */
export const PLANE_EDGE_MIN = 0.08;
export const PLANE_EDGE_FADE = 0.32; // |normal·eye| at which the plane is at full opacity
/** rotation-ring radius as a fraction of the gizmo size — rings sit inside the axis shafts. */
export const RING_SCALE = 0.78;
/** the screen-facing trackball ring's radius as a fraction of the gizmo size — the outer ring, sitting
 * just past the axis rings (three.js / Unity convention) so its rim never overlaps them on the pick. */
export const SCREEN_RING_SCALE = 0.9;
const RING_SEGS = 48;

// a ring's world-space points: a closed polyline of RING_SEGS+1 points around `origin` in the plane
// perpendicular to unit `normal`, radius `r`. Shared by the axis rings (normal = a frame axis) and the
// trackball (normal = the view dir), so draw + pick + the two consumers can't drift.
function ringPoints(origin: Vec3, normal: Vec3, r: number): Vec3[] {
    const u = norm(cross(normal, Math.abs(normal[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0]));
    const v = norm(cross(normal, u));
    const pts: Vec3[] = [];
    for (let i = 0; i <= RING_SEGS; i++) {
        const t = (i / RING_SEGS) * Math.PI * 2;
        pts.push(add(origin, add(mul(u, r * Math.cos(t)), mul(v, r * Math.sin(t)))));
    }
    return pts;
}

// a ring handle's normal (the view dir for the trackball, else its frame axis) + radius fraction
function ringFrame(
    h: Handle,
    axes: readonly [Vec3, Vec3, Vec3],
    eye: Vec3,
): { normal: Vec3; scale: number } {
    return h.kind === "trackball"
        ? { normal: eye, scale: SCREEN_RING_SCALE }
        : { normal: axes[h.axis], scale: RING_SCALE };
}

// project a world point, or null when it's off the near side of the camera
function pt(
    viewProj: ArrayLike<number>,
    p: Vec3,
    width: number,
    height: number,
): { x: number; y: number } | null {
    const s = project(viewProj, p, width, height);
    return s && !s.behind ? { x: s.x, y: s.y } : null;
}

/**
 * the screen geometry for each handle in `set`, or null where it can't be drawn (origin or an endpoint off
 * the near side, or a plane edge-on). The handles are built in WORLD space at `scale` world units (see
 * {@link gizmoScale}) and their true world points projected — so they read as stable 3D and foreshorten
 * correctly. `axes` is the active frame (world or local), `eye` the camera-forward world dir (plane cull),
 * `cap` the axis tip glyph (arrow for translate, box for scale). Render and hit-test call this with the
 * same args — the invariant that keeps drawn and grabbable identical.
 */
export function glyphs(
    set: readonly Handle[],
    origin: Vec3,
    axes: readonly [Vec3, Vec3, Vec3],
    viewProj: ArrayLike<number>,
    width: number,
    height: number,
    scale: number,
    eye: Vec3,
    cap: "arrow" | "box" = "arrow",
): (Glyph | null)[] {
    const o = pt(viewProj, origin, width, height);
    if (!o) return set.map(() => null);
    return set.map((h): Glyph | null => {
        if (h.kind === "axis") {
            const tip = pt(viewProj, add(origin, mul(axes[h.axis], scale)), width, height);
            return tip && { id: h.id, kind: "axis", ox: o.x, oy: o.y, ex: tip.x, ey: tip.y, cap };
        }
        if (h.kind === "plane") {
            // drop a near-edge-on plane (the render has already faded it to nothing by here, so no pop)
            if (Math.abs(dot(axes[h.axis], eye)) < PLANE_EDGE_MIN) return null;
            const u = axes[(h.axis + 1) % 3];
            const v = axes[(h.axis + 2) % 3];
            const corner = (a: number, b: number) =>
                pt(viewProj, add(origin, add(mul(u, a * scale), mul(v, b * scale))), width, height);
            const c0 = corner(PLANE_IN, PLANE_IN);
            const c1 = corner(PLANE_OUT, PLANE_IN);
            const c2 = corner(PLANE_OUT, PLANE_OUT);
            const c3 = corner(PLANE_IN, PLANE_OUT);
            if (!c0 || !c1 || !c2 || !c3) return null;
            return {
                id: h.id,
                kind: "quad",
                pts: [c0.x, c0.y, c1.x, c1.y, c2.x, c2.y, c3.x, c3.y],
            };
        }
        if (h.kind === "ring" || h.kind === "trackball") {
            const { normal, scale: frac } = ringFrame(h, axes, eye);
            const pts: number[] = [];
            for (const wp of ringPoints(origin, normal, scale * frac)) {
                const p = pt(viewProj, wp, width, height);
                if (p) pts.push(p.x, p.y);
            }
            return pts.length >= 4 ? { id: h.id, kind: "ring", pts } : null;
        }
        if (h.kind === "free") {
            // the interior disc, the same radius as the trackball ring (its rim) — a rim point projected
            // and measured from the centre gives the screen radius, so the fill exactly bounds the ring
            const rim = pt(
                viewProj,
                ringPoints(origin, eye, scale * SCREEN_RING_SCALE)[0],
                width,
                height,
            );
            return (
                rim && {
                    id: h.id,
                    kind: "disc",
                    cx: o.x,
                    cy: o.y,
                    r: Math.hypot(rim.x - o.x, rim.y - o.y),
                }
            );
        }
        // uniform — the projected origin as a center point
        return { id: h.id, kind: "point", cx: o.x, cy: o.y };
    });
}

/** one world-space line of a handle, tagged with its {@link Handle} id (for coloring). */
export interface HandleLine {
    id: number;
    a: Vec3;
    b: Vec3;
}

/**
 * the world-space line segments of each handle in `set` — axis shafts, ring polylines, and plane-square
 * edges — for the lines renderer (which near-plane-clips + constant-pixel-widths them in its VS, the proven
 * path; `extras/lines`). The solid bits (arrowheads, scale caps, centre, plane fill) are drawn separately
 * at projected points. `eye` culls edge-on planes, matching {@link glyphs} so draw == pick.
 */
export function handleSegments(
    set: readonly Handle[],
    origin: Vec3,
    axes: readonly [Vec3, Vec3, Vec3],
    scale: number,
    eye: Vec3,
): HandleLine[] {
    const out: HandleLine[] = [];
    for (const h of set) {
        if (h.kind === "axis") {
            out.push({ id: h.id, a: origin, b: add(origin, mul(axes[h.axis], scale)) });
        } else if (h.kind === "plane") {
            if (Math.abs(dot(axes[h.axis], eye)) < PLANE_EDGE_MIN) continue;
            const u = axes[(h.axis + 1) % 3];
            const v = axes[(h.axis + 2) % 3];
            const c = (a: number, b: number) =>
                add(origin, add(mul(u, a * scale), mul(v, b * scale)));
            // PLANE_IN = 0, so the inner two edges (c0→c1, c0→c3) lie on the axis shafts already drawn —
            // emit only the two OUTER edges so the square reads flush with the axis lines
            const c1 = c(PLANE_OUT, PLANE_IN);
            const c2 = c(PLANE_OUT, PLANE_OUT);
            const c3 = c(PLANE_IN, PLANE_OUT);
            out.push({ id: h.id, a: c1, b: c2 }, { id: h.id, a: c2, b: c3 });
        } else if (h.kind === "ring" || h.kind === "trackball") {
            const { normal, scale: frac } = ringFrame(h, axes, eye);
            const wp = ringPoints(origin, normal, scale * frac);
            for (let i = 1; i < wp.length; i++) out.push({ id: h.id, a: wp[i - 1], b: wp[i] });
        }
        // screen / uniform: no lines — drawn as a solid centre marker
    }
    return out;
}

const POINT_PX = 9; // screen/uniform centre-cube hit radius (matches the drawn cube extent)

// the cursor's distance to a glyph in CSS px (Infinity for a miss), the per-kind hit-test
function glyphDist(cx: number, cy: number, g: Glyph): number {
    if (g.kind === "axis") return segDist(cx, cy, g.ox, g.oy, g.ex, g.ey);
    if (g.kind === "point")
        return Math.hypot(cx - g.cx, cy - g.cy) <= POINT_PX ? 0 : Number.POSITIVE_INFINITY;
    if (g.kind === "quad") return pointInQuad(cx, cy, g.pts) ? 0 : Number.POSITIVE_INFINITY;
    // a disc (free-rotate fill) isn't distance-picked — pickHandles handles it as a fallback
    if (g.kind === "disc") return Number.POSITIVE_INFINITY;
    // ring: nearest projected polyline segment
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i + 3 < g.pts.length; i += 2) {
        const d = segDist(cx, cy, g.pts[i], g.pts[i + 1], g.pts[i + 2], g.pts[i + 3]);
        if (d < best) best = d;
    }
    return best;
}

/**
 * the handle id under the cursor, or -1. Builds the glyphs once and returns the nearest within `threshPx`;
 * `set` is ordered planes/screen before axes so an overlap near the origin resolves to the plane (first
 * wins on equal distance).
 */
export function pickHandles(
    set: readonly Handle[],
    cursor: { x: number; y: number },
    origin: Vec3,
    axes: readonly [Vec3, Vec3, Vec3],
    viewProj: ArrayLike<number>,
    width: number,
    height: number,
    eye: Vec3,
    scale: number,
    threshPx = 8,
    cap: "arrow" | "box" = "arrow",
): number {
    const gs = glyphs(set, origin, axes, viewProj, width, height, scale, eye, cap);
    let best = -1;
    let bestDist = threshPx;
    let fallback = -1; // a disc (free-rotate fill) is a fallback: it wins only when no handle is hit
    for (const g of gs) {
        if (!g) continue;
        if (g.kind === "disc") {
            if (Math.hypot(cursor.x - g.cx, cursor.y - g.cy) <= g.r) fallback = g.id;
            continue;
        }
        const d = glyphDist(cursor.x, cursor.y, g);
        if (d < bestDist) {
            bestDist = d;
            best = g.id;
        }
    }
    return best >= 0 ? best : fallback;
}

/** an entity's transform as a {@link Manipulator} reads and writes it — position, rotation quat, scale. */
export interface Pose {
    pos: Vec3;
    rot: Quat;
    scale: Vec3;
}

/**
 * a viewport transform tool: its handle set ({@link Manipulator.handles}), how it hit-tests
 * ({@link Manipulator.pick}), and how a drag transforms an entity's pose ({@link Manipulator.drag}). The
 * interaction driver and the handle render are tool-agnostic — both read a Manipulator over its declared
 * handles — so a new tool is one more concretization, no driver change. `axes` is the active frame (world
 * or local); `anchor` its origin; `eye` the camera-forward world dir; `snap` quantizes the drag.
 *
 * A new tool reuses the existing handle kinds — adding a kind is the rare case that also touches `glyphs`,
 * the hit-test, and the render. Define the manipulator beside Move/Scale/Rotate, then wire three sites:
 * the `Tool` enum + `TOOLS` (`tool.ts`) and a `manipulatorFor` case. `manipulatorFor` is a closed switch
 * by design (no registry) — tools are engine-owned, not consumer-extended.
 *
 * @example
 * // a composite move+rotate: rings rotate, arrows translate. Both kinds already exist, so pick is the
 * // shared hit-test and drag delegates to the built-in math by handle kind.
 * export const Transform: Manipulator = {
 *     handles: [...ringHandles(), ...axisHandles()],
 *     pick(cursor, origin, axes, viewProj, w, h, scale, threshPx, eye = [0, 0, 1]) {
 *         return pickHandles(Transform.handles, cursor, origin, axes, viewProj, w, h, eye, scale, threshPx);
 *     },
 *     drag(handle, anchor, axes, pose, start, now, eye = [0, 0, 1], snap = false, scale = 1) {
 *         const driver = decodeHandle(handle).kind === "ring" ? Rotate : Move;
 *         return driver.drag(handle, anchor, axes, pose, start, now, eye, snap, scale);
 *     },
 * };
 */
export interface Manipulator {
    handles: readonly Handle[];
    /** which handle the cursor is over (a {@link Handle} id), or -1 for none. */
    pick(
        cursor: { x: number; y: number },
        origin: Vec3,
        axes: readonly [Vec3, Vec3, Vec3],
        viewProj: ArrayLike<number>,
        width: number,
        height: number,
        scale: number,
        threshPx?: number,
        eye?: Vec3,
    ): number;
    /** the entity's new pose after dragging `handle` from the `start` ray to the `now` ray, given its
     * captured start `pose`. `scale` is the gizmo's on-screen world size ({@link gizmoScale}) — the
     * uniform-scale handle references it so its sensitivity is zoom-independent; other handles ignore it. */
    drag(
        handle: number,
        anchor: Vec3,
        axes: readonly [Vec3, Vec3, Vec3],
        pose: Pose,
        start: Ray,
        now: Ray,
        eye?: Vec3,
        snap?: boolean,
        scale?: number,
    ): Pose;
}

// All drags measure their geometry from `anchor` (the gizmo origin — the selection centroid for a
// multi-entity grab) and apply the result to each entity's `pose`. For a lone entity anchor === pose.pos,
// so the pivot terms vanish; for a group they keep the transform rigid (entities orbit / scale about the
// shared anchor). `pose.pos − anchor` is the entity's offset from the pivot.

const SCALE_MIN_T = 1e-3;

// translate along an axis: the grabbed point tracks the cursor
function axisMove(anchor: Vec3, axis: Vec3, pose: Pose, start: Ray, now: Ray, snap: boolean): Pose {
    let d = axisDrag(anchor, axis, start, now);
    if (snap) d = snapTo(d, GRID_SNAP);
    return { ...pose, pos: add(pose.pos, [d * axis[0], d * axis[1], d * axis[2]]) };
}

// translate across a plane: the grabbed point tracks the cursor in two axes
function planeMove(
    anchor: Vec3,
    normal: Vec3,
    axes: readonly [Vec3, Vec3, Vec3],
    plane: number,
    pose: Pose,
    start: Ray,
    now: Ray,
    snap: boolean,
): Pose {
    let d = planeDrag(anchor, normal, start, now);
    if (snap) {
        // snap each in-plane component independently (the off-plane axis is already zero)
        for (let a = 0; a < 3; a++) {
            if (a === plane) continue;
            const ax = axes[a];
            const comp = snapTo(dot(d, ax), GRID_SNAP) - dot(d, ax);
            d = add(d, [comp * ax[0], comp * ax[1], comp * ax[2]]);
        }
    }
    return { ...pose, pos: add(pose.pos, d) };
}

// scale one axis by the cursor's distance ratio (now / grab) from the anchor; the entity's offset along
// that axis scales with it (pivot about the anchor)
// scale lane `handle` by the cursor's distance ratio along the local axis `dir` (= the entity's local axis
// `handle`). Scale is always local (the App passes the local frame): `Transform.scale` is a per-axis LOCAL
// diagonal, and a per-axis WORLD scale of a rotated object needs shear that TRS can't hold — so, like
// three.js ("scale always oriented to local rotation") and Unity, the scale gizmo is local-only.
function axisScale(
    anchor: Vec3,
    dir: Vec3,
    handle: number,
    pose: Pose,
    start: Ray,
    now: Ray,
    snap: boolean,
): Pose {
    const t0 = closestAxisT(anchor, dir, start);
    const t1 = closestAxisT(anchor, dir, now);
    if (!Number.isFinite(t0) || !Number.isFinite(t1) || Math.abs(t0) < SCALE_MIN_T) return pose;
    let f = t1 / t0;
    if (snap) f = snapTo(f, SCALE_SNAP) || SCALE_SNAP;
    const scale: [number, number, number] = [pose.scale[0], pose.scale[1], pose.scale[2]];
    scale[handle] *= f;
    const k = dot(sub(pose.pos, anchor), dir) * (f - 1);
    return { ...pose, scale, pos: add(pose.pos, [k * dir[0], k * dir[1], k * dir[2]]) };
}

/** translate along an axis ({@link axisDrag}); rotation + scale ride through. Handle set: three axes
 * (arrow caps) + three planes. No centre handle — like Unity / Godot, Move has no free-translate dot (the
 * planes cover two-axis moves); only Scale keeps a centre handle. */
export const Move: Manipulator = {
    handles: [...planeHandles(), ...axisHandles()],
    pick(cursor, origin, axes, viewProj, width, height, scale, threshPx, eye = [0, 0, 1]) {
        return pickHandles(
            Move.handles,
            cursor,
            origin,
            axes,
            viewProj,
            width,
            height,
            eye,
            scale,
            threshPx,
        );
    },
    drag(handle, anchor, axes, pose, start, now, _eye, snap = false) {
        const { kind, axis } = decodeHandle(handle);
        if (kind === "axis") return axisMove(anchor, axes[axis], pose, start, now, snap);
        return planeMove(anchor, axes[axis], axes, axis, pose, start, now, snap);
    },
};

/** scale along an axis by the cursor's distance ratio (no tuned sensitivity — the ratio is the factor),
 * plus a uniform-center handle that scales all three axes together. */
export const Scale: Manipulator = {
    handles: [UNIFORM_HANDLE, ...axisHandles()],
    pick(cursor, origin, axes, viewProj, width, height, scale, threshPx, eye = [0, 0, 1]) {
        return pickHandles(
            Scale.handles,
            cursor,
            origin,
            axes,
            viewProj,
            width,
            height,
            eye,
            scale,
            threshPx,
            "box",
        );
    },
    drag(handle, anchor, axes, pose, start, now, eye = [0, 0, 1], snap = false, scale = 1) {
        const { kind, axis } = decodeHandle(handle);
        if (kind === "axis") return axisScale(anchor, axes[axis], axis, pose, start, now, snap);
        // uniform: the centre handle is grabbed AT the anchor, so a radial distance *ratio* would divide by
        // ~0 and explode. Drive the factor by the cursor's drag across the camera plane (`delta`), projected
        // onto the screen up-right diagonal — the PlayCanvas scale-gizmo model (`up + right`): a SIGNED
        // scalar, so up-right grows and down-left shrinks (the standard, and what makes scale-down work). It
        // is measured against the gizmo's own on-screen world size (`scale`) so the feel is zoom-independent
        // (one gizmo-length of diagonal drag doubles); `scale` also sidesteps the reverse-Z cursor ray, whose
        // far-plane origin makes a camera-distance read ~200× too large (why this once felt dead).
        const a = rayPlane(anchor, eye, start);
        const b = rayPlane(anchor, eye, now);
        if (!a || !b) return pose;
        const { up, right } = camBasis(eye);
        const grow = dot(sub(b, a), norm(add(up, right))); // signed: + up-right (grow), − down-left (shrink)
        let f = 1 + grow / Math.max(scale, SCALE_MIN_T);
        if (snap) f = snapTo(f, SCALE_SNAP) || SCALE_SNAP;
        f = Math.max(f, 0.01); // never flip to zero / negative
        const off = sub(pose.pos, anchor);
        return {
            ...pose,
            scale: [pose.scale[0] * f, pose.scale[1] * f, pose.scale[2] * f],
            pos: add(anchor, [off[0] * f, off[1] * f, off[2] * f]),
        };
    },
};

interface CamBasis {
    right: Vec3;
    up: Vec3;
    toward: Vec3;
}

// the camera's screen basis from the toward-camera dir `eye`. Under reverse-Z `cursorRay`'s dir points
// back at the camera, so `eye` is toward-camera — the hemisphere must bulge that way, or the tumble mirrors.
function camBasis(eye: Vec3): CamBasis {
    const toward = norm(eye);
    const ref: Vec3 = Math.abs(toward[1]) > 0.99 ? [1, 0, 0] : [0, 1, 0];
    const right = norm(cross(ref, toward));
    return { right, up: norm(cross(toward, right)), toward };
}

// the unit vector from the arcball centre to the sphere point under `ray`: project the cursor onto the
// camera plane through the centre, normalize by the radius, and lift onto the camera-facing hemisphere
// (Shoemake's mapping). A cursor past the rim clamps to the silhouette (z = 0). Mapping through the camera
// plane (not a ray-sphere hit) keeps it correct whatever the ray's orientation, which reverse-Z flips.
function arcballHit(center: Vec3, r: number, ray: Ray, basis: CamBasis): Vec3 {
    const hit = rayPlane(center, basis.toward, ray);
    if (!hit) return basis.toward;
    const d = sub(hit, center);
    const x = dot(d, basis.right) / r;
    const y = dot(d, basis.up) / r;
    const r2 = x * x + y * y;
    if (r2 <= 1) {
        const z = Math.sqrt(1 - r2);
        return add(add(mul(basis.right, x), mul(basis.up, y)), mul(basis.toward, z));
    }
    const s = 1 / Math.sqrt(r2);
    return add(mul(basis.right, x * s), mul(basis.up, y * s));
}

// the free arcball rotation: the quaternion taking the sphere point under the start ray to the one under
// the now ray (axis = their cross, angle = the angle between them) — a two-axis tumble, three.js's XYZE.
function arcball(center: Vec3, r: number, eye: Vec3, start: Ray, now: Ray, snap: boolean): Quat {
    const basis = camBasis(eye);
    const a = arcballHit(center, r, start, basis);
    const b = arcballHit(center, r, now, basis);
    const axis = cross(a, b);
    const len = Math.hypot(axis[0], axis[1], axis[2]);
    if (len < 1e-9) return [0, 0, 0, 1]; // aligned (no rotation) — identity, never NaN
    let angle = Math.atan2(len, dot(a, b));
    if (snap) angle = snapTo(angle, ANGLE_SNAP);
    return axisAngle([axis[0] / len, axis[1] / len, axis[2] / len], angle);
}

/** rotate about a ring: the cursor sweeps an angle in the ring plane ({@link ringAngle}), composed onto
 * the pose's orientation; the entity orbits the anchor (in place for a lone selection). The handle set is
 * the three axis rings, the screen-facing trackball ring (a roll about the view dir `eye`, the outer ring
 * three.js / Unity draw — same in-screen-plane sweep, no special drag), and the interior `free` disc, which
 * grabs anywhere off a ring and does free arcball rotation ({@link arcball}) over a sphere of the gizmo's
 * radius (`scale`). */
export const Rotate: Manipulator = {
    handles: [SCREEN_RING_HANDLE, ...ringHandles(), FREE_HANDLE],
    pick(cursor, origin, axes, viewProj, width, height, scale, threshPx, eye = [0, 0, 1]) {
        return pickHandles(
            Rotate.handles,
            cursor,
            origin,
            axes,
            viewProj,
            width,
            height,
            eye,
            scale,
            threshPx,
        );
    },
    drag(handle, anchor, axes, pose, start, now, eye = [0, 0, 1], snap = false, scale = 1) {
        const { kind, axis } = decodeHandle(handle);
        let delta: Quat;
        if (kind === "free") {
            delta = arcball(anchor, scale * SCREEN_RING_SCALE, eye, start, now, snap);
        } else {
            // the trackball rolls about the view dir; an axis ring about its frame axis
            const dir = kind === "trackball" ? eye : axes[axis];
            let a = ringAngle(anchor, dir, start, now);
            if (snap) a = snapTo(a, ANGLE_SNAP);
            delta = axisAngle(dir, a);
        }
        return {
            ...pose,
            rot: qnorm(qmul(delta, pose.rot)),
            pos: add(anchor, qrotvec(delta, sub(pose.pos, anchor))),
        };
    },
};

/** the {@link Manipulator} a tool drives, or null for Select (pick only). */
export function manipulatorFor(tool: Tool): Manipulator | null {
    switch (tool) {
        case Tool.Move:
            return Move;
        case Tool.Rotate:
            return Rotate;
        case Tool.Scale:
            return Scale;
        default:
            return null;
    }
}
