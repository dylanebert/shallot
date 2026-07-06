// CPU raycast (roadmap §6.5) — analytic ray-vs-shape tests + a nearest-hit query over a body list, plus a
// screen-cursor → world ray for picking (`screenToRay`/`generateRay`). The shared pick primitive for player
// grab + god-mode pick/drag + acoustics. CPU over poses the caller passes in (Mirror'd from
// the GPU `bodies` for live bodies, the authored slab for statics) — the right call for grab's low volume
// + latency tolerance, vs a GPU LBVH traverse (roadmap §6.5 experiment). Ported from the legacy
// `raycast.ts`; gold-tested closed-form (`raycast.test.ts`). No GJK — each shape is a closed-form solve.

import { ShapeKind } from "./index";

/** a world-space ray. `dir` MUST be normalized; the returned `distance` is then world units along it. */
export interface Ray {
    origin: readonly [number, number, number];
    dir: readonly [number, number, number];
}

/** one candidate body for {@link raycast}: its world pose + collider shape (the fields the analytic tests
 * read). `half` is the box half-extents (or a hull's AABB half); `radius` the sphere/capsule rounding
 * (`Body.halfExtents.w`); `half[1]` the capsule core half-height. Build it from a live pose + the Body slab. */
export interface RayBody {
    eid: number;
    shape: number;
    pos: readonly [number, number, number];
    quat: readonly [number, number, number, number];
    half: readonly [number, number, number];
    radius: number;
}

/** a ray hit: the body, the world distance along the ray, and the world hit point + surface normal. */
export interface RayHit {
    eid: number;
    distance: number;
    point: [number, number, number];
    normal: [number, number, number];
}

interface ShapeHit {
    t: number;
    nx: number;
    ny: number;
    nz: number;
}

/** rotate a vector by a quaternion (q · v). Pass the conjugate (`-qx, -qy, -qz, qw`) for the inverse
 * rotation, world → body-local. */
export function qRotate(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    vx: number,
    vy: number,
    vz: number,
): [number, number, number] {
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ];
}

/** ray vs a sphere of `radius` centred at `c`. Nearest non-negative root; normal points outward. */
export function raySphere(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
): ShapeHit | null {
    const lx = ox - cx;
    const ly = oy - cy;
    const lz = oz - cz;
    const a = dx * dx + dy * dy + dz * dz;
    const b = 2 * (lx * dx + ly * dy + lz * dz);
    const c = lx * lx + ly * ly + lz * lz - radius * radius;
    const disc = b * b - 4 * a * c;
    if (disc < 0) return null;
    const sd = Math.sqrt(disc);
    let t = (-b - sd) / (2 * a);
    if (t < 0) {
        t = (-b + sd) / (2 * a);
        if (t < 0) return null;
    }
    const px = ox + dx * t - cx;
    const py = oy + dy * t - cy;
    const pz = oz + dz * t - cz;
    const len = Math.sqrt(px * px + py * py + pz * pz) || 1;
    return { t, nx: px / len, ny: py / len, nz: pz / len };
}

/** ray vs an oriented box (centre `c`, half-extents `h`, orientation `q`). Slab test in the box's local
 * frame; the entry-face axis gives the normal, rotated back to world. */
export function rayOBB(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
): ShapeHit | null {
    const [rox, roy, roz] = qRotate(-qx, -qy, -qz, qw, ox - cx, oy - cy, oz - cz);
    const [rdx, rdy, rdz] = qRotate(-qx, -qy, -qz, qw, dx, dy, dz);
    let tmin = -Infinity;
    let tmax = Infinity;
    let axis = 0;
    let sign = 1;
    for (let i = 0; i < 3; i++) {
        const o = i === 0 ? rox : i === 1 ? roy : roz;
        const d = i === 0 ? rdx : i === 1 ? rdy : rdz;
        const h = i === 0 ? hx : i === 1 ? hy : hz;
        if (Math.abs(d) < 1e-12) {
            if (o < -h || o > h) return null;
        } else {
            const inv = 1 / d;
            let t1 = (-h - o) * inv;
            let t2 = (h - o) * inv;
            let s = -1;
            if (t1 > t2) {
                const tmp = t1;
                t1 = t2;
                t2 = tmp;
                s = 1;
            }
            if (t1 > tmin) {
                tmin = t1;
                axis = i;
                sign = s;
            }
            if (t2 < tmax) tmax = t2;
            if (tmin > tmax) return null;
        }
    }
    if (tmax < 0) return null;
    const t = tmin >= 0 ? tmin : tmax;
    if (t < 0) return null;
    const [nx, ny, nz] = qRotate(
        qx,
        qy,
        qz,
        qw,
        axis === 0 ? sign : 0,
        axis === 1 ? sign : 0,
        axis === 2 ? sign : 0,
    );
    return { t, nx, ny, nz };
}

/** ray vs a capsule (core segment along local Y, `halfHeight` each way, inflated by `radius`). Tests the
 * cylinder body + the two hemisphere caps in the capsule's local frame, keeps the nearest. */
export function rayCapsule(
    ox: number,
    oy: number,
    oz: number,
    dx: number,
    dy: number,
    dz: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
    halfHeight: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
): ShapeHit | null {
    const [rox, roy, roz] = qRotate(-qx, -qy, -qz, qw, ox - cx, oy - cy, oz - cz);
    const [rdx, rdy, rdz] = qRotate(-qx, -qy, -qz, qw, dx, dy, dz);
    let bestT = Infinity;
    let bnx = 0;
    let bny = 0;
    let bnz = 0;

    // infinite cylinder along Y, clamped to the core segment
    const a = rdx * rdx + rdz * rdz;
    const b = 2 * (rox * rdx + roz * rdz);
    const c = rox * rox + roz * roz - radius * radius;
    const disc = b * b - 4 * a * c;
    if (a > 1e-12 && disc >= 0) {
        const sd = Math.sqrt(disc);
        for (const s of [-1, 1]) {
            const t = (-b + s * sd) / (2 * a);
            if (t >= 0 && t < bestT) {
                const hy = roy + rdy * t;
                if (hy >= -halfHeight && hy <= halfHeight) {
                    const hx = rox + rdx * t;
                    const hz = roz + rdz * t;
                    const len = Math.sqrt(hx * hx + hz * hz);
                    if (len > 1e-12) {
                        bestT = t;
                        bnx = hx / len;
                        bny = 0;
                        bnz = hz / len;
                    }
                }
            }
        }
    }

    // hemisphere caps at ±halfHeight
    for (const cap of [-1, 1]) {
        const capY = cap * halfHeight;
        const sly = roy - capY;
        const sa = rdx * rdx + rdy * rdy + rdz * rdz;
        const sb = 2 * (rox * rdx + sly * rdy + roz * rdz);
        const sc = rox * rox + sly * sly + roz * roz - radius * radius;
        const sd2 = sb * sb - 4 * sa * sc;
        if (sd2 < 0) continue;
        const sd = Math.sqrt(sd2);
        for (const s of [-1, 1]) {
            const t = (-sb + s * sd) / (2 * sa);
            if (t >= 0 && t < bestT) {
                const hy = roy + rdy * t - capY;
                if (cap * hy >= 0) {
                    const hx = rox + rdx * t;
                    const hz = roz + rdz * t;
                    const len = Math.sqrt(hx * hx + hy * hy + hz * hz);
                    if (len > 1e-12) {
                        bestT = t;
                        bnx = hx / len;
                        bny = hy / len;
                        bnz = hz / len;
                    }
                }
            }
        }
    }

    if (bestT === Infinity) return null;
    const [nx, ny, nz] = qRotate(qx, qy, qz, qw, bnx, bny, bnz);
    return { t: bestT, nx, ny, nz };
}

function hitShape(ray: Ray, b: RayBody): ShapeHit | null {
    const [ox, oy, oz] = ray.origin;
    const [dx, dy, dz] = ray.dir;
    const [px, py, pz] = b.pos;
    const [qx, qy, qz, qw] = b.quat;
    if (b.shape === ShapeKind.Sphere) {
        return raySphere(ox, oy, oz, dx, dy, dz, px, py, pz, b.radius);
    }
    if (b.shape === ShapeKind.Capsule) {
        return rayCapsule(ox, oy, oz, dx, dy, dz, px, py, pz, b.radius, b.half[1], qx, qy, qz, qw);
    }
    // Box, and Hull approximated by its AABB box (exact hull-mesh raycast isn't needed for grab/pick)
    return rayOBB(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        px,
        py,
        pz,
        b.half[0],
        b.half[1],
        b.half[2],
        qx,
        qy,
        qz,
        qw,
    );
}

const DEG2RAD = Math.PI / 180;

/**
 * a world-space pick ray through a normalized-device-coordinate point (`ndcX`/`ndcY` in [-1, 1], x right /
 * y up; (0, 0) is screen centre). Unprojects through the camera's vertical `fov` (degrees) + `aspect`,
 * rotates the camera-space ray into world by the camera `quat`, and offsets the origin to the `near` plane.
 * The returned `dir` is normalized, so a {@link RayHit} distance is world units. Pair with {@link screenToRay}
 * for pixel input.
 */
export function generateRay(
    ndcX: number,
    ndcY: number,
    aspect: number,
    fov: number,
    near: number,
    origin: readonly [number, number, number],
    quat: readonly [number, number, number, number],
): Ray {
    const t = Math.tan((fov * DEG2RAD) / 2);
    const [dx, dy, dz] = qRotate(
        quat[0],
        quat[1],
        quat[2],
        quat[3],
        ndcX * aspect * t,
        ndcY * t,
        -1,
    );
    const len = Math.hypot(dx, dy, dz) || 1;
    const nx = dx / len;
    const ny = dy / len;
    const nz = dz / len;
    return {
        origin: [origin[0] + nx * near, origin[1] + ny * near, origin[2] + nz * near],
        dir: [nx, ny, nz],
    };
}

/**
 * a world-space pick ray through a canvas pixel (`screenX`/`screenY` in [0, width]×[0, height], origin
 * top-left). Converts the pixel to NDC + aspect and defers to {@link generateRay}: the cursor-driven pick
 * primitive (god-mode pick/drag). `origin`/`quat` are the camera's world pose, `fov`/`near` its params.
 */
export function screenToRay(
    screenX: number,
    screenY: number,
    width: number,
    height: number,
    fov: number,
    near: number,
    origin: readonly [number, number, number],
    quat: readonly [number, number, number, number],
): Ray {
    return generateRay(
        (screenX / width) * 2 - 1,
        1 - (screenY / height) * 2,
        width / height,
        fov,
        near,
        origin,
        quat,
    );
}

/**
 * nearest hit of `ray` against `bodies`, or null. `ray.dir` must be normalized; the returned `distance`
 * is world units along it. `maxDist` (default ∞) rejects farther hits. The caller supplies the candidate
 * poses (live ones Mirror'd from the GPU `bodies`), so this stays a pure function the gold test pins.
 */
export function raycast(ray: Ray, bodies: Iterable<RayBody>, maxDist = Infinity): RayHit | null {
    let best: RayHit | null = null;
    for (const b of bodies) {
        const r = hitShape(ray, b);
        if (!r || r.t > maxDist) continue;
        if (best && r.t >= best.distance) continue;
        best = {
            eid: b.eid,
            distance: r.t,
            point: [
                ray.origin[0] + ray.dir[0] * r.t,
                ray.origin[1] + ray.dir[1] * r.t,
                ray.origin[2] + ray.dir[2] * r.t,
            ],
            normal: [r.nx, r.ny, r.nz],
        };
    }
    return best;
}
