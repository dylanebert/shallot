import type { World } from "@dylanebert/shallot";
import { CollisionState } from "./collision";

type Vec = { x: number; y: number; z: number };
const sub = (a: Vec, b: Vec): Vec => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec, b: Vec) => a.x * b.x + a.y * b.y + a.z * b.z;
function sphere(origin: Vec, direction: Vec, center: Vec, radius: number): boolean {
    const p = sub(origin, center),
        a = dot(direction, direction),
        b = dot(p, direction),
        c = dot(p, p) - radius * radius;
    if (c <= 0) return true;
    const d = b * b - a * c;
    if (d < 0) return false;
    const t = (-b - Math.sqrt(d)) / a;
    return t >= 0 && t <= 1;
}
function capsule(origin: Vec, direction: Vec, a: Vec, b: Vec, radius: number): boolean {
    if (sphere(origin, direction, a, radius) || sphere(origin, direction, b, radius)) return true;
    const axis = sub(b, a),
        p = sub(origin, a),
        aa = dot(axis, axis),
        ad = dot(axis, direction),
        ap = dot(axis, p);
    const A = aa * dot(direction, direction) - ad * ad,
        B = aa * dot(p, direction) - ap * ad,
        C = aa * dot(p, p) - ap * ap - radius * radius * aa;
    const d = B * B - A * C;
    if (A <= 0 || d < 0) return false;
    const t = (-B - Math.sqrt(d)) / A,
        y = ap + t * ad;
    return t >= 0 && t <= 1 && y >= 0 && y <= aa;
}
function hull(origin: Vec, direction: Vec, p: Vec[]): boolean {
    let lower = 0,
        upper = 1;
    const center = {
        x: p.reduce((s, v) => s + v.x, 0) / 8,
        y: p.reduce((s, v) => s + v.y, 0) / 8,
        z: p.reduce((s, v) => s + v.z, 0) / 8,
    };
    for (const [a, b, c] of [
        [0, 1, 2],
        [4, 7, 6],
        [0, 4, 5],
        [3, 2, 6],
        [1, 5, 6],
        [0, 3, 7],
    ]) {
        const start = p[a!]!,
            u = sub(p[b!]!, start),
            v = sub(p[c!]!, start);
        let n = { x: u.y * v.z - u.z * v.y, y: u.z * v.x - u.x * v.z, z: u.x * v.y - u.y * v.x };
        if (dot(n, sub(center, start)) > 0) n = { x: -n.x, y: -n.y, z: -n.z };
        const distance = dot(n, sub(origin, start)),
            speed = dot(n, direction);
        if (speed === 0) {
            if (distance > 0) return false;
            continue;
        }
        const t = -distance / speed;
        if (speed < 0) lower = Math.max(lower, t);
        else upper = Math.min(upper, t);
        if (lower > upper) return false;
    }
    return true;
}
export function collisionDrag(
    world: World,
    origin: Vec,
    translation: Vec,
    press: boolean,
    held: boolean,
): boolean {
    const s = world.resource(CollisionState);
    if (s.scene !== "shapeCast" && s.scene !== "overlapWorld") return false;
    const f = Math.fround,
        length = f(
            Math.sqrt(
                f(
                    f(f(translation.x * translation.x) + f(translation.y * translation.y)) +
                        f(translation.z * translation.z),
                ),
            ),
        );
    const at = {
        x: f(origin.x + f(10 * f(translation.x / length))),
        y: f(origin.y + f(10 * f(translation.y / length))),
        z: f(origin.z + f(10 * f(translation.z / length))),
    };
    if (press && held) {
        s.tracking = s.sweeps.some((q) =>
            q.points.length === 8
                ? hull(origin, translation, q.points)
                : q.points.length === 1
                  ? sphere(origin, translation, q.points[0]!, q.radius)
                  : capsule(origin, translation, q.points[0]!, q.points[1]!, q.radius),
        );
        if (s.tracking) {
            Object.assign(s.dragOrigin, at);
            Object.assign(s.dragBase, s.offset);
        }
    }
    if (s.tracking) {
        s.offset.x = f(s.dragBase.x + f(at.x - s.dragOrigin.x));
        s.offset.y = f(s.dragBase.y + f(at.y - s.dragOrigin.y));
        s.offset.z = f(s.dragBase.z + f(at.z - s.dragOrigin.z));
    }
    if (!held) s.tracking = false;
    return s.tracking;
}
