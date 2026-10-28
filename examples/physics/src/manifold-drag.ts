import type { World } from "@dylanebert/shallot";
import { rayHitsCapsule, rayHitsSphere } from "./collision-drag";
import { ManifoldSample, manifoldRotate, vector } from "./manifold";

type Vec = ReturnType<typeof vector>;
export function manifoldDrag(
    w: World,
    origin: Vec,
    translation: Vec,
    press: boolean,
    held: boolean,
): boolean {
    const s = w.resource(ManifoldSample);
    if (!s.scene) return false;
    const f = Math.fround;
    if (press && held) {
        const inv = { v: vector(-s.b.q.v.x, -s.b.q.v.y, -s.b.q.v.z), s: s.b.q.s };
        const o = manifoldRotate(
                inv,
                vector(origin.x - s.b.p.x, origin.y - s.b.p.y, origin.z - s.b.p.z),
            ),
            d = manifoldRotate(inv, translation);
        if (s.scene.endsWith("Sphere"))
            s.tracking = rayHitsSphere(o, d, s.sphere.center, s.sphere.radius);
        else if (
            s.scene === "capsuleVsCapsule" ||
            s.scene === "capsuleVsHull" ||
            s.scene === "triangleVsCapsule"
        )
            s.tracking = rayHitsCapsule(
                o,
                d,
                s.capsule.center1,
                s.capsule.center2,
                s.capsule.radius,
            );
        else {
            let lower = 0,
                upper = 1;
            for (const p of s.hullB!.planes) {
                const n = p.normal,
                    distance = n.x * o.x + n.y * o.y + n.z * o.z - p.offset,
                    speed = n.x * d.x + n.y * d.y + n.z * d.z;
                if (speed === 0) {
                    if (distance > 0) {
                        lower = 2;
                        break;
                    }
                    continue;
                }
                const t = -distance / speed;
                if (speed < 0) lower = Math.max(lower, t);
                else upper = Math.min(upper, t);
            }
            s.tracking = lower <= upper;
        }
        if (s.tracking) Object.assign(s.base, s.b.p);
    }
    const length = f(
        Math.sqrt(
            f(
                f(f(translation.x * translation.x) + f(translation.y * translation.y)) +
                    f(translation.z * translation.z),
            ),
        ),
    );
    if (length > 0 && s.tracking) {
        const p = vector(
            f(origin.x + f(10 * f(translation.x / length))),
            f(origin.y + f(10 * f(translation.y / length))),
            f(origin.z + f(10 * f(translation.z / length))),
        );
        if (press && held) Object.assign(s.origin, p);
        s.b.p = vector(
            f(s.base.x + f(p.x - s.origin.x)),
            f(s.base.y + f(p.y - s.origin.y)),
            f(s.base.z + f(p.z - s.origin.z)),
        );
    }
    if (!held) s.tracking = false;
    return s.tracking;
}
