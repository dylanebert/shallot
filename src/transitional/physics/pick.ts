import type { Ray, World } from "../../engine";
import { Body, type BodyState } from "./index";
import { qRotate, type RayBody, type RayHit, raycast } from "./raycast";

/** the raycast candidates: every Body at its live pose (`read`, usually `Physics.readBody`), minus `exclude`, occluders and
 *  grabbables alike. Statics/kinematics (mass ≤ 0) are kept so the ray stops on a wall; {@link grabHit}
 *  filters the nearest hit down to a grabbable one. Empty until `read` has a live pose to report. */
export function bodyCandidates(
    world: World,
    read: (eid: number) => BodyState | null,
    exclude?: (eid: number) => boolean,
): RayBody[] {
    const out: RayBody[] = [];
    for (const eid of world.query([Body])) {
        if (exclude?.(eid)) continue;
        const live = read(eid);
        if (!live) continue;
        out.push({
            eid,
            shape: world.storage(Body).shape.get(eid),
            pos: live.position,
            quat: live.rotation,
            half: [
                world.storage(Body).halfExtents.x.get(eid),
                world.storage(Body).halfExtents.y.get(eid),
                world.storage(Body).halfExtents.z.get(eid),
            ],
            radius: world.storage(Body).halfExtents.w.get(eid),
        });
    }
    return out;
}

/** the body the crosshair grabs along `ray` (null ray = no aim): the nearest solid Body within `maxDist`,
 *  returned ONLY when it's dynamic (grabbable). Statics occlude: a wall nearer than any dynamic body
 *  blocks the grab (returns null), so you can't grab through walls. `exclude` drops a body from the cast
 *  entirely (neither occludes nor grabs, e.g. the player's own capsule). */
export function grabHit(
    world: World,
    read: (eid: number) => BodyState | null,
    ray: Ray | null,
    maxDist?: number,
    exclude?: (eid: number) => boolean,
): RayHit | null {
    if (!ray) return null;
    const hit = raycast(ray, bodyCandidates(world, read, exclude), maxDist);
    return hit && world.storage(Body).mass.get(hit.eid) > 0 ? hit : null;
}

/** a world point in the held body's local frame (rB for the grab joint): conj(quat) · (point − pos), or
 *  `null` when `read` has no live pose for `eid` (a body that despawned between the cast and the grab
 *  — the caller drops the grab rather than pinning to a bogus local anchor). */
export function worldToLocal(
    read: (eid: number) => BodyState | null,
    eid: number,
    point: readonly [number, number, number],
): [number, number, number] | null {
    const live = read(eid);
    if (!live) return null;
    const [qx, qy, qz, qw] = live.rotation;
    const [px, py, pz] = live.position;
    return qRotate(-qx, -qy, -qz, qw, point[0] - px, point[1] - py, point[2] - pz);
}
