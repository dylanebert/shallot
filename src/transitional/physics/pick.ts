// Pick utilities — the layer binding the pose-agnostic raycast to live ECS + backend state: candidate
// gathering off the installed backend's live pose, world↔body-local conversion for joint anchors, and
// the two pick rays (first-person centre, screen cursor). Consumers build their own pick/drag state
// machines on these (the sandbox gravity gun).

import { Devices } from "../../core/input";
import { Camera } from "../../core/rendering";
import type { World } from "../../engine";
import { GlobalTransform } from "../../engine";
import { Body, type BodyState } from "./index";
import { qRotate, type Ray, type RayBody, type RayHit, raycast, screenToRay } from "./raycast";

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

/** the first-person centre ray: camera position + its normalized forward (−Z). The player's crosshair pick.
 *  Unlike {@link cursorRay} (which offsets the origin to the near plane), the origin stays AT the camera. */
export function forwardRay(world: World, cam: number): Ray | null {
    if (cam < 0 || !world.has(cam, Camera) || !world.has(cam, GlobalTransform)) return null;
    const global = world.storage(GlobalTransform);
    const [dx, dy, dz] = qRotate(
        global.rotation.x.get(cam),
        global.rotation.y.get(cam),
        global.rotation.z.get(cam),
        global.rotation.w.get(cam),
        0,
        0,
        -1,
    );
    const len = Math.hypot(dx, dy, dz) || 1;
    return {
        origin: [
            global.translation.x.get(cam),
            global.translation.y.get(cam),
            global.translation.z.get(cam),
        ],
        dir: [dx / len, dy / len, dz / len],
    };
}

/** the screen-cursor ray for an orbit camera: `null` when the cursor is off the canvas. The god pick aims with it. The pick aspect derives from the World-scoped viewport row,
 * so it can diverge from the render aspect under an aspect-distorting `Resolution` override. */
export function cursorRay(world: World, cam: number): Ray | null {
    if (cam < 0 || !world.has(cam, Camera) || !world.has(cam, GlobalTransform)) return null;
    const global = world.storage(GlobalTransform);
    const input = world.resource(Devices);
    if (!input.pointer.hover) return null;
    const viewport = input.viewport.get(input.focused);
    return screenToRay(
        input.pointer.x,
        input.pointer.y,
        viewport?.cssWidth ?? 0,
        viewport?.cssHeight ?? 0,
        world.storage(Camera).fov.get(cam),
        world.storage(Camera).near.get(cam),
        [
            global.translation.x.get(cam),
            global.translation.y.get(cam),
            global.translation.z.get(cam),
        ],
        [
            global.rotation.x.get(cam),
            global.rotation.y.get(cam),
            global.rotation.z.get(cam),
            global.rotation.w.get(cam),
        ],
    );
}
