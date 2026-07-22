// Pick utilities — the layer binding the pose-agnostic raycast to live ECS + backend state: candidate
// gathering off the installed backend's live pose, world↔body-local conversion for joint anchors, and
// the two pick rays (first-person centre, screen cursor). Consumers build their own pick/drag state
// machines on these (the sandbox gravity gun).

import type { State } from "../../engine";
import { Inputs } from "../input";
import { Camera } from "../render";
import { Transform } from "../transforms";
import { Body, type PhysicsBackend } from "./index";
import { qRotate, type Ray, type RayBody, type RayHit, raycast, screenToRay } from "./raycast";

/** the raycast candidates: every Body at its live backend pose, minus `exclude`, occluders and
 *  grabbables alike. Statics/kinematics (mass ≤ 0) are kept so the ray stops on a wall; {@link grabHit}
 *  filters the nearest hit down to a grabbable one. Empty until the backend has a live pose to report. */
export function bodyCandidates(
    state: State,
    backend: PhysicsBackend,
    exclude?: (eid: number) => boolean,
): RayBody[] {
    const out: RayBody[] = [];
    for (const eid of state.query([Body])) {
        if (exclude?.(eid)) continue;
        const live = backend.readBody(eid);
        if (!live) continue;
        out.push({
            eid,
            shape: Body.shape.get(eid),
            pos: live.pos,
            quat: live.quat,
            half: [
                Body.halfExtents.x.get(eid),
                Body.halfExtents.y.get(eid),
                Body.halfExtents.z.get(eid),
            ],
            radius: Body.halfExtents.w.get(eid),
        });
    }
    return out;
}

/** the body the crosshair grabs along `ray` (null ray = no aim): the nearest solid Body within `maxDist`,
 *  returned ONLY when it's dynamic (grabbable). Statics occlude: a wall nearer than any dynamic body
 *  blocks the grab (returns null), so you can't grab through walls. `exclude` drops a body from the cast
 *  entirely (neither occludes nor grabs, e.g. the player's own capsule). */
export function grabHit(
    state: State,
    backend: PhysicsBackend,
    ray: Ray | null,
    maxDist?: number,
    exclude?: (eid: number) => boolean,
): RayHit | null {
    if (!ray) return null;
    const hit = raycast(ray, bodyCandidates(state, backend, exclude), maxDist);
    return hit && Body.mass.get(hit.eid) > 0 ? hit : null;
}

/** a world point in the held body's local frame (rB for the grab joint): conj(quat) · (point − pos), or
 *  `null` when the backend has no live pose for `eid` (a body that despawned between the cast and the grab
 *  — the caller drops the grab rather than pinning to a bogus local anchor). */
export function worldToLocal(
    backend: PhysicsBackend,
    eid: number,
    point: readonly [number, number, number],
): [number, number, number] | null {
    const live = backend.readBody(eid);
    if (!live) return null;
    const [qx, qy, qz, qw] = live.quat;
    const [px, py, pz] = live.pos;
    return qRotate(-qx, -qy, -qz, qw, point[0] - px, point[1] - py, point[2] - pz);
}

/** the first-person centre ray: camera position + its normalized forward (−Z). The player's crosshair pick.
 *  Unlike {@link cursorRay} (which offsets the origin to the near plane), the origin stays AT the camera. */
export function forwardRay(state: State, cam: number): Ray | null {
    if (cam < 0 || !state.has(cam, Camera) || !state.has(cam, Transform)) return null;
    const [dx, dy, dz] = qRotate(
        Transform.rot.x.get(cam),
        Transform.rot.y.get(cam),
        Transform.rot.z.get(cam),
        Transform.rot.w.get(cam),
        0,
        0,
        -1,
    );
    const len = Math.hypot(dx, dy, dz) || 1;
    return {
        origin: [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)],
        dir: [dx / len, dy / len, dz / len],
    };
}

/** the screen-cursor ray for an orbit camera: `null` when the cursor is off the canvas. The god pick + the
 *  voxel carve both aim with it. The pick aspect derives from the canvas CSS box (`Inputs.mouse.canvas*`),
 *  so it can diverge from the render aspect under an aspect-distorting `Resolution` override. */
export function cursorRay(state: State, cam: number): Ray | null {
    if (cam < 0 || !state.has(cam, Camera) || !state.has(cam, Transform)) return null;
    if (!Inputs.mouse.hover) return null;
    return screenToRay(
        Inputs.mouse.x,
        Inputs.mouse.y,
        Inputs.mouse.canvasWidth,
        Inputs.mouse.canvasHeight,
        Camera.fov.get(cam),
        Camera.near.get(cam),
        [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)],
        [
            Transform.rot.x.get(cam),
            Transform.rot.y.get(cam),
            Transform.rot.z.get(cam),
            Transform.rot.w.get(cam),
        ],
    );
}
