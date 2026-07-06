// Pick utilities — the layer binding the pose-agnostic raycast to live ECS + Mirror state: candidate
// gathering off the GPU `bodies` snapshot, world↔body-local conversion for joint anchors, and the two
// pick rays (first-person centre, screen cursor). Consumers build their own pick/drag state machines on
// these (the sandbox gravity gun).

import type { State } from "../../engine";
import { Inputs } from "../input";
import type { Mirror } from "../mirror";
import { Camera } from "../render";
import { Transform } from "../transforms";
import { Body } from "./index";
import { qRotate, type Ray, type RayBody, type RayHit, raycast, screenToRay } from "./raycast";
import { BODY_VEC4 } from "./step";

/** the raycast candidates: every Body at its live (Mirror'd) pose, minus `exclude`, occluders and
 *  grabbables alike. Statics/kinematics (mass ≤ 0) are kept so the ray stops on a wall; {@link grabHit}
 *  filters the nearest hit down to a grabbable one. */
export function bodyCandidates(
    state: State,
    bodyMirror: Mirror,
    exclude?: (eid: number) => boolean,
): RayBody[] {
    const snap = bodyMirror.snapshot;
    if (!snap) return [];
    const f = new Float32Array(snap.bytes);
    const cap = f.length / (BODY_VEC4 * 4);
    const out: RayBody[] = [];
    for (const eid of state.query([Body])) {
        if (exclude?.(eid)) continue;
        const po = (0 * cap + eid) * 4;
        const qo = (1 * cap + eid) * 4;
        out.push({
            eid,
            shape: Body.shape.get(eid),
            pos: [f[po], f[po + 1], f[po + 2]],
            quat: [f[qo], f[qo + 1], f[qo + 2], f[qo + 3]],
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
    bodyMirror: Mirror,
    ray: Ray | null,
    maxDist?: number,
    exclude?: (eid: number) => boolean,
): RayHit | null {
    if (!ray) return null;
    const hit = raycast(ray, bodyCandidates(state, bodyMirror, exclude), maxDist);
    return hit && Body.mass.get(hit.eid) > 0 ? hit : null;
}

/** a world point in the held body's local frame (rB for the grab joint): conj(quat) · (point − pos). */
export function worldToLocal(
    bodyMirror: Mirror,
    eid: number,
    point: readonly [number, number, number],
): [number, number, number] {
    const snap = bodyMirror.snapshot;
    if (!snap) return [0, 0, 0];
    const f = new Float32Array(snap.bytes);
    const cap = f.length / (BODY_VEC4 * 4);
    const po = (0 * cap + eid) * 4;
    const qo = (1 * cap + eid) * 4;
    return qRotate(
        -f[qo],
        -f[qo + 1],
        -f[qo + 2],
        f[qo + 3],
        point[0] - f[po],
        point[1] - f[po + 1],
        point[2] - f[po + 2],
    );
}

/** the first-person centre ray: camera position + its forward (−Z). The player's crosshair pick. */
export function forwardRay(state: State, cam: number): Ray | null {
    if (cam < 0 || !state.has(cam, Camera) || !state.has(cam, Transform)) return null;
    const dir = qRotate(
        Transform.rot.x.get(cam),
        Transform.rot.y.get(cam),
        Transform.rot.z.get(cam),
        Transform.rot.w.get(cam),
        0,
        0,
        -1,
    );
    return {
        origin: [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)],
        dir,
    };
}

/** the screen-cursor ray for an orbit camera: `null` when the cursor is off the canvas. The god pick + the
 *  voxel carve both aim with it. */
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
