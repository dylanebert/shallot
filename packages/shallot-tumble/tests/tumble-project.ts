// World→screen projection for the tumble sample host, shared by the interaction probe (which aims the pick
// ray at a projected body pixel) and the overlay layer (which pins `string3d` HTML labels over projected
// world points). One source of truth for the inverse of `generateRay`'s camera convention — a re-derived
// second copy is exactly the drift this avoids.

import { Camera, Transform } from "@dylanebert/shallot";
import { qRotate } from "@dylanebert/shallot/physics/core";

/** the camera pose the projection needs, read off the orbit-driven camera entity's `Transform` + `Camera`. */
export interface CameraPose {
    pos: [number, number, number];
    quat: [number, number, number, number];
    fovDeg: number;
    near: number;
}

/** read the live camera pose (world position, orientation quat, vertical fov in degrees, near plane). */
export function cameraPose(cam: number): CameraPose {
    return {
        pos: [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)],
        quat: [
            Transform.rot.x.get(cam),
            Transform.rot.y.get(cam),
            Transform.rot.z.get(cam),
            Transform.rot.w.get(cam),
        ],
        fovDeg: Camera.fov.get(cam),
        near: Camera.near.get(cam),
    };
}

/** project a world point to a canvas pixel, inverting `generateRay`'s camera-space convention (camera looks
 *  −Z; fov is vertical degrees). `front` is false when the point is behind the camera (caller hides it). */
export function worldToScreen(
    camPos: [number, number, number],
    camQuat: [number, number, number, number],
    fovDeg: number,
    w: number,
    h: number,
    p: { x: number; y: number; z: number },
): { x: number; y: number; front: boolean } {
    const [lx, ly, lz] = qRotate(
        -camQuat[0],
        -camQuat[1],
        -camQuat[2],
        camQuat[3],
        p.x - camPos[0],
        p.y - camPos[1],
        p.z - camPos[2],
    );
    if (lz >= -1e-4) return { x: 0, y: 0, front: false };
    const t = Math.tan((fovDeg * Math.PI) / 180 / 2);
    const aspect = w / h;
    const ndcX = lx / (-lz * aspect * t);
    const ndcY = ly / (-lz * t);
    return { x: ((ndcX + 1) / 2) * w, y: ((1 - ndcY) / 2) * h, front: true };
}
