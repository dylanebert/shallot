import { Inputs, type Mirror, play, type State } from "@dylanebert/shallot";
import { Outline } from "@dylanebert/shallot/extras";
import {
    forwardRay,
    grabHit,
    type JointDef,
    type PhysicsStep,
    WORLD,
    worldToLocal,
} from "@dylanebert/shallot/physics/core";

// The gravity gun — the legacy sandbox verb, rebuilt on the world-anchor grab (physics.md "the grab
// dangles from a WORLD anchor"). Click a dynamic body within reach to grab it (a soft spherical joint
// pins it to a world point on the crosshair ray); held, it reels toward HOLD_DIST and follows the aim;
// click again to launch it along the ray (setVelocity), right-click to drop. Unlike a hold-at-pick-depth,
// button-held drag, the gun is click-latched and reels — its own machine on the same physics/core utilities.

const MAX_RANGE = 15; // pick reach
const HOLD_DIST = 2.0; // preferred hold distance ahead of the camera
const REEL_SPEED = 20; // m/s the hold distance reels toward HOLD_DIST
const STIFFNESS = 5000; // soft grab joint — drags without yanking
const LAUNCH_SPEED = 20; // launch velocity along the aim ray

export type GunMode = "default" | "hover" | "grab";

export interface Gun {
    /** advance the gun a frame against the camera's crosshair ray and return the mode. */
    update(state: State, cam: number): GunMode;
}

export function gun(
    step: PhysicsStep,
    bodyMirror: Mirror,
    baseJoints: readonly JointDef[],
    exclude?: (eid: number) => boolean, // drop the player's own capsule so it never occludes the crosshair
): Gun {
    const slot = baseJoints.length; // the grab joint appends after the scene's authored joints
    let held = -1;
    let holdDist = 0;
    let grabTick = -1; // the fixed tick the grab was authored on — see the reel gate below
    let wasLeft = false;
    let wasRight = false;
    let outlined = -1;

    function bodyPos(eid: number): [number, number, number] {
        const snap = bodyMirror.snapshot;
        if (!snap) return [0, 0, 0];
        const f = new Float32Array(snap.bytes);
        const o = eid * 4; // col 0 (posLin), eid-indexed
        return [f[o], f[o + 1], f[o + 2]];
    }

    // a muted cool tint says "grabbable", the warm accent says "holding". Depth-tested (occlude 1) so
    // the highlight never reads through walls — this is a diegetic FPS cue, not the god-view marker.
    function highlight(state: State, active: number, mode: GunMode): void {
        if (active !== outlined) {
            if (outlined >= 0 && state.has(outlined, Outline)) state.remove(outlined, Outline);
            if (active >= 0) {
                state.add(active, Outline);
                Outline.occlude.set(active, 1);
            }
            outlined = active;
        }
        if (active < 0) return;
        const heldNow = mode === "grab";
        Outline.color.set(
            active,
            heldNow ? 0.8 : 0.5,
            heldNow ? 0.5 : 0.56,
            heldNow ? 0.2 : 0.62,
            1,
        );
        Outline.width.set(active, heldNow ? 3 : 2);
    }

    return {
        update(state, cam) {
            const ray = forwardRay(state, cam);
            const leftPressed = Inputs.mouse.left && !wasLeft;
            const rightPressed = Inputs.mouse.right && !wasRight;
            wasLeft = Inputs.mouse.left;
            wasRight = Inputs.mouse.right;

            let mode: GunMode = "default";
            let active = -1;

            if (held >= 0) {
                if (leftPressed && ray) {
                    // launch: release the joint, then kick along the aim — both queue-ordered, so the
                    // next fixed step integrates the throw with no joint fighting it
                    const eid = held;
                    held = -1;
                    step.setJoints([...baseJoints]);
                    step.setVelocity(
                        eid,
                        ray.dir[0] * LAUNCH_SPEED,
                        ray.dir[1] * LAUNCH_SPEED,
                        ray.dir[2] * LAUNCH_SPEED,
                    );
                    play(state, "launch", { pos: bodyPos(eid) });
                } else if (rightPressed) {
                    held = -1;
                    step.setJoints([...baseJoints]);
                } else {
                    // hold the anchor at the hit point until one fixed tick has run: the joint's
                    // one-shot construction guard judges |anchor − pin| at its init step, and a thin
                    // link's reach is centimetres — reeling before init moves the anchor past it and
                    // the grab is rejected silently (the "nothing happens at all" dead grab)
                    if (ray && state.time.fixedTick > grabTick) {
                        const reel = REEL_SPEED * state.time.deltaTime;
                        holdDist =
                            holdDist > HOLD_DIST
                                ? Math.max(HOLD_DIST, holdDist - reel)
                                : Math.min(HOLD_DIST, holdDist + reel);
                        step.setJointAnchor(
                            slot,
                            ray.origin[0] + ray.dir[0] * holdDist,
                            ray.origin[1] + ray.dir[1] * holdDist,
                            ray.origin[2] + ray.dir[2] * holdDist,
                        );
                    }
                    mode = "grab";
                    active = held;
                }
            } else {
                const hit = grabHit(state, bodyMirror, ray, MAX_RANGE, exclude);
                if (leftPressed && hit) {
                    held = hit.eid;
                    holdDist = Math.max(hit.distance, 0.1);
                    grabTick = state.time.fixedTick;
                    step.setJoints([
                        ...baseJoints,
                        {
                            a: WORLD,
                            b: held,
                            rA: hit.point,
                            rB: worldToLocal(bodyMirror, held, hit.point),
                            stiffnessLin: STIFFNESS,
                            stiffnessAng: 0, // spherical — the held body dangles
                        },
                    ]);
                    play(state, "grab", { pos: hit.point });
                    mode = "grab";
                    active = held;
                } else if (hit) {
                    mode = "hover";
                    active = hit.eid;
                }
            }

            highlight(state, active, mode);
            return mode;
        },
    };
}
