import { Camera, Compute, Inputs, type Mirror, Transform } from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { Views } from "@dylanebert/shallot/render/core";

/** Read actual view/camera/input state independently of the edit and pixel predicates. */
export function cameraSnapshot() {
    return {
        frame: Compute.frame,
        mouse: { ...Inputs.mouse },
        touch: { ...Inputs.touch },
        cameras: [...Views].map(([eid, view]) => ({
            eid,
            canvas: view.canvas !== null,
            fov: Camera.fov.get(eid),
            near: Camera.near.get(eid),
            pos: [Transform.pos.x.get(eid), Transform.pos.y.get(eid), Transform.pos.z.get(eid)],
            rot: [
                Transform.rot.x.get(eid),
                Transform.rot.y.get(eid),
                Transform.rot.z.get(eid),
                Transform.rot.w.get(eid),
            ],
            yaw: Orbit.yaw.get(eid),
            pitch: Orbit.pitch.get(eid),
            distance: Orbit.distance.get(eid),
        })),
    };
}

/** Set only the live display camera for device controls, never the authored scene. */
export function poseCamera(values: {
    distance?: number;
    pitch?: number;
    yaw?: number;
    smoothness?: number;
    mode?: number;
}): void {
    for (const [eid, view] of Views) {
        if (!view.canvas) continue;
        for (const key of ["distance", "pitch", "yaw", "smoothness", "mode"] as const) {
            const value = values[key];
            if (value !== undefined) Orbit[key].set(eid, value);
        }
    }
}

// The project's own tiny test/boot helpers — published-surface-only (no reach into the repo harness),
// the same shape as voxel's `src/harness.ts`. {@link Check} is the gate's verdict shape, read by
// `window.__roadsGate`.

export interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

/** await `n` animation frames — lets the running render loop advance a known amount. */
export function frames(n: number): Promise<void> {
    return new Promise((resolve) => {
        let i = 0;
        const tick = () => (++i >= n ? resolve() : requestAnimationFrame(tick));
        requestAnimationFrame(tick);
    });
}

// Mirror is 1-2 frames stale by design (a staging ring + async map). After mutating state the GPU reads,
// wait until a snapshot encoded *after* now lands, so a readback reflects the new state. Bounded — a stuck
// map resolves to the loop cap.
export async function settle(m: Mirror, max = 120): Promise<void> {
    const target = Compute.frame + 2;
    for (let i = 0; i < max; i++) {
        await frames(1);
        if (m.snapshot && m.snapshot.frame >= target) return;
    }
}
