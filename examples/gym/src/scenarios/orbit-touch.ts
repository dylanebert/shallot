import {
    AmbientLight,
    Camera,
    CameraMode,
    Color,
    GlazePlugin,
    InputPlugin,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    SlabPlugin,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { type Params, register, type Scenario } from "../gym";

// orbit-touch — the input-substrate touch surface's own atom: an Orbit camera targeting a box, driven
// externally by the driver-level touch gate (`../../test/touch.playwright.ts`) via real CDP
// `Input.dispatchTouchEvent` — see that file's header for why an external driver, not `assert`, owns the
// verdict here (a scenario's `assert` runs environment-unaware, and dispatching real multi-touch needs a
// real browser session; `bun bench`/a human already play the external-driver role for other scenarios).
// `window.__orbitPose()` exposes the pose the driver reads: the public `Orbit` component's own
// yaw/pitch/distance/pan (the immediate per-frame target the gesture math writes, reached the same way
// `orbit.test.ts` reads it) folded with the camera's live `Transform` — the direct-component half is what
// lets the driver isolate one gesture's own quantity from `smoothLerp`'s multi-frame lag (extras/orbit).
//
// Headless-deterministic, no `assert`: nothing here moves on its own (no live-only randomness, no
// physics), so a bare boot is the whole headless contract — the verdict this atom exists for lives
// entirely in the external driver.

export interface OrbitTouchPose {
    yaw: number;
    pitch: number;
    distance: number;
    pan: [number, number, number];
    pos: [number, number, number] | null;
    quat: [number, number, number, number] | null;
}

declare global {
    interface Window {
        __orbitPose?: () => OrbitTouchPose | null;
    }
}

let camEid = -1;

const scenario: Scenario = {
    name: "orbit-touch",

    async build(_canvas: HTMLCanvasElement, _params: Params) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        state.add(state.create(), AmbientLight);

        const box = state.create();
        state.add(box, Transform);
        state.add(box, Part);
        state.add(box, Color);
        Color.rgba.set(box, 0.8, 0.4, 0.3, 1);

        camEid = state.create();
        state.add(camEid, Transform);
        state.add(camEid, Camera);
        state.add(camEid, Sear);
        state.add(camEid, Orbit);
        Camera.mode.set(camEid, CameraMode.Perspective);
        Orbit.distance.set(camEid, 10);
        Orbit.yaw.set(camEid, 0);
        Orbit.pitch.set(camEid, 0);
        Orbit.target.set(camEid, box);

        // Transform read directly (gym's own harness wiring, `../gym.ts`, doesn't expose the published
        // `@dylanebert/shallot/harness` `read()` — this is the same source that channel would return for
        // a Transform-only entity).
        window.__orbitPose = (): OrbitTouchPose | null => {
            if (camEid < 0) return null;
            const pos = Transform.pos.read(camEid, new Float32Array(4));
            const rot = Transform.rot.read(camEid, new Float32Array(4));
            return {
                yaw: Orbit.yaw.get(camEid),
                pitch: Orbit.pitch.get(camEid),
                distance: Orbit.distance.get(camEid),
                pan: [Orbit.pan.x.get(camEid), Orbit.pan.y.get(camEid), Orbit.pan.z.get(camEid)],
                pos: [pos[0], pos[1], pos[2]],
                quat: [rot[0], rot[1], rot[2], rot[3]],
            };
        };

        return {
            state,
            dispose() {
                camEid = -1;
                window.__orbitPose = undefined;
                dispose();
            },
        };
    },

    live(): string {
        return "orbit-touch — drag to orbit, pinch/two-finger to zoom/pan; window.__orbitPose() is the external touch driver's read";
    },
};

register(scenario);
