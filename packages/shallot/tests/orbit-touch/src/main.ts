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

// The touch tier's fixture app: an Orbit camera targeting a box, driven externally by
// `../touch.playwright.ts` through real CDP `Input.dispatchTouchEvent`. The verdict lives entirely in
// that driver — see its header for why an external driver owns it — so nothing here asserts anything.
//
// `window.__orbitPose()` is the read the driver isolates a gesture with: the public `Orbit` component's
// own yaw/pitch/distance/pan is the immediate per-frame target the gesture math writes, while the
// camera's live `Transform` is the smoothed rendered pose that lags a drag by several frames
// (`smoothLerp`, extras/orbit). Folding both is what lets each assertion name one quantity's own
// movement and hold the others.
//
// Nothing moves on its own — no randomness, no physics — so the page is deterministic from boot.

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

const camEid = state.create();
state.add(camEid, Transform);
state.add(camEid, Camera);
state.add(camEid, Sear);
state.add(camEid, Orbit);
Camera.mode.set(camEid, CameraMode.Perspective);
Orbit.distance.set(camEid, 10);
Orbit.yaw.set(camEid, 0);
Orbit.pitch.set(camEid, 0);
Orbit.target.set(camEid, box);

window.__orbitPose = (): OrbitTouchPose | null => {
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

// HMR re-runs this module — dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => {
        window.__orbitPose = undefined;
        dispose();
    });
}
