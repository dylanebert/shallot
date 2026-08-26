import { run } from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { installHarness } from "@dylanebert/shallot/harness";

// The fixture app behind S4's touch verification gate (`../test/touch.playwright.ts`). A camera with
// `Orbit` targets a box; the harness exposes `window.__orbitPose()` so the Playwright driver can read
// the orbit pose straight off the public `Orbit` component (yaw/pitch/distance/pan) rather than
// re-deriving it from the smoothed camera transform, which lags a live drag by several frames
// (`smoothLerp` in extras/orbit). `harness.read()` is folded in too — the published camera-pose channel
// `shallot verify` itself drives — so the gate reads both the raw orbit target pose and the rendered
// camera transform from one call.
const scene = `<scene>
    <a ambient-light="intensity: 0.6" />
    <a id="camera" camera sear transform orbit="distance: 10; yaw: 0; pitch: 0; target: @box" />
    <a id="box" part transform color="rgba: 0.8 0.4 0.3" />
</scene>`;

const { state, dispose } = await run({ plugins: [OrbitPlugin], scene });
const harness = installHarness(state);
harness.run = async () => ({ ok: true, checks: [{ name: "booted", ok: true }] });

interface OrbitPose {
    yaw: number;
    pitch: number;
    distance: number;
    pan: [number, number, number];
    pos: [number, number, number] | null;
    quat: [number, number, number, number] | null;
}

declare global {
    interface Window {
        __orbitPose?: () => OrbitPose | null;
    }
}

window.__orbitPose = (): OrbitPose | null => {
    const [eid] = state.query([Orbit]);
    if (eid === undefined) return null;
    const camera = harness.read?.(eid) ?? null;
    return {
        yaw: Orbit.yaw.get(eid),
        pitch: Orbit.pitch.get(eid),
        distance: Orbit.distance.get(eid),
        pan: [Orbit.pan.x.get(eid), Orbit.pan.y.get(eid), Orbit.pan.z.get(eid)],
        pos: camera?.pos ?? null,
        quat: camera?.quat ?? null,
    };
};

// HMR re-runs this module — dispose the old State + RAF loop, or each edit stacks another.
if (import.meta.hot) {
    import.meta.hot.dispose(() => dispose());
}
