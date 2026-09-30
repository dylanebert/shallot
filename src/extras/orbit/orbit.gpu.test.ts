import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.gpu);

import {
    devices,
    focus,
    pointerButton,
    pointerMove,
    resizeViewport,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { gpuApps } from "../../../scripts/gpu.fixture";

const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [OrbitPlugin] }]);

test("the public Orbit consumer consumes held, released and neutral pointer facts to produce a sensitivity-scaled camera pose without a canvas, browser producer or renderer", async () => {
    const app = subjects()[0];
    try {
        const state = app.state;
        const camera = state.create();
        state.add(camera, Transform);
        state.add(camera, Orbit);
        state.of(Orbit).sensitivity.set(camera, 0.01);
        resizeViewport(state, 0, 320, 180, 2);
        focus(state, 0);

        state.step(0); // initializes OrbitSmooth and produces the initial pose
        const initialYaw = state.of(Orbit).yaw.get(camera);
        const initialX = state.of(Transform).pos.x.get(camera);
        const initialZ = state.of(Transform).pos.z.get(camera);

        pointerButton(state, "left", true);
        pointerMove(state, {
            x: 160,
            y: 90,
            deltaX: 20,
            deltaY: -6,
            canvasIndex: 0,
        });
        state.step(Time.FIXED_DT);
        if (Math.abs(state.of(Orbit).yaw.get(camera) - (initialYaw - 20 * 0.01)) > 0.000001)
            throw new Error("Orbit did not consume the held drag at its sensitivity");
        if (
            state.of(Transform).pos.x.get(camera) === initialX &&
            state.of(Transform).pos.z.get(camera) === initialZ
        )
            throw new Error("Orbit did not produce a camera pose from the supplied drag");
        if (!devices(state).mouse.left) throw new Error("Orbit lost the held button fact");

        pointerButton(state, "left", false);
        state.step(Time.FIXED_DT);
        const releasedYaw = state.of(Orbit).yaw.get(camera);
        if (devices(state).mouse.left) throw new Error("Orbit retained a released button");
        state.step(Time.FIXED_DT); // neutral: no stale drag delta may be replayed
        if (state.of(Orbit).yaw.get(camera) !== releasedYaw)
            throw new Error("Orbit replayed released drag input on a neutral step");
        if (devices(state).keys.released.size !== 0)
            throw new Error("unrelated keyboard release state leaked into Orbit");
    } finally {
        app.dispose();
    }
});
