import { setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { resizeViewport } from "../../engine";

setDefaultTimeout(CEILING.gpu);

import {
    Devices,
    focusCanvas,
    pointerButton,
    pointerMove,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { gpuApps } from "../../../scripts/gpu.fixture";

const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [OrbitPlugin] }]);

test("the public Orbit consumer consumes held, released and neutral pointer facts to produce a sensitivity-scaled camera pose without a canvas, browser producer or renderer", async () => {
    const app = subjects()[0];
    try {
        const world = app.world;
        const _devices = world.resource(Devices);
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Orbit);
        world.storage(Orbit).sensitivity.set(camera, 0.01);
        resizeViewport(world, 0, 320, 180, 2);
        focusCanvas(world, 0);

        world.step(0); // initializes OrbitSmooth and produces the initial pose
        const initialYaw = world.storage(Orbit).yaw.get(camera);
        const initialX = world.storage(Transform).translation.x.get(camera);
        const initialZ = world.storage(Transform).translation.z.get(camera);

        pointerButton(world, "left", true);
        pointerMove(world, {
            x: 160,
            y: 90,
            deltaX: 20,
            deltaY: -6,
            canvasIndex: 0,
        });
        world.step(Time.FIXED_DT);
        if (Math.abs(world.storage(Orbit).yaw.get(camera) - (initialYaw - 20 * 0.01)) > 0.000001)
            throw new Error("Orbit did not consume the held drag at its sensitivity");
        if (
            world.storage(Transform).translation.x.get(camera) === initialX &&
            world.storage(Transform).translation.z.get(camera) === initialZ
        )
            throw new Error("Orbit did not produce a camera pose from the supplied drag");
        if (!_devices.pointer.left) throw new Error("Orbit lost the held button fact");

        pointerButton(world, "left", false);
        world.step(Time.FIXED_DT);
        const releasedYaw = world.storage(Orbit).yaw.get(camera);
        if (_devices.pointer.left) throw new Error("Orbit retained a released button");
        world.step(Time.FIXED_DT); // neutral: no stale drag delta may be replayed
        if (world.storage(Orbit).yaw.get(camera) !== releasedYaw)
            throw new Error("Orbit replayed released drag input on a neutral step");
        if (_devices.keys.released.size !== 0)
            throw new Error("unrelated keyboard release state leaked into Orbit");
    } finally {
        app.dispose();
    }
});
