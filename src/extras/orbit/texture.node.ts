import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { focusCanvas, pointerMove, pressButton, releaseButton } from "../../core/input";
import { attachTexture, Camera, RenderingPlugin } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { resizeViewport, Time } from "../../engine";
import { Orbit, OrbitPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

test("orbit uses the texture target height when CSS viewport height is missing or zero", async () => {
    const app = await openGpuApp({ defaults: false, plugins: [RenderingPlugin, OrbitPlugin] });
    try {
        const world = app.world;
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, Orbit);
        attachTexture(world, camera, { width: 320, height: 180 });
        const orbit = world.storage(Orbit);
        const turns: number[] = [];
        for (const cssHeight of [undefined, 0, 360]) {
            if (cssHeight !== undefined) {
                resizeViewport(world, 0, 320, cssHeight, 1);
                focusCanvas(world, 0);
            }
            const yaw = orbit.yaw.get(camera);
            pressButton(world, "left");
            pointerMove(world, 45, 0, 45, 0);
            world.step(Time.FIXED_DT);
            turns.push(yaw - orbit.yaw.get(camera));
            releaseButton(world, "left");
        }
        expect(turns[0]!).toBeCloseTo(3.8 / 4, 5);
        expect(turns[1]!).toBeCloseTo(3.8 / 4, 5);
        expect(turns[2]!).toBeCloseTo(3.8 / 8, 5);
    } finally {
        app.dispose();
    }
});
