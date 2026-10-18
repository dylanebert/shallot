import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import type { createApp } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Motes, MotesPlugin } from "./motes";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
let app: Awaited<ReturnType<typeof createApp>>;
beforeAll(async () => {
    app = await openGpuApp({ plugins: [MotesPlugin] });
    console.log("shaft adapter", app.world.gpu.adapter);
});
afterAll(() => app?.dispose());

test("the converged shaft has no coarse luminance jumps across its shadow edge", async () => {
    const { world } = app;
    const camera = world.resource(Room).camera;
    world.resource(Motes).visible = false;
    attachTexture(world, camera, { width: 320, height: 180 });
    world.step(1 / 60);
    world.step(1 / 60);
    for (let i = 0; i < 64; i++) world.step(0);
    const { rgba } = await captureTexture(world, camera);
    const linear = (v: number) => {
        const x = v / 255;
        return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
    };
    const luminance = (x: number, y: number) => {
        const i = (y * 320 + x) * 4;
        return (
            0.2126 * linear(rgba[i]) + 0.7152 * linear(rgba[i + 1]) + 0.0722 * linear(rgba[i + 2])
        );
    };
    let largest = 0;
    // Left shaft edge below the window and above the lit floor; no surface edge crosses it.
    for (let y = 108; y < 158; y++) {
        for (let x = 99; x < 129; x++) {
            largest = Math.max(largest, Math.abs(luminance(x + 1, y) - luminance(x, y)));
        }
    }
    console.log("shaft largest adjacent luminance step", largest);
    expect(largest).toBeLessThan(0.015);
});
