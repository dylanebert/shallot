import { expect, setDefaultTimeout, test } from "bun:test";
import { Transform } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { MOTE_COUNT, Motes, MotesPlugin } from "./motes";
import { Player } from "./player";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
const decode = (x: number) => {
    const v = x / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
test("a shaft mote adds at least five times an outside mote's display luminance without clipping", async () => {
    const app = await openGpuApp({ plugins: [MotesPlugin] });
    try {
        const { world } = app;
        console.log("mote adapter", world.gpu.adapter);
        const camera = world.resource(Room).camera;
        world.remove(world.resource(Room).player, Player);
        attachTexture(world, camera, { width: 1280, height: 720 });
        world.storage(Transform).translation.set(camera, 0, 1.7, 2, 0);
        world.tick();
        const motes = world.resource(Motes);
        const positions = new Float32Array(MOTE_COUNT * 4);
        for (let i = 0; i < MOTE_COUNT; i++) positions[i * 4] = 100;
        const frame = async (position: readonly number[] | null, steps: number) => {
            if (position) positions.set(position, 0);
            world.gpu.device.queue.writeBuffer(motes.positions!, 0, positions);
            motes.visible = position !== null;
            for (let i = 0; i < steps; i++) world.step(0);
            return (await captureTexture(world, camera)).rgba;
        };
        const luminance = (rgba: Uint8ClampedArray, at: number) =>
            0.2126 * decode(rgba[at]!) +
            0.7152 * decode(rgba[at + 1]!) +
            0.0722 * decode(rgba[at + 2]!);
        await frame(null, 64);
        const added: number[] = [];
        const maxima: number[] = [];
        for (const [position, centerX] of [
            [[-1, 1.05, -3, 0], 537],
            [[1.8, 1.05, -3, 0], 825],
        ] as const) {
            const on = await frame(position, 1);
            const off = await frame(null, 1);
            let peak = 0;
            let max = 0;
            // Search the projected mote's footprint, not a shaft or wall pixel elsewhere.
            for (let y = 422; y <= 432; y++) {
                for (let x = centerX - 5; x <= centerX + 5; x++) {
                    const at = (y * 1280 + x) * 4;
                    peak = Math.max(peak, luminance(on, at) - luminance(off, at));
                    max = Math.max(max, on[at]!, on[at + 1]!, on[at + 2]!);
                }
            }
            added.push(peak);
            maxima.push(max);
        }
        console.log("mote contrast", { added, maxima, ratio: added[0]! / added[1]! });
        expect(added[1]!).toBeGreaterThan(0);
        expect(added[0]!).toBeGreaterThanOrEqual(5 * added[1]!);
        for (const max of maxima) expect(max).toBeLessThan(255);
    } finally {
        app.dispose();
    }
});
