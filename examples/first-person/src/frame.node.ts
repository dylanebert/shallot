import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    CharacterPlugin,
    createApp,
    PlayerPlugin,
    StandardPhysicsPlugin,
    Time,
} from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { CEILING } from "../../../scripts/test-tiers";
import { Demo } from "./demo";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("first-person presents a nonuniform final frame with byte-identical captures at one state", async () => {
    const app = await createApp({
        plugins: [StandardPhysicsPlugin, CharacterPlugin, PlayerPlugin, Demo],
    });
    try {
        const { adapter } = app.world.gpu;
        console.info(`first-person frame adapter: ${adapter.class} (${adapter.identity})`);
        const camera = [...app.world.query([Camera])][0];
        if (camera === undefined) throw new Error("first-person has no camera");
        // A small frame and one tick suffice to observe the composed final pass.
        attachTexture(app.world, camera, { width: 320, height: 180 });
        app.world.step(Time.FIXED_DT);
        const first = await captureTexture(app.world, camera);
        const second = await captureTexture(app.world, camera);
        expect(first.rgba.length).toBe(320 * 180 * 4);
        const pixel = first.rgba.subarray(0, 4);
        expect(first.rgba.some((value, index) => value !== pixel[index % 4])).toBe(true);
        expect(second.rgba).toEqual(first.rgba);
    } finally {
        app.dispose();
    }
});
