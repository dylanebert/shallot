import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Welcome } from "./index";

setDefaultTimeout(TIMEOUT.node);
test("the welcome scene draws visible pixels above its dark clear", async () => {
    const app = await openGpuApp({
        plugins: [
            {
                name: "CaptureWelcome",
                dependencies: [Welcome],
                initialize(world) {
                    const [camera] = world.query([Camera]);
                    attachTexture(world, camera!, { width: 160, height: 90 });
                },
            },
        ],
    });
    try {
        const { world } = app;
        const [camera] = world.query([Camera]);
        world.step(0);
        const { rgba } = await captureTexture(world, camera!);
        let visible = 0;
        for (let i = 0; i < rgba.length; i += 4)
            if (rgba[i + 3]! > 0 && (rgba[i]! > 35 || rgba[i + 1]! > 35 || rgba[i + 2]! > 35))
                visible++;
        expect(visible).toBeGreaterThan(100);
    } finally {
        app.dispose();
    }
});
