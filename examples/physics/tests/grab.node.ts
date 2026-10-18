import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { setTargetTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Grab, pick, release } from "../src/grab";
import { createScene } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("grab, moving kinematic target and release equal native Sample::MouseDown on every step", async () => {
    const gold = (await Bun.file(new URL("./grab.gold.txt", import.meta.url)).text())
        .trim()
        .split("\n")
        .map((line) => BigInt(line.split(" ")[1]!));
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            createScene(world, "boxStack");
        },
    });
    try {
        const world = app.world;
        const grab = world.resource(Grab);
        for (let i = 0; i < gold.length; i++) {
            if (i === 1) {
                Object.assign(grab.origin, { x: 0, y: 25.75, z: 10 });
                Object.assign(grab.translation, { x: 0, y: 0, z: -20 });
                pick(world);
                expect(grab.body).not.toBe(0);
            }
            if (i > 1 && i < 31) grab.point[0] = Math.fround(Math.fround(0.05) * (i - 1));
            if (i === 31) release(world);
            if (grab.mouse) setTargetTransform(world, grab.mouse, grab.point, [0, 0, 0, 1]);
            world.tick();
            expect(hashPhysics(world), `tick ${i}`).toBe(gold[i]!);
        }
    } finally {
        app.dispose();
    }
});
