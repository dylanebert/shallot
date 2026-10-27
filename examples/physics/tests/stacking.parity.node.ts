import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, Shape } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createScene, type Scene } from "../src/scenes";
import { StackingSamplePlugin, stackingScenes } from "../src/stacking";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);

test("creation impulses recover before the first solve and apply only once", async () => {
    const gold = sampleGold(
        await Bun.file(new URL("./doubleDomino.gold.txt", import.meta.url)).text(),
    );
    const app = await createApp({
        defaults: false,
        plugins: [StackingSamplePlugin],
        setup(world) {
            createScene(world, "doubleDomino");
        },
    });
    try {
        const pending = app.world.snapshot();
        for (let replay = 0; replay < 2; replay++) {
            if (replay) app.world.restore(pending);
            for (const expected of gold.slice(0, 2)) {
                app.world.tick();
                expect(hashPhysics(app.world)).toBe(expected.hash);
            }
        }
    } finally {
        app.dispose();
    }
});

for (const scene of ["boxStack", ...Object.keys(stackingScenes)] as Scene[]) {
    test(`${scene} equals every native step and rejects a density change`, async () => {
        const gold = sampleGold(
            await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text(),
        );
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [StackingSamplePlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                if (mutated) app.world.storage(Shape).density.set(bodies[0]!, 1001);
                let diverged = false;
                for (const [step, expected] of gold.entries()) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (!mutated) expect(hash, `step ${step} hash`).toBe(expected.hash);
                    else if (hash !== expected.hash) {
                        diverged = true;
                        break;
                    }
                }
                if (mutated) expect(diverged).toBe(true);
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
