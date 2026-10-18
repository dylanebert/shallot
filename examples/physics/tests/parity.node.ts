import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { createScene, type Scene, setGravityScale } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

for (const scene of ["boxStack", "bridge"] satisfies Scene[]) {
    test(`${scene} equals every recorded native sample step and rejects a density change`, async () => {
        const gold = (await Bun.file(new URL(`./${scene}.gold.txt`, import.meta.url)).text())
            .trim()
            .split("\n")
            .map((line) => BigInt(line.split(" ")[1]!));
        const start = performance.now();
        for (const mutated of [false, true]) {
            let bodies: number[] = [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin],
                setup(world) {
                    bodies = createScene(world, scene);
                },
            });
            try {
                if (mutated)
                    app.world
                        .storage(Shape)
                        .density.set(bodies[0]!, scene === "bridge" ? 21 : 1001);
                let diverged = false;
                for (const expected of gold) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (!mutated) expect(hash).toBe(expected);
                    else if (hash !== expected) {
                        diverged = true;
                        break;
                    }
                }
                if (mutated) expect(diverged).toBe(true);
                if (!mutated && scene === "bridge") {
                    setGravityScale(app.world, bodies, -0.5);
                    app.world.tick();
                    const gravity = app.world.storage(Body).gravityScale;
                    for (const eid of app.world.query([Body]))
                        expect(gravity.get(eid)).toBe(bodies.includes(eid) ? -0.5 : 1);
                }
            } finally {
                app.dispose();
            }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms; cheap budget 250 ms`);
    });
}
