import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, Materials, Meshes, MeshInstance } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { City, COLORS, cityPlugin } from "./city";
import { layout, POPULATION, SEED } from "./layout";

setDefaultTimeout(TIMEOUT.node);
test("layout is seeded, nonempty and composed from the bounded kit", () => {
    const a = layout(SEED, POPULATION);
    expect(a).toHaveLength(POPULATION);
    expect(layout(SEED, POPULATION)).toEqual(a);
    expect(layout(SEED + 1, POPULATION)).not.toEqual(a);
    expect(new Set(a.map((p) => p.part))).toEqual(new Set(["block"]));
    expect(new Set(a.map((p) => p.material)).size).toBe(COLORS.length);
});
test("doubling parts preserves kit sizes and a fixed camera draws buildings across the centre", async () => {
    await setupGlobals();
    for (const population of [POPULATION, POPULATION * 2]) {
        const app = await createApp({ plugins: [cityPlugin(SEED, population)] });
        try {
            const { world } = app;
            console.info("city adapter", world.gpu.adapter);
            expect([...world.query([MeshInstance])]).toHaveLength(layout(SEED, population).length);
            expect(world.resource(Meshes).size).toBe(1);
            // Row zero is the renderer's built-in default material.
            expect(world.resource(Materials).table.highWater).toBe(COLORS.length + 1);
            const camera = world.resource(City).camera;
            attachTexture(world, camera, { width: 320, height: 180 });
            world.step(1 / 60);
            world.step(1 / 60);
            const frame = await captureTexture(world, camera);
            let covered = 0,
                accents = 0,
                total = 0;
            for (let y = 60; y < 110; y++)
                for (let x = 100; x < 220; x++) {
                    const i = (y * frame.width + x) * 4;
                    covered += Number(
                        Math.abs(frame.rgba[i] - 25) +
                            Math.abs(frame.rgba[i + 1] - 24) +
                            Math.abs(frame.rgba[i + 2] - 22) >
                            25,
                    );
                    const [r, g, b] = frame.rgba.slice(i, i + 3);
                    accents += Number(Math.max(r, g, b) - Math.min(r, g, b) > 35);
                    total++;
                }
            expect(covered / total).toBeGreaterThan(0.8);
            expect(accents / total).toBeGreaterThan(0.01);
        } finally {
            app.dispose();
        }
    }
});
