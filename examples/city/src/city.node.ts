import { expect, setDefaultTimeout, test } from "bun:test";
import { Materials, Meshes, MeshInstance } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "../tests/gpu-app";
import { City, COLORS, cityPlugin } from "./city";
import { layout, POPULATION, SEED } from "./layout";

setDefaultTimeout(TIMEOUT.node);

test("doubling a small population preserves kit sizes and exact per-mesh group counts", async () => {
    for (const population of [128, 256]) {
        const app = await openGpuApp({ plugins: [cityPlugin(SEED, population)] });
        try {
            const { world } = app;
            console.info("city adapter", world.gpu.adapter);
            const placements = layout(SEED, population);
            expect([...world.query([MeshInstance])]).toHaveLength(placements.length);
            expect(world.resource(Meshes).size).toBe(2);
            const groups = new Map<number, number>();
            for (const eid of world.query([MeshInstance])) {
                const mesh = world.storage(MeshInstance).mesh.get(eid);
                groups.set(mesh, (groups.get(mesh) ?? 0) + 1);
            }
            expect(groups.size).toBe(2);
            expect([...groups.values()].sort((a, b) => a - b)).toEqual([
                placements.filter((p) => p.part === "dome").length,
                placements.filter((p) => p.part === "block").length,
            ]);
            // Row zero is the renderer's built-in default material.
            expect(world.resource(Materials).table.highWater).toBe(COLORS.length + 1);
        } finally {
            app.dispose();
        }
    }
});

test("a fixed camera draws the default population's buildings across the centre", async () => {
    const app = await openGpuApp({ plugins: [cityPlugin()] });
    try {
        const { world } = app;
        console.info("city frame adapter", world.gpu.adapter);
        expect([...world.query([MeshInstance])]).toHaveLength(POPULATION);
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
});
