import { setDefaultTimeout, test } from "bun:test";
import { Meshes } from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "../tests/gpu-app";
import { Terrain, TerrainPlugin } from "./terrain";
import { CONFIG } from "./terrain-data";

setDefaultTimeout(TIMEOUT.node);
test("report the shipped terrain's fenced step time without a performance assertion", async () => {
    const app = await openGpuApp({ plugins: [TerrainPlugin] });
    try {
        const { world } = app;
        const terrain = world.resource(Terrain);
        attachTexture(world, terrain.camera, { width: 1280, height: 720 });
        for (let i = 0; i < 30; i++) {
            world.step(1 / 60);
            await world.gpu.sync();
        }
        const turnovers = terrain.turnovers;
        const start = performance.now();
        const frames = 180;
        for (let i = 0; i < frames; i++) {
            world.step(1 / 60);
            await world.gpu.sync();
        }
        console.log({
            adapter: world.gpu.adapter,
            config: CONFIG,
            resolution: "1280x720 offscreen",
            frames,
            meanFencedStepMs: (performance.now() - start) / frames,
            loadedChunks: terrain.slots.length,
            meshes: world.resource(Meshes).size,
            turnovers: terrain.turnovers - turnovers,
        });
    } finally {
        app.dispose();
    }
});
