import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { type createApp, Meshes, probeBuffer } from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "../tests/gpu-app";
import { faces } from "./noise";
import { Terrain, terrainPlugin } from "./terrain";
import { slotCount, type TerrainConfig } from "./terrain-data";

setDefaultTimeout(TIMEOUT.node);
const config: TerrainConfig = { chunkSize: 8, viewDistance: 1, height: 16, relief: 4 };
let app: Awaited<ReturnType<typeof createApp>>;
beforeAll(async () => {
    app = await openGpuApp({ plugins: [terrainPlugin(config)] });
    console.log("voxel adapter", app.world.gpu.adapter);
    attachTexture(app.world, app.world.resource(Terrain).camera, { width: 16, height: 16 });
    app.world.step(1 / 60);
});
afterAll(() => app?.dispose());

async function countsMatch() {
    const world = app.world,
        s = world.resource(Terrain);
    const counts = new Uint32Array((await probeBuffer(world, s.counts!)).bytes);
    expect(s.slots).toHaveLength(slotCount(config));
    // MeshPlugin's built-in cube is registered even when no entity draws it.
    expect(world.resource(Meshes).size).toBe(s.slots.length + 1);
    for (const [i, slot] of s.slots.entries())
        expect(counts[i] / 6).toBe(faces(slot.x, slot.z, config));
}

test("small-world GPU counts match the oracle after fixed ticks and streaming; 100 warm turnovers create no buffers or bind groups", async () => {
    const world = app.world,
        s = world.resource(Terrain);
    for (let i = 0; i < 120; i++) world.tick();
    world.step(0);
    await countsMatch();
    for (let i = 0; i < 4; i++) {
        s.viewer += config.chunkSize;
        world.tick();
        world.step(0);
        await world.gpu.sync();
    }
    const start = s.turnovers;
    const buffers = spyOn(world.gpu.device, "createBuffer");
    const groups = spyOn(world.gpu.device, "createBindGroup");
    try {
        while (s.turnovers - start < 100) {
            s.viewer += config.chunkSize;
            world.tick();
            world.step(0);
            await world.gpu.sync();
            expect(world.resource(Meshes).size).toBe(s.slots.length + 1);
        }
        expect(buffers).not.toHaveBeenCalled();
        expect(groups).not.toHaveBeenCalled();
    } finally {
        buffers.mockRestore();
        groups.mockRestore();
    }
    await countsMatch();
});
