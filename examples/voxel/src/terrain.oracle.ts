import { setDefaultTimeout, test } from "bun:test";
import { Meshes, probeBuffer } from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Terrain, TerrainPlugin } from "./terrain";
import { CONFIG, ringSlots } from "./terrain-data";

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
        const meanFencedStepMs = (performance.now() - start) / frames;
        const indices = new Uint32Array((await probeBuffer(world, terrain.counts!)).bytes);
        let reservedGeometryBytes = 0;
        for (const { geometry } of terrain.slots) {
            reservedGeometryBytes += geometry.uniform.size;
            for (const buffer of [
                geometry.mesh.vertices,
                geometry.mesh.position!,
                geometry.mesh.indices,
                geometry.mesh.attributes!.voxel,
            ]) {
                reservedGeometryBytes += world.gpu.root.unwrap(buffer).size;
            }
        }
        console.log({
            adapter: world.gpu.adapter,
            config: CONFIG,
            resolution: "1280x720 offscreen",
            frames,
            meanFencedStepMs,
            reservedGeometryBytes,
            transitionChunks: terrain.slots.filter((slot) => slot.geometry!.allFaces).length,
            maxTransitionChunks: ringSlots(CONFIG),
            emittedFaces: indices.reduce((sum, count) => sum + count / 6, 0),
            loadedChunks: terrain.slots.length,
            detailRows: new Uint32Array(
                (await probeBuffer(world, world.gpu.root.unwrap(terrain.placement!.table.count)))
                    .bytes,
            )[0],
            meshes: world.resource(Meshes).size,
            turnovers: terrain.turnovers - turnovers,
        });
    } finally {
        app.dispose();
    }
});
