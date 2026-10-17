import { afterAll, beforeAll, expect, setDefaultTimeout, spyOn, test } from "bun:test";
import {
    type createApp,
    lookAtRotation,
    Meshes,
    MeshInstance,
    probeBuffer,
    Transform,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { faces } from "./noise";
import { Terrain, terrainPlugin } from "./terrain";
import { inRing, slotCount, type TerrainConfig } from "./terrain-data";

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

test("orbit opens around the terrain centre and stays put while terrain scrolls", () => {
    const world = app.world;
    const camera = world.resource(Terrain).camera;
    const span = (config.viewDistance * 2 + 1) * config.chunkSize;
    const position = [
        span * 0.99 * Math.cos(0.75) * 0.6,
        span * 0.99 * Math.sin(0.75),
        span * 0.99 * Math.cos(0.75) * 0.8,
    ] as const;
    const target = [0, 0, 0] as const;
    const rotation = lookAtRotation(...position, ...target);
    const orbit = world.storage(Orbit);
    for (const [i, lane] of [orbit.pan.x, orbit.pan.y, orbit.pan.z].entries())
        expect(lane.get(camera)).toBeCloseTo(target[i], 5);
    expect(orbit.distance.get(camera)).toBeCloseTo(span * 0.99, 4);
    expect(orbit.yaw.get(camera)).toBeCloseTo(Math.atan2(0.9, 1.2), 6);
    expect(orbit.pitch.get(camera)).toBeCloseTo(0.75, 6);
    const assertPose = () => {
        const transform = world.storage(Transform);
        const { translation: t, rotation: r } = transform;
        for (const [i, lane] of [t.x, t.y, t.z].entries())
            expect(lane.get(camera)).toBeCloseTo(position[i], 4);
        for (const [i, lane] of [r.x, r.y, r.z, r.w].entries())
            expect(lane.get(camera)).toBeCloseTo(rotation[i], 6);
    };
    assertPose();
    const viewer = world.resource(Terrain).viewer;
    world.step(1 / 60);
    expect(world.resource(Terrain).viewer).toBeGreaterThan(viewer);
    assertPose();
});

async function countsMatch() {
    const world = app.world,
        s = world.resource(Terrain);
    const counts = new Uint32Array((await probeBuffer(world, s.counts!)).bytes);
    expect(s.slots).toHaveLength(slotCount(config));
    // MeshPlugin's built-in cube is registered even when no entity draws it.
    expect(world.resource(Meshes).size).toBe(s.slots.length + 1);
    expect(new Set(s.slots.map((slot) => slot.geometry!.allFaces))).toEqual(new Set([false, true]));
    for (const [i, slot] of s.slots.entries()) {
        expect(slot.geometry!.allFaces).toBe(inRing(slot.x, slot.z, s.viewer, config));
        expect(counts[i] / 6).toBe(faces(slot.x, slot.z, config, slot.geometry!.allFaces));
        expect(counts[i]).toBeLessThan(slot.geometry!.mesh.indexCount);
    }
}

test("small-world GPU counts match the oracle after fixed ticks and streaming; 100 warm turnovers create no buffers or bind groups", async () => {
    const world = app.world,
        s = world.resource(Terrain);
    for (let i = 0; i < 120; i++) world.tick();
    world.step(0);
    await countsMatch();
    const coordinates = s.slots.map((slot) => slot.x);
    const rules = s.slots.map((slot) => slot.geometry!.allFaces);
    s.viewer += config.chunkSize / 2;
    world.tick();
    world.step(0);
    expect(s.slots.map((slot) => slot.x)).toEqual(coordinates);
    expect(s.slots.some((slot, i) => slot.geometry!.allFaces !== rules[i])).toBe(true);
    await countsMatch();
    for (let i = 0; i < 4; i++) {
        s.viewer += config.chunkSize;
        world.tick();
        world.step(0);
        await world.gpu.sync();
    }
    const snapshot = world.snapshot();
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
    world.restore(snapshot);
    world.step(0);
    await countsMatch();
    for (const slot of s.slots)
        expect(world.storage(MeshInstance).mesh.get(slot.eid)).toBe(slot.instance.mesh);
});
