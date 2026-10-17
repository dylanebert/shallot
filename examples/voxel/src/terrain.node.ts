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
import { columnHash, DetailKind, detailCount, detailKind } from "./detail";
import { faces, height } from "./noise";
import { Terrain, terrainPlugin } from "./terrain";
import { inRing, radii, slotCount, type TerrainConfig, voxelScale } from "./terrain-data";

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
    const detailCounts = new Uint32Array((await probeBuffer(world, s.detailCounts!)).bytes);
    const details = new Uint32Array((await probeBuffer(world, s.details!)).bytes);
    const table = s.placement!.table;
    const count = new Uint32Array(
        (await probeBuffer(world, world.gpu.root.unwrap(table.count))).bytes,
    )[0];
    const placements = new Float32Array(
        (await probeBuffer(world, world.gpu.root.unwrap(table.placements))).bytes,
    );
    const instances = new Uint32Array(
        (await probeBuffer(world, world.gpu.root.unwrap(table.meshInstances))).bytes,
    );
    // Kind-to-part selection is independent of the GPU part rows.
    const parts = {
        [DetailKind.none]: [],
        [DetailKind.grass]: [0],
        [DetailKind.amber]: [1, 2],
        [DetailKind.cyan]: [1, 3],
        [DetailKind.rock]: [4],
        [DetailKind.tree]: [5, 6],
    };
    const viewer = s.previousViewer + (s.viewer - s.previousViewer) * world.time.fixedAlpha;
    const { inner, outer } = radii(config);
    const anchors: number[][] = [];
    let annulus = 0;
    expect(s.slots).toHaveLength(slotCount(config));
    // MeshPlugin's built-in cube is registered even when no entity draws it.
    expect(world.resource(Meshes).size).toBe(s.slots.length + 8);
    expect(new Set(s.slots.map((slot) => slot.geometry!.allFaces))).toEqual(new Set([false, true]));
    for (const [i, slot] of s.slots.entries()) {
        expect(slot.geometry!.allFaces).toBe(inRing(slot.x, slot.z, s.viewer, config));
        expect(counts[i] / 6).toBe(faces(slot.x, slot.z, config, slot.geometry!.allFaces));
        expect(counts[i]).toBeLessThan(slot.geometry!.mesh.indexCount);
        expect(detailCounts[i]).toBe(detailCount(slot.x, slot.z, config));
        const expected: string[] = [];
        for (let z = 0; z < config.chunkSize; z++) {
            for (let x = 0; x < config.chunkSize; x++) {
                const wx = slot.x * config.chunkSize + x;
                const wz = slot.z * config.chunkSize + z;
                const kind = detailKind(wx, wz, config);
                if (kind !== DetailKind.none) {
                    const cx = wx + 0.5 - viewer,
                        cz = wz + 0.5;
                    const scale = voxelScale(Math.hypot(cx, cz), inner, outer);
                    if (scale > 0) {
                        if (scale < 1) annulus++;
                        const hash = columnHash(wx, wz);
                        const jx =
                            kind === DetailKind.tree
                                ? 0
                                : ((hash & 255) / 255 - 0.5) * 0.35 * scale;
                        const jz =
                            kind === DetailKind.tree
                                ? 0
                                : (((hash >>> 8) & 255) / 255 - 0.5) * 0.35 * scale;
                        const yaw =
                            kind === DetailKind.tree ? 0 : ((hash >>> 16) / 65535) * Math.PI * 2;
                        for (const part of parts[kind]) {
                            const material = s.placement!.materials[part];
                            anchors.push([
                                cx + jx,
                                height(wx, wz, config) - 0.5 - config.height / 2 + 0.5 * scale,
                                cz + jz,
                                scale,
                                scale,
                                scale,
                                0,
                                Math.sin(yaw / 2),
                                0,
                                Math.cos(yaw / 2),
                                s.placement!.meshes[part],
                                material.type,
                                material.material,
                            ]);
                        }
                    }
                    expected.push(
                        `${x},${height(wx, wz, config)},${z},${kind},${columnHash(wx, wz)}`,
                    );
                }
            }
        }
        const actual: string[] = [];
        for (let j = 0; j < detailCounts[i]; j++) {
            const offset = (i * config.chunkSize ** 2 + j) * 2;
            const packed = details[offset];
            actual.push(
                `${packed & 255},${(packed >>> 8) & 255},${(packed >>> 16) & 255},${packed >>> 24},${details[offset + 1]}`,
            );
        }
        expect(actual.sort()).toEqual(expected.sort());
    }
    expect(annulus).toBeGreaterThan(0);
    expect(count, "GPU producer count at interpolated viewer").toBe(anchors.length);
    const actual = Array.from({ length: count }, (_, i) => [
        placements[i * 12],
        placements[i * 12 + 1],
        placements[i * 12 + 2],
        placements[i * 12 + 8],
        placements[i * 12 + 9],
        placements[i * 12 + 10],
        placements[i * 12 + 4],
        placements[i * 12 + 5],
        placements[i * 12 + 6],
        placements[i * 12 + 7],
        instances[i * 12],
        instances[i * 12 + 1],
        instances[i * 12 + 2],
    ]);
    const order = (a: number[], b: number[]) =>
        a[0] - b[0] || a[2] - b[2] || a[10] - b[10] || a[11] - b[11] || a[12] - b[12];
    actual.sort(order);
    anchors.sort(order);
    for (let i = 0; i < anchors.length; i++) {
        for (let lane = 0; lane < 10; lane++) {
            const field = lane < 3 ? "anchor" : lane < 6 ? "scale" : "quaternion";
            expect(actual[i][lane], `producer ${field} row ${i} lane ${lane}`).toBeCloseTo(
                anchors[i][lane],
                4,
            );
        }
        expect(actual[i].slice(10), `producer mesh/material row ${i}`).toEqual(
            anchors[i].slice(10),
        );
    }
}

test("small-world GPU counts match the oracle after fixed ticks and streaming; 100 warm turnovers create no buffers or bind groups", async () => {
    const world = app.world,
        s = world.resource(Terrain);
    for (let i = 0; i < 120; i++) world.tick();
    world.step(1 / 240);
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
            expect(world.resource(Meshes).size).toBe(s.slots.length + 8);
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
    for (const viewer of [2 ** 24 + 13.25, -(2 ** 24) - 7.5]) {
        s.viewer = viewer;
        world.tick();
        world.step(1 / 240);
        await countsMatch();
    }
});
