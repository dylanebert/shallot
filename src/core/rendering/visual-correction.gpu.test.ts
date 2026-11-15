import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { probeBuffer, Time } from "../../engine";
import { GlobalTransform, Transform } from "../transform";
import { globalTransformTable, readRenderGlobalTransform } from "./global-transform";
import {
    attachTexture,
    Camera,
    PointLight,
    PrepareViewsSystem,
    RenderingPlugin,
    VisualCorrection,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
let eid = -1;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            RenderingPlugin,
            {
                name: "Motion",
                dependencies: [RenderingPlugin],
                systems: [
                    {
                        group: "fixed",
                        update(world) {
                            world
                                .storage(Transform)
                                .translation.set(eid, world.time.fixedTick * 10, 2, 3);
                            world
                                .storage(Transform)
                                .rotation.set(eid, Math.SQRT1_2, 0, 0, Math.SQRT1_2);
                        },
                    },
                ],
                initialize(world) {
                    eid = world.create();
                    world.add(eid, Transform);
                    world.add(eid, PointLight);
                    const camera = world.create();
                    world.add(camera, Transform);
                    world.add(camera, Camera);
                    attachTexture(world, camera, { width: 8, height: 8 });
                    globalTransformTable(world);
                },
            },
        ],
    },
]);

function offset() {
    const world = subjects()[0].world;
    world.add(eid, VisualCorrection);
    world.storage(VisualCorrection).translation.set(eid, 4, -2, 1);
    world.storage(VisualCorrection).rotation.set(eid, 0, 0, Math.SQRT1_2, Math.SQRT1_2);
}

test("GPU placement applies world-space visual correction after fractional interpolation and removes it without accumulation", async () => {
    const world = subjects()[0].world;
    world.step(Time.FIXED_DT * 2.5);
    const table = globalTransformTable(world);
    const read = async () =>
        new Float32Array(
            (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
        ).slice(table.rowIndex(eid) * 12, table.rowIndex(eid) * 12 + 12);
    const plain = await read();
    expect(plain[0]).toBeCloseTo(15, 5);
    offset();
    world.step(0);
    const corrected = await read();
    expect(Array.from(corrected.slice(0, 3))).toEqual([19, 0, 4]);
    for (const i of [4, 5, 6, 7]) expect(corrected[i]).toBeCloseTo(0.5, 5);
    expect(Array.from(corrected.slice(8, 11))).toEqual([1, 1, 1]);
    world.step(0);
    expect(Array.from(await read())).toEqual(Array.from(corrected));
    world.remove(eid, VisualCorrection);
    world.step(0);
    expect(Array.from(await read())).toEqual(Array.from(plain));
});

test("CPU light placement applies the same post-interpolation visual correction", () => {
    const world = subjects()[0].world;
    offset();
    world.step(Time.FIXED_DT * 2.5);
    const out = new Float32Array(12);
    readRenderGlobalTransform(world, eid, out);
    expect(out[0]).toBeCloseTo((world.time.fixedTick - 1 + world.time.fixedAlpha) * 10 + 4, 5);
    expect(Array.from(out.slice(1, 3))).toEqual([0, 4]);
    for (const i of [4, 5, 6, 7]) expect(out[i]).toBeCloseTo(0.5, 5);
});

test("fixed ticks never incorporate visual correction into GlobalTransform", () => {
    const world = subjects()[0].world;
    offset();
    world.tick();
    const storage = world.storage(GlobalTransform);
    expect(storage.translation.x.get(eid)).toBe(world.time.fixedTick * 10);
    expect(storage.translation.y.get(eid)).toBe(2);
    expect(storage.translation.z.get(eid)).toBe(3);
    expect(storage.rotation.x.get(eid)).toBeCloseTo(Math.SQRT1_2, 5);
    expect(storage.rotation.y.get(eid)).toBe(0);
    expect(storage.rotation.z.get(eid)).toBe(0);
    expect(storage.rotation.w.get(eid)).toBeCloseTo(Math.SQRT1_2, 5);
    world.remove(eid, VisualCorrection);
});

test("a CPU follower enrolls history for a non-light entity and reads the corrected GPU placement", async () => {
    const world = subjects()[0].world;
    const target = world.create();
    world.add(target, Transform);
    world.add(target, VisualCorrection);
    world.addSystem({
        group: "draw",
        first: true,
        before: [PrepareViewsSystem],
        update(world) {
            world.storage(VisualCorrection).translation.set(target, 4, 0, 0);
        },
    });
    world.step(0);
    const out = new Float32Array(12);
    readRenderGlobalTransform(world, target, out);
    expect(out[0]).toBe(4);
    world.addSystem({
        group: "fixed",
        update(world) {
            world.storage(Transform).translation.set(target, 20, 0, 0);
        },
    });
    world.tick();
    world.step(0);
    readRenderGlobalTransform(world, target, out);
    expect(out[0]).toBeCloseTo(20 * world.time.fixedAlpha + 4, 5);
    const table = globalTransformTable(world);
    const gpu = new Float32Array(
        (await probeBuffer(world, table.buffer, { size: table.buffer.size })).bytes,
    );
    expect(out[0]).toBeCloseTo(gpu[table.rowIndex(target) * 12]!, 5);
});

test("cleanup", async () => {
    await disposeGpuApps(subjects());
});
