import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import {
    attachTexture,
    Camera,
    PointLight,
    SpotLight,
    VolumetricLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import { CLUSTER_COUNT, LIGHT_GRID_OFFSET, lightInputKey } from "./cluster";
import { StandardRenderer, StandardRenderingPlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [StandardRenderingPlugin] },
    { defaults: false, plugins: [StandardRenderingPlugin] },
    { defaults: false, plugins: [StandardRenderingPlugin] },
]);

test("render light inputs upload as active dense table rows", async () => {
    const app = subjects()[0];
    try {
        const { world } = app;
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, PointLight);
        world.storage(Transform).translation.set(eid, 2, 3, 4, 0);
        world.storage(PointLight).color.set(eid, 0xffd9a0);
        world.storage(PointLight).intensity.set(eid, 2.5);
        world.storage(PointLight).range.set(eid, 7);
        world.storage(PointLight).radius.set(eid, 0.25);

        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        await world.gpu.device.queue.onSubmittedWorkDone();
        expect(await world.gpu.device.popErrorScope()).toBeNull();
        const active = await probeBuffer(world, world.gpu.buffers.get("lightInputs:active-rows")!, {
            size: 8,
        });
        const [activeEid, row] = new Uint32Array(active.bytes);
        expect(activeEid).toBe(eid);
        const record = await probeBuffer(world, world.gpu.buffers.get("lightInputs")!, {
            offset: row * 32,
            size: 32,
        });
        const data = new DataView(record.bytes);
        expect(data.getFloat32(0, true)).toBe(Math.fround(0xffd9a0));
        expect(data.getFloat32(4, true)).toBe(2.5);
        expect(data.getFloat32(8, true)).toBe(7);
        expect(data.getFloat32(12, true)).toBe(0.25);
    } finally {
        app.dispose();
    }
});

test("empty clustered-light grids skip culling and clear once when their last light is removed", async () => {
    const app = subjects()[2];
    try {
        const { world } = app;
        const culls: string[] = [];
        world.gpu.span = (name) => {
            if (name === "light:cull") culls.push(name);
            return undefined;
        };
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 6, 0] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 8, height: 8 });

        const grid = async () => {
            const { bytes } = await probeBuffer(world, world.gpu.buffers.get("lightClusters")!, {
                offset: LIGHT_GRID_OFFSET,
                size: CLUSTER_COUNT * 8,
            });
            return new Uint32Array(bytes);
        };
        const empty = (values: Uint32Array) =>
            Array.from(values).every((value, index) => index % 2 === 0 || value === 0);

        world.step(0);
        expect(culls).toHaveLength(0);
        expect(empty(await grid())).toBe(true);

        const light = world.create();
        world.add(light, Transform, { translation: [0, 0, 3, 0] });
        world.add(light, PointLight, { intensity: 1, range: 20 });
        world.step(0);
        expect(culls).toHaveLength(1);
        expect(empty(await grid())).toBe(false);

        world.remove(light, PointLight);
        world.step(0);
        expect(culls).toHaveLength(1);
        expect(empty(await grid())).toBe(true);
        world.step(0);
        expect(culls).toHaveLength(1);
    } finally {
        app.dispose();
    }
});

test("a standalone spot light owns and updates a dense row, then releases it on removal", async () => {
    const { world } = subjects()[1];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, SpotLight, {
        color: 0xff8844,
        intensity: 3,
        range: 8,
        radius: 0.2,
        innerAngle: 20,
        outerAngle: 40,
    });
    expect(world.has(eid, PointLight)).toBe(false);
    const table = world.resource(lightInputKey);
    async function record() {
        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        const result = await probeBuffer(world, table.buffer, {
            offset: table.rowIndex(eid) * 32,
            size: 32,
        });
        expect(await world.gpu.device.popErrorScope()).toBeNull();
        return new DataView(result.bytes);
    }
    const initial = await record();
    expect(table.count).toBe(1);
    expect(initial.getFloat32(0, true)).toBe(Math.fround(0xff8844));
    expect(initial.getFloat32(4, true)).toBe(3);
    expect(initial.getFloat32(8, true)).toBe(8);
    expect(initial.getFloat32(12, true)).toBe(Math.fround(0.2));
    expect(initial.getFloat32(16, true)).toBe(20);
    expect(initial.getFloat32(20, true)).toBe(40);
    expect(initial.getUint32(24, true)).toBe(1);
    world.add(eid, PointLight, { intensity: 99 });
    expect((await record()).getFloat32(4, true)).toBe(3);
    expect(table.count).toBe(1);
    world.remove(eid, PointLight);
    world.storage(SpotLight).outerAngle.set(eid, 45);
    world.add(eid, VolumetricLight);
    const changed = await record();
    expect(changed.getFloat32(20, true)).toBe(45);
    expect(changed.getUint32(24, true)).toBe(3);
    world.remove(eid, SpotLight);
    world.step(0);
    expect(table.count).toBe(0);
    world.add(eid, PointLight, { intensity: 5 });
    const point = await record();
    expect(table.count).toBe(1);
    expect(point.getFloat32(4, true)).toBe(5);
    expect(point.getUint32(24, true)).toBe(2);
});
