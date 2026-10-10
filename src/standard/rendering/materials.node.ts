import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { readFromArrayBuffer } from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { MeshInstance } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    PointLight,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { probeBuffer } from "../../engine/runtime";
import { DEFAULT_PLUGINS, MeshRenderPlugin, StandardRenderingPlugin } from "../index";
import { StandardRenderer } from "./forward";
import {
    Materials,
    MeshMaterial,
    materialTable,
    StandardMaterial,
    StandardMaterialInput,
} from "./material";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS] },
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);

test("StandardMaterial keeps its frame while material types own independent parameter tables", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 64 });
    world.add(world.create(), AmbientLight, { intensity: 0.2 });
    world.add(world.create(), DirectionalLight, { direction: [-0.4, -0.8, -0.5, 0] });
    const point = world.create();
    world.add(point, Transform, { translation: [1, 1, 2, 0] });
    world.add(point, PointLight, { intensity: 8, range: 10 });
    const eid = world.create();
    world.add(eid, Transform, { rotation: [0.0996005, 0.199201, 0, 0.974884] });
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial);
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const before = await captureTexture(world, camera);
    expect(await world.gpu.device.popErrorScope()).toBeNull();

    // Adding another type's rows must not move or rewrite StandardMaterial's row zero.
    const standard = world.resource(Materials);
    const standardHandle = standard.add(
        StandardMaterial({
            baseColor: [0.25, 0.5, 0.75, 1],
            metallic: 0.25,
            perceptualRoughness: 0.5,
            emissive: [0.03125, 0.0625, 0.09375],
            occlusion: 0.75,
        }),
    );
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const unchangedDefault = await captureTexture(world, camera);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
    expect(unchangedDefault.rgba).toEqual(before.rgba);

    world.storage(MeshMaterial).type.set(eid, standardHandle.type);
    world.storage(MeshMaterial).material.set(eid, standardHandle.material);
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    world.step(0);
    const after = await captureTexture(world, camera);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
    expect(after.rgba).not.toEqual(before.rgba);
    const directory = process.env.SHALLOT_MATERIAL_FRAMES;
    if (directory) {
        await mkdir(directory, { recursive: true });
        const path = `${directory}/standard.rgba`;
        if (process.env.SHALLOT_RECORD_MATERIAL_FRAMES) await writeFile(path, after.rgba);
        else expect(Buffer.from(after.rgba).equals(await readFile(path))).toBe(true);
    }
});

test("steady mesh rendering adds no materials and creates no bind groups", () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 16, height: 16 });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance);
    world.add(eid, MeshMaterial);
    const materials = world.resource(Materials);
    const material = materials.add(StandardMaterial({ metallic: 0.25 }));
    world.storage(MeshMaterial).material.set(eid, material.material);
    world.step(0);
    world.step(0);
    const add = materials.add.bind(materials);
    const createBindGroup = world.gpu.device.createBindGroup.bind(world.gpu.device);
    let additions = 0;
    let groups = 0;
    const values = { metallic: 0 };
    materials.add = (values) => {
        additions++;
        return add(values);
    };
    world.gpu.device.createBindGroup = (descriptor) => {
        groups++;
        return createBindGroup(descriptor);
    };
    try {
        for (let i = 0; i < 10; i++) world.step(0);
        expect(additions).toBe(0);
        expect(groups).toBe(0);
        for (let i = 0; i < 10; i++) {
            values.metallic = (i % 2) * 0.25;
            materials.update(material, values);
            world.step(0);
        }
        expect(additions).toBe(0);
        expect(groups).toBe(0);
    } finally {
        materials.add = add;
        world.gpu.device.createBindGroup = createBindGroup;
    }
});

test("type-owned adds return distinct rows and partial updates preserve the other GPU values", async () => {
    const { world } = subjects()[1];
    const materials = world.resource(Materials);
    const values = StandardMaterial({ metallic: 0.25, occlusion: 0.75 });
    const a = materials.add(values);
    const b = materials.add(values);
    expect(b.material).not.toBe(a.material);
    materials.update(a, {
        baseColor: [0.75, 0.25, 0.5, 1],
        emissive: [0.5, 0.25, 0.125],
        diffuseWrap: 0,
    });
    expect(() => materials.update(-1, {})).toThrow("unknown StandardMaterial row");
    expect(() => materials.update({ ...b, material: b.material + 1 }, {})).toThrow(
        "unknown StandardMaterial row",
    );
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    async function read(handle: typeof a) {
        const snapshot = await probeBuffer(world, materialTable(world).buffer, {
            offset: handle.material * d.sizeOf(StandardMaterialInput),
            size: d.sizeOf(StandardMaterialInput),
        });
        return readFromArrayBuffer(snapshot.bytes, StandardMaterialInput);
    }
    const changed = await read(a);
    const unchanged = await read(b);
    expect(changed.baseColor).toEqual(d.vec4f(0.75, 0.25, 0.5, 1));
    expect(changed.metallic).toBe(0.25);
    expect(changed.perceptualRoughness).toBe(0.5);
    expect(changed.occlusion).toBe(0.75);
    expect(changed.emissive).toEqual(d.vec3f(0.5, 0.25, 0.125));
    expect(changed.diffuseWrap).toBe(0);
    expect(unchanged.baseColor).toEqual(d.vec4f(1));
    expect(unchanged.emissive).toEqual(d.vec3f(0));
    expect(unchanged.diffuseWrap).toBe(1);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
});
