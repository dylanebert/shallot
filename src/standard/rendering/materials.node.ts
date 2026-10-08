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
import { Surfaces } from "./contract";
import { StandardRenderer } from "./forward";
import {
    MaterialInput,
    Materials,
    MeshMaterial,
    materialTable,
    StandardMaterial,
} from "./material";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS] },
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);

test("anonymous materials preserve every built-in surface frame including coloured emission", async () => {
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
    for (const surface of ["default", "unlit", "vertex"]) {
        const material = world.resource(Materials).add(
            StandardMaterial({
                surface: world.resource(Surfaces).id(surface)!,
                baseColor: [0.25, 0.5, 0.75, 1],
                metallic: 0.25,
                perceptualRoughness: 0.5,
                emissive: [0.03125, 0.0625, 0.09375],
                occlusion: 0.75,
            }),
        );
        world.storage(MeshMaterial).material.set(eid, material);
        world.gpu.device.pushErrorScope("validation");
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        expect(await world.gpu.device.popErrorScope()).toBeNull();
        const colors = new Set<string>();
        for (let i = 0; i < rgba.length; i += 4)
            colors.add(`${rgba[i]},${rgba[i + 1]},${rgba[i + 2]}`);
        expect(colors.size).toBeGreaterThan(3);
        const directory = process.env.SHALLOT_MATERIAL_FRAMES;
        if (directory) {
            await mkdir(directory, { recursive: true });
            const path = `${directory}/${surface}.rgba`;
            if (process.env.SHALLOT_RECORD_MATERIAL_FRAMES) await writeFile(path, rgba);
            else expect(Buffer.from(rgba).equals(await readFile(path))).toBe(true);
        }
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
    const materials = world.resource(Materials);
    const material = materials.add(StandardMaterial({ metallic: 0.25 }));
    world.add(eid, MeshMaterial, { material });
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

test("anonymous adds return distinct ids and partial updates preserve the other GPU values", async () => {
    const { world } = subjects()[1];
    const materials = world.resource(Materials);
    const values = StandardMaterial({ metallic: 0.25, occlusion: 0.75 });
    const a = materials.add(values);
    const b = materials.add(values);
    expect(b).not.toBe(a);
    materials.update(a, {
        baseColor: [0.75, 0.25, 0.5, 1],
        emissive: [0.5, 0.25, 0.125],
        diffuseWrap: 0,
    });
    expect(() => materials.update(-1, {})).toThrow("unknown material id");
    expect(() => materials.update(b + 1, {})).toThrow("unknown material id");
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    async function read(id: number) {
        const snapshot = await probeBuffer(world, materialTable(world).buffer, {
            offset: id * d.sizeOf(MaterialInput),
            size: d.sizeOf(MaterialInput),
        });
        return readFromArrayBuffer(snapshot.bytes, MaterialInput);
    }
    const changed = await read(a);
    const unchanged = await read(b);
    expect(changed.baseColor).toEqual(d.vec4f(0.75, 0.25, 0.5, 1));
    expect(changed.params).toEqual(d.vec4f(0.25, 0.5, a, 0.75));
    expect(changed.emissive).toEqual(d.vec3f(0.5, 0.25, 0.125));
    expect(changed.diffuseWrap).toBe(0);
    expect(unchanged.baseColor).toEqual(d.vec4f(1));
    expect(unchanged.params).toEqual(d.vec4f(0.25, 0.5, b, 0.75));
    expect(unchanged.emissive).toEqual(d.vec3f(0));
    expect(unchanged.diffuseWrap).toBe(1);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
});
