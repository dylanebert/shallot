import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Mesh3d } from "../../core/mesh";
import {
    AmbientLight,
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    PointLight,
} from "../../core/rendering";
import { Transform } from "../../engine";
import { DEFAULT_PLUGINS, PartPlugin, StandardRenderingPlugin } from "../index";
import { Surfaces } from "./contract";
import { StandardRenderer } from "./forward";
import { Materials, MeshMaterial3d, StandardMaterial } from "./material";

setDefaultTimeout(CEILING.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS] },
    { defaults: false, plugins: [PartPlugin, StandardRenderingPlugin] },
]);

test("registered materials preserve every built-in surface frame including coloured emission", async () => {
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
    world.add(eid, Mesh3d);
    world.add(eid, MeshMaterial3d);
    for (const surface of ["default", "unlit", "vertex"]) {
        const material = world.resource(Materials).register({
            name: surface,
            ...StandardMaterial({
                surface: world.resource(Surfaces).id(surface)!,
                base_color: [0.25, 0.5, 0.75, 1],
                metallic: 0.25,
                perceptual_roughness: 0.5,
                emissive: [0.03125, 0.0625, 0.09375],
                occlusion: 0.75,
            }),
        });
        world.storage(MeshMaterial3d).material.set(eid, material);
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

test("steady mesh rendering registers no materials and creates no bind groups", () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 16, height: 16 });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Mesh3d);
    const materials = world.resource(Materials);
    const material = materials.register({
        name: "steady",
        ...StandardMaterial({ metallic: 0.25 }),
    });
    world.add(eid, MeshMaterial3d, { material });
    world.step(0);
    world.step(0);
    const register = materials.register.bind(materials);
    const createBindGroup = world.gpu.device.createBindGroup.bind(world.gpu.device);
    let registrations = 0;
    let groups = 0;
    materials.register = (record) => {
        registrations++;
        return register(record);
    };
    world.gpu.device.createBindGroup = (descriptor) => {
        groups++;
        return createBindGroup(descriptor);
    };
    try {
        for (let i = 0; i < 10; i++) world.step(0);
        expect(registrations).toBe(0);
        expect(groups).toBe(0);
    } finally {
        materials.register = register;
        world.gpu.device.createBindGroup = createBindGroup;
    }
});
