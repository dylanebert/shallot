import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance } from "../../core/mesh";
import { attachTexture, Camera } from "../../core/rendering";
import { Transform } from "../../core/transform";
import {
    AlphaMode,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] },
]);
test("a later view growing from one to forty transparent count runs preserves the earlier view's encoded table", async () => {
    const { world } = subjects()[0]!;
    const meshes = world.resource(Meshes);
    const geometry = meshes.values().next().value!;
    const count = world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    world.own(count);
    world.gpu.device.queue.writeBuffer(count, 0, new Uint32Array([3]));
    const material = world
        .resource(Materials)
        .add(StandardMaterial({ emissive: [1, 1, 1] }), { alphaMode: AlphaMode.Blend });
    for (let i = 0; i < 40; i++) {
        const mesh = meshes.add({
            ...geometry,
            name: `run-${i}`,
            indirectIndexCount: { buffer: count },
        });
        const eid = world.create();
        world.add(eid, Transform, { translation: [i * 10, 0, 0, 0] });
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, MeshMaterial, material);
    }
    for (const [x, size] of [
        [0, 3],
        [200, 1000],
    ]) {
        const camera = world.create();
        world.add(camera, Transform, { translation: [x!, 0, 5, 0] });
        world.add(camera, Camera, { mode: 1, size: size! });
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 96, height: 96 });
    }
    const device = world.gpu.device,
        create = device.createBuffer.bind(device);
    const sizes: number[] = [];
    device.createBuffer = (descriptor) => {
        if (descriptor.label === "shallot-transparent-count-runs") sizes.push(descriptor.size);
        return create(descriptor);
    };
    device.pushErrorScope("validation");
    try {
        world.step(0);
    } finally {
        device.createBuffer = create;
    }
    const error = await device.popErrorScope();
    expect(sizes).toEqual([256, 512]);
    expect(error?.message).toBeUndefined();
});
