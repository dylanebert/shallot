import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import { cube, Meshes, MeshInstance, sphere } from "../../core/mesh";
import { attachTexture, Camera, RenderLayers } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import { StandardRenderer } from "../../standard/rendering";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 0, 5] });
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        attachTexture(world, camera, { width: 32, height: 32 });
        return camera;
    },
);

test("interleaved outline meshes draw first-seen groups with contiguous visible instance slices", () => {
    const { world, scene: camera } = subject();
    const meshes = world.resource(Meshes);
    const a = meshes.add({ name: "group-a", ...cube() });
    const b = meshes.add({ name: "group-b", ...sphere() });
    const entities = [a, b, a, b, a].map((mesh) => {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, MeshInstance, { mesh });
        world.add(eid, Outline);
        return eid;
    });
    world.add(entities[4]!, RenderLayers, { mask: 2 });
    expect(Array.from(world.query([Outline, MeshInstance]))).toEqual(entities);

    const device = world.gpu.device;
    const queue = device.queue;
    const writeBuffer = queue.writeBuffer.bind(queue);
    const createEncoder = device.createCommandEncoder.bind(device);
    const packed: number[] = [];
    const draws: number[][] = [];
    queue.writeBuffer = (buffer, offset, data, dataOffset, size) => {
        if (buffer.label === "outline-eids") {
            expect(offset).toBe(0);
            expect(data).toBeInstanceOf(Uint32Array);
            packed.push(...(data as Uint32Array).slice(dataOffset, dataOffset! + size!));
        }
        writeBuffer(buffer, offset, data, dataOffset, size);
    };
    device.createCommandEncoder = (descriptor) => {
        const encoder = createEncoder(descriptor);
        const begin = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (descriptor) => {
            const pass = begin(descriptor);
            if (descriptor.label === `outline-mask/${camera}`) {
                const draw = pass.draw.bind(pass);
                pass.draw = (vertices, instances, firstVertex, firstInstance) => {
                    draws.push([vertices, instances!, firstVertex!, firstInstance!]);
                    draw(vertices, instances, firstVertex, firstInstance);
                };
            }
            return pass;
        };
        return encoder;
    };
    try {
        world.step(0);
    } finally {
        queue.writeBuffer = writeBuffer;
        device.createCommandEncoder = createEncoder;
    }
    const meshA = meshes.get(a)!;
    const meshB = meshes.get(b)!;
    expect(draws).toEqual([
        [meshA.indexCount, 2, meshA.indexBase, 0],
        [meshB.indexCount, 2, meshB.indexBase, 2],
    ]);
    expect(packed).toEqual([entities[0]!, entities[2]!, entities[1]!, entities[3]!]);
});
