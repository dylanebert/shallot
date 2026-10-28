import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes, MeshInstance, MeshPlugin } from "../../core/mesh";
import { attachTexture, Camera, captureTexture, DirectionalLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { type AppConfig, precompile, type World } from "../../engine";
import {
    AlphaMode,
    Draws,
    Materials,
    MeshMaterial,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";
import { MeshDrawBuffers, MeshPreprocessSystem } from "./preprocess";

setDefaultTimeout(TIMEOUT.node);
const scenes = new WeakMap<World, ReturnType<typeof initialize>>();
const subjects = gpuApps(import.meta.path, [config(false), config(true)]);
function config(transparent: boolean): AppConfig {
    let original: ReturnType<typeof geometry>;
    let producer: { pipeline: GPUComputePipeline; group: GPUBindGroup } | undefined;
    return {
        defaults: false,
        plugins: [
            {
                name: "count-producer",
                dependencies: [MeshPlugin],
                initialize(world) {
                    original = geometry(world);
                },
                warm(world) {
                    const scene = initialize(world, original, transparent);
                    producer = scene.producer;
                    scenes.set(world, scene);
                },
                systems: [
                    {
                        group: "draw",
                        before: [MeshPreprocessSystem],
                        update(world) {
                            if (!producer) return;
                            const pass = world.frameEncoder()!.beginComputePass();
                            pass.setPipeline(producer.pipeline);
                            pass.setBindGroup(0, producer.group);
                            pass.dispatchWorkgroups(1);
                            pass.end();
                        },
                    },
                ],
            },
            MeshRenderPlugin,
            StandardRenderingPlugin,
        ],
    };
}
function geometry(world: World) {
    const meshes = world.resource(Meshes);
    const vertices: number[] = [];
    for (let i = 0; i < 4; i++) {
        const x = (i - 1.5) * 0.8;
        for (const [dx, y] of [
            [-0.3, -0.4],
            [0.3, -0.4],
            [0, 0.4],
        ])
            vertices.push(x + dx!, y!, 0, 0, 0, 1, 0, 0);
    }
    return meshes.add({
        name: "triangles",
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(Array.from({ length: 12 }, (_, i) => i)),
    });
}
function initialize(world: World, original: ReturnType<typeof geometry>, transparent: boolean) {
    const meshes = world.resource(Meshes);
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 4, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    const sun = world.create();
    world.add(sun, Transform);
    world.add(sun, DirectionalLight, { shadowMapsEnabled: 1 });
    attachTexture(world, camera, { width: 128, height: 64 });
    const buffer = world.gpu.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    world.own(buffer);
    // A nonzero byte offset ensures the count is not assumed to start at the buffer's origin.
    const mesh = meshes.add({
        ...meshes.get(original)!,
        name: "gpu-count",
        indirectIndexCount: { buffer, offset: 4 },
        indexCount: 12,
    });
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, MeshInstance, { mesh });
    world.add(
        eid,
        MeshMaterial,
        world.resource(Materials).add(StandardMaterial({ emissive: [1, 1, 1] })),
    );
    if (transparent)
        world.add(
            eid,
            MeshMaterial,
            world
                .resource(Materials)
                .add(StandardMaterial({ emissive: [1, 1, 1] }), { alphaMode: AlphaMode.Blend }),
        );
    const input = world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    world.own(input);
    const pipeline = world.gpu.device.createComputePipeline({
        label: "count-producer",
        layout: "auto",
        compute: {
            module: world.gpu.device.createShaderModule({
                code: "@group(0) @binding(0) var<storage, read_write> counts: array<u32>; @group(0) @binding(1) var<uniform> value: u32; @compute @workgroup_size(1) fn main() { counts[1] = value; }",
            }),
            entryPoint: "main",
        },
    });
    precompile(world, "count-producer", () => [pipeline]);
    const producer = {
        pipeline,
        group: world.gpu.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer } },
                { binding: 1, resource: { buffer: input } },
            ],
        }),
    };
    return { world, mesh, buffer, camera, eid, input, producer };
}
async function setup(index = 0) {
    const { world } = subjects()[index];
    world.step(0);
    return scenes.get(world)!;
}
let scene: Awaited<ReturnType<typeof setup>>;
async function draw(count: number) {
    scene ??= await setup();
    const { world, mesh } = scene;
    world.gpu.device.queue.writeBuffer(scene.input, 0, new Uint32Array([count]));
    const target = world.resource(MeshDrawBuffers).drawArgs;
    const raw = target && world.gpu.root.unwrap(target);
    const queue = world.gpu.device.queue;
    const write = queue.writeBuffer.bind(queue);
    let recordWrites = 0;
    queue.writeBuffer = (...args) => {
        if (args[0] === raw) recordWrites++;
        write(...args);
    };
    try {
        world.step(0);
    } finally {
        queue.writeBuffer = write;
    }
    // Registration seeds records once; changing the producer's count must not republish them.
    if (target === world.resource(MeshDrawBuffers).drawArgs) expect(recordWrites).toBe(0);
    const draw = [...world.resource(Draws)].find((draw) => draw.mesh === mesh)!;
    const records = (await draw.args.indirect.read()) as { indexCount: number }[];
    const stride = draw.args.viewStride! / 20;
    for (let at = draw.args.offset! / 20; at < records.length; at += stride)
        expect(records[at]!.indexCount).toBe(Math.min(count, 12));
    const { rgba } = await captureTexture(world, scene.camera);
    let triangles = 0;
    let inside = false;
    for (let x = 0; x < 128; x++) {
        let lit = false;
        for (let y = 0; y < 64; y++) if (rgba[(y * 128 + x) * 4]! > 128) lit = true;
        if (lit && !inside) triangles++;
        inside = lit;
    }
    expect(triangles).toBe(Math.min(count, 12) / 3);
    return records[draw.args.offset! / 20]!.indexCount;
}
test("a producer compute pass sizes this frame's triangles and changes the next frame without CPU draw-record writes", async () => {
    expect(await draw(3)).toBe(3);
    expect(await draw(9)).toBe(9);
});
test("transparent draws honor the GPU count and its next-frame change", async () => {
    scene = await setup(1);
    expect(await draw(3)).toBe(3);
    expect(await draw(9)).toBe(9);
});
test("a GPU index count above the mesh bound cannot draw beyond it", async () => {
    expect(await draw(6)).toBe(6);
    expect(await draw(99)).toBe(12);
});
test("a rebound count-bearing mesh draws its successor's different triangle count", async () => {
    scene ??= await setup();
    const { world, mesh, buffer } = scene;
    const meshes = world.resource(Meshes);
    const previous = meshes.remove(mesh)!;
    world.gpu.device.queue.writeBuffer(buffer, 0, new Uint32Array([6]));
    scene.mesh = meshes.add({ ...previous, indirectIndexCount: { buffer, offset: 0 } });
    expect(scene.mesh).not.toBe(mesh);
    world.storage(MeshInstance).mesh.set(scene.eid, scene.mesh);
    world.step(0);
    expect(await draw(6)).toBe(6);
});
