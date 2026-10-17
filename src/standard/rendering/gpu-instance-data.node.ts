import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import { attachTexture, Camera, captureTexture, DirectionalLight } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { ClearChangeMarksSystem, lookAtRotation, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { sunShadowView } from "./atlas";
import {
    MeshInstanceInput,
    MeshMaterial,
    MeshPreprocessSystem,
    materialLayout,
    materialType,
    meshInstanceTable,
    StandardRenderer,
} from "./index";
import { Draws } from "./registry";

setDefaultTimeout(TIMEOUT.node);
const surface = materialLayout(d.struct({ color: d.vec4f }), {
    positions: { type: "storage", element: d.vec4f },
});
const material = materialType({
    name: "GpuInstanceData",
    layout: surface,
    defaults: { color: d.vec4f(1) },
    bindings: (world) => ({ positions: world.resource(producer).positions }),
    vertex: (input) => {
        "use gpu";
        return {
            world: d.vec4f(std.add(input.localPos, surface.$.positions[input.eid].xyz), 1),
            worldNormal: input.worldNormal,
        };
    },
    fragment: (input) => {
        "use gpu";
        return d.vec4f(input.parameters.color);
    },
    depthPass: { shadows: true },
});
const layout = tgpu.bindGroupLayout({
    positions: { storage: d.arrayOf(d.vec4f), access: "mutable" },
    rows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    instances: { storage: d.arrayOf(MeshInstanceInput), access: "mutable" },
    phase: { uniform: d.u32 },
});
const kernel = tgpu.computeFn({ workgroupSize: [1], in: { gid: d.builtin.globalInvocationId } })(
    (input) => {
        "use gpu";
        const row = layout.$.rows[input.gid.x];
        const slot = (input.gid.x + layout.$.phase) % 3;
        let x = -1.2;
        if (slot === 1) x = 1.2;
        if (slot === 2) x = 100;
        if (layout.$.phase === 3) {
            x = 100;
            if (input.gid.x === 0) x = 0;
        }
        layout.$.positions[row.x] = d.vec4f(x, 0, 0, 0);
        // Cull spheres are local to the authored transform, not the vertex override's world position.
        layout.$.instances[row.y].cullBounds = d.vec4f(x - 100, 0, 0, 0.9);
    },
);
const shadowLayout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepth2d() },
    samples: { storage: d.arrayOf(d.f32), access: "mutable" },
});
const sampleShadow = tgpu
    .computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(/* wgsl */ `{
    let size = textureDimensions(bound.depth);
    bound.samples[in.gid.y * 512u + in.gid.x] = textureLoad(bound.depth, vec2i(in.gid.xy * size / 512u), 0);
}`)
    .$uses({ bound: shadowLayout.$ });
const producer = {
    create(world: World) {
        const root = world.gpu.root;
        const positions = root.createBuffer(d.arrayOf(d.vec4f, 128)).$usage("storage");
        const phase = root.createBuffer(d.u32, 0).$usage("uniform");
        const pipeline = root.createComputePipeline({ compute: kernel });
        const shadow = root.createComputePipeline({ compute: sampleShadow });
        root.unwrap(pipeline);
        root.unwrap(shadow);
        return { positions, phase, pipeline, shadow };
    },
};
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            {
                name: "GpuInstanceDataProof",
                warm(world) {
                    world.resource(producer);
                    world.resource(material);
                },
                systems: [
                    {
                        name: "produce-instance-data",
                        group: "draw",
                        after: [ClearChangeMarksSystem],
                        before: [MeshPreprocessSystem],
                        update(world) {
                            const instances = meshInstanceTable(world);
                            if (!instances.count) return;
                            const state = world.resource(producer);
                            const root = world.gpu.root;
                            const group = root.createBindGroup(layout, {
                                positions: state.positions,
                                phase: state.phase,
                                rows: instances.activeRowsTyped!,
                                instances: instances.typed,
                            } as never);
                            const pass = world.frameEncoder()!.beginComputePass();
                            pass.setPipeline(root.unwrap(state.pipeline));
                            pass.setBindGroup(0, root.unwrap(group));
                            pass.dispatchWorkgroups(instances.count);
                            pass.end();
                        },
                    },
                ],
            },
        ],
    },
]);

test("compute-written eid positions move opaque entities, culling and shadow vertices in the next frame", async () => {
    const { world } = subjects()[0];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 6] });
    world.add(camera, Camera);
    world.storage(Camera).antialias.set(camera, 0);
    world.add(camera, StandardRenderer);
    attachTexture(world, camera, { width: 64, height: 32 });
    const sun = world.create();
    world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, -0.4, -0.8, -0.5) });
    world.add(sun, DirectionalLight);
    world.storage(DirectionalLight).shadowMapsEnabled.set(sun, 1);
    const eids: number[] = [];
    const colors = [d.vec4f(1, 0, 0, 1), d.vec4f(0, 1, 0, 1), d.vec4f(0, 0, 1, 1)];
    for (const color of colors) {
        // Sparse eids deliberately differ from both compact instance rows and invocation indices.
        world.create();
        const eid = world.create();
        eids.push(eid);
        world.add(eid, Transform, { translation: [100, 0, 0] });
        world.add(eid, MeshInstance);
        world.add(eid, MeshMaterial, world.resource(material).add({ color }));
    }
    const samples = world.gpu.device.createBuffer({
        size: 512 * 512 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    world.own(samples);
    async function frame(phase: number) {
        world.resource(producer).phase.write(phase);
        world.step(0);
        const draw = [...world.resource(Draws)].find(
            (entry) => entry.materialType === world.storage(MeshMaterial).type.get(eids[0]!),
        )!;
        const args = await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
            offset: draw.args.offset ?? 0,
            size: 20,
        });
        expect(new Uint32Array(args.bytes)[1]).toBe(phase === 3 ? 1 : 2);
        const pixels = (await captureTexture(world, camera)).rgba;
        const counts = [0, 0, 0];
        const xs = [0, 0, 0];
        for (let i = 0; i < pixels.length; i += 4) {
            for (let c = 0; c < 3; c++) {
                if (
                    pixels[i + c]! > 100 &&
                    pixels[i + c]! > 2 * pixels[i + ((c + 1) % 3)]! &&
                    pixels[i + c]! > 2 * pixels[i + ((c + 2) % 3)]!
                ) {
                    counts[c]++;
                    xs[c] += (i / 4) % 64;
                }
            }
        }
        const root = world.gpu.root;
        const group = root.createBindGroup(shadowLayout, { depth: sunShadowView(world)!, samples });
        const encoder = world.gpu.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(root.unwrap(world.resource(producer).shadow));
        pass.setBindGroup(0, root.unwrap(group));
        pass.dispatchWorkgroups(64, 64);
        pass.end();
        world.gpu.device.queue.submit([encoder.finish()]);
        const depth = new Float32Array((await probeBuffer(world, samples)).bytes);
        expect(depth.some((v) => v > 0)).toBe(true);
        return { counts, xs: xs.map((x, i) => x / counts[i]!), depth };
    }
    world.step(0);
    world.step(0);
    const first = await frame(0);
    expect(first.counts[0]).toBeGreaterThan(0);
    expect(first.counts[1]).toBeGreaterThan(0);
    expect(first.counts[2]).toBe(0);
    expect(first.xs[0]).toBeLessThan(first.xs[1]!);
    const next = await frame(1);
    expect(next.counts[0]).toBeGreaterThan(0);
    expect(next.counts[1]).toBe(0);
    expect(next.counts[2]).toBeGreaterThan(0);
    expect(next.xs[2]).toBeLessThan(next.xs[0]!);
    // Change occupied geometry as well as identity so a stationary shadow shader cannot pass.
    const centered = await frame(3);
    expect(centered.counts[0]).toBeGreaterThan(0);
    expect(centered.counts[1]).toBe(0);
    expect(centered.counts[2]).toBe(0);
    expect(centered.depth).not.toEqual(first.depth);
    expect((await frame(0)).depth).toEqual(first.depth);
    for (const eid of eids) expect(world.storage(Transform).translation.x.get(eid)).toBe(100);
});
