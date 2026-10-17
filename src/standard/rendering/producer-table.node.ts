import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { cube, Meshes, MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    DirectionalLight,
    Xform,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation, type World } from "../../engine";
import { probeBuffer } from "../../engine/runtime";
import { DEFAULT_PLUGINS } from "../../standard";
import { sunShadowView } from "./atlas";
import {
    AlphaMode,
    MeshInstanceInput,
    MeshMaterial,
    MeshPreprocessSystem,
    materialLayout,
    materialType,
    registerProducerTable,
    StandardRenderer,
} from "./index";
import { MeshDrawBuffers } from "./preprocess";
import { Draws } from "./registry";

setDefaultTimeout(TIMEOUT.node);
const material = materialType({
    name: "ProducerProof",
    layout: materialLayout(d.struct({ color: d.vec4f }), {}),
    defaults: { color: d.vec4f(1, 0, 0, 1) },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
    depthPass: { shadows: true },
});
const secondMaterial = materialType({
    name: "ProducerSecondProof",
    layout: materialLayout(d.struct({ color: d.vec4f }), {}),
    defaults: { color: d.vec4f(0, 0, 1, 1) },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.color);
    },
    depthPass: { shadows: true },
});
const layout = tgpu.bindGroupLayout({
    placements: { storage: d.arrayOf(Xform), access: "mutable" },
    instances: { storage: d.arrayOf(MeshInstanceInput), access: "mutable" },
    count: { storage: d.atomic(d.u32), access: "mutable" },
    config: { uniform: d.vec4u },
    other: { uniform: d.vec3u },
    flags: { uniform: d.u32 },
});
const kernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
    "use gpu";
    for (let row = d.u32(0); row < std.min(3, std.arrayLength(layout.$.placements)); row++) {
        let x = -1.2;
        if (row === 1) x = 1.2;
        if (row === 2) x = 100;
        layout.$.placements[row] = Xform({
            pos: d.vec3f(x, 0, 0),
            quat: d.vec4f(0, 0, 0, 1),
            scale: d.vec3f(1),
        });
        let selection = d.vec3u(layout.$.config.y, layout.$.config.z, layout.$.config.w);
        if (row === 1) selection = d.vec3u(layout.$.other);
        layout.$.instances[row] = MeshInstanceInput({
            mesh: selection.x,
            materialType: selection.y,
            material: selection.z,
            alphaMode: 0,
            alphaCutoff: 0.5,
            flags: layout.$.flags,
            cullBounds: d.vec4f(0, 0, 0, -1),
        });
    }
    std.atomicStore(layout.$.count, layout.$.config.x);
});

import * as std from "typegpu/std";

const depthLayout = tgpu.bindGroupLayout({
    depth: { texture: d.textureDepth2d() },
    samples: { storage: d.arrayOf(d.f32), access: "mutable" },
});
const depthKernel = tgpu
    .computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })(/* wgsl */ `{
    let size = textureDimensions(bound.depth);
    bound.samples[in.gid.y * 512u + in.gid.x] = textureLoad(bound.depth, vec2i(in.gid.xy * size / 512u), 0);
}`)
    .$uses({ bound: depthLayout.$ });
const state = {
    create(world: World) {
        const assets = world.resource(material);
        const red = assets.add({ color: d.vec4f(1, 0, 0, 1) });
        const blue = world.resource(secondMaterial).add({ color: d.vec4f(0, 0, 1, 1) });
        const mesh = [...world.resource(Meshes).entries()][0]![0];
        const otherMesh = world
            .resource(Meshes)
            .add({ name: "producer-other", ...cube([0.5, 1, 1]) });
        const table = registerProducerTable(world, {
            name: "proof",
            capacity: 8,
            meshes: [mesh, otherMesh],
            materials: [red, blue],
        });
        const root = world.gpu.root;
        const config = root
            .createBuffer(d.vec4u, d.vec4u(3, mesh, red.type, red.material))
            .$usage("uniform");
        const other = root
            .createBuffer(d.vec3u, d.vec3u(mesh, red.type, red.material))
            .$usage("uniform");
        const flags = root.createBuffer(d.u32, 0).$usage("uniform");
        const pipe = root.createComputePipeline({ compute: kernel });
        const group = root.createBindGroup(layout, {
            placements: table.placements,
            instances: table.meshInstances,
            count: table.count,
            config,
            other,
            flags,
        });
        return {
            table,
            mesh,
            otherMesh,
            red,
            blue,
            config,
            other,
            flags,
            pipe: root.unwrap(pipe),
            group: root.unwrap(group),
            camera: 0,
        };
    },
};
const apps = gpuApps(
    import.meta.path,
    [false, true].map((producer) => ({
        defaults: false,
        plugins: [
            ...DEFAULT_PLUGINS,
            {
                name: producer ? "ProducerProof" : "EntityProof",
                warm(world: World) {
                    world.resource(material);
                    world.resource(secondMaterial);
                    if (producer) world.resource(state);
                },
                systems: producer
                    ? [
                          {
                              name: "ProduceRows",
                              group: "draw" as const,
                              before: [MeshPreprocessSystem],
                              update(world: World) {
                                  const s = world.resource(state);
                                  const pass = world.frameEncoder()!.beginComputePass();
                                  pass.setPipeline(s.pipe);
                                  pass.setBindGroup(0, s.group);
                                  pass.dispatchWorkgroups(1);
                                  pass.end();
                              },
                          },
                      ]
                    : [],
            },
        ],
    })),
);
function scene(world: World) {
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
    return camera;
}
async function shadow(world: World) {
    const root = world.gpu.root;
    const samples = root.createBuffer(d.arrayOf(d.f32, 512 * 512)).$usage("storage");
    const group = root.createBindGroup(depthLayout, { depth: sunShadowView(world)!, samples });
    const encoder = world.gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(root.unwrap(root.createComputePipeline({ compute: depthKernel })));
    pass.setBindGroup(0, root.unwrap(group));
    pass.dispatchWorkgroups(64, 64);
    pass.end();
    world.gpu.device.queue.submit([encoder.finish()]);
    return new Float32Array((await probeBuffer(world, root.unwrap(samples))).bytes);
}
async function producerArgs(world: World) {
    const draw = [...world.resource(Draws)].find((draw) =>
        draw.name.startsWith("producer:proof:"),
    )!;
    const args = new Uint32Array(
        (
            await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                offset: draw.args.offset!,
                size: 20,
            })
        ).bytes,
    );
    const packed = new Uint32Array(
        (
            await probeBuffer(
                world,
                world.gpu.root.unwrap(world.resource(MeshDrawBuffers).packedEids!),
                { offset: args[4]! * 16, size: Math.max(16, args[1]! * 16) },
            )
        ).bytes,
    );
    return { args, packed };
}
test("compute-written producer placements draw exactly the entity colour and sun-shadow pixels", async () => {
    const entity = apps()[0].world;
    const producer = apps()[1].world;
    const ec = scene(entity);
    const pc = scene(producer);
    producer.resource(state).camera = pc;
    const red = entity.resource(material).add({ color: d.vec4f(1, 0, 0, 1) });
    for (const x of [-1.2, 1.2, 100]) {
        const eid = entity.create();
        entity.add(eid, Transform, { translation: [x, 0, 0] });
        entity.add(eid, MeshInstance);
        entity.add(eid, MeshMaterial, red);
    }
    for (let i = 0; i < 3; i++) {
        entity.step(0);
        producer.step(0);
    }
    const expected = (await captureTexture(entity, ec)).rgba;
    expect(expected.some((v) => v > 50)).toBe(true);
    expect((await captureTexture(producer, pc)).rgba).toEqual(expected);
    const depth = await shadow(entity);
    expect(depth.some((v) => v > 0)).toBe(true);
    expect(await shadow(producer)).toEqual(depth);
});
test("GPU count changes the next frame without queue writes to producer columns or count", async () => {
    const world = apps()[1].world;
    const s = world.resource(state);
    const buffers = [s.table.placements, s.table.meshInstances, s.table.count].map((buffer) =>
        world.gpu.root.unwrap(buffer),
    );
    const spy = spyOn(world.gpu.device.queue, "writeBuffer");
    try {
        s.config.write(d.vec4u(1, s.mesh, s.red.type, s.red.material));
        world.step(0);
        expect((await producerArgs(world)).args[1]).toBe(1);
        const one = (await captureTexture(world, s.camera)).rgba;
        s.config.write(d.vec4u(3, s.mesh, s.red.type, s.red.material));
        world.step(0);
        expect((await producerArgs(world)).args[1]).toBe(2);
        expect((await captureTexture(world, s.camera)).rgba).not.toEqual(one);
        expect(spy.mock.calls.filter((call) => buffers.includes(call[0])).length).toBe(0);
    } finally {
        spy.mockRestore();
    }
});
test("producer rows outside the camera frustum are absent from its packed list", async () => {
    const world = apps()[1].world;
    world.step(0);
    const { args, packed } = await producerArgs(world);
    expect(args[1]).toBe(2);
    expect([packed[0], packed[4]].sort()).toEqual([0, 1]);
    expect(packed[1]).toBe(packed[0]);
    expect(packed[2]).toBe(packed[0]! + 1);
});
test("different producer meshes and material types have their own indirect draw groups", async () => {
    const world = apps()[1].world;
    const s = world.resource(state);
    s.other.write(d.vec3u(s.otherMesh, s.blue.type, s.blue.material));
    world.step(0);
    const draws = [...world.resource(Draws)].filter((draw) =>
        draw.name.startsWith("producer:proof:"),
    );
    expect(draws.length).toBe(4);
    for (const draw of draws) {
        const args = new Uint32Array(
            (
                await probeBuffer(world, world.gpu.root.unwrap(draw.args.indirect), {
                    offset: draw.args.offset!,
                    size: 20,
                })
            ).bytes,
        );
        expect(args[1]).toBe(
            (draw.mesh === s.mesh && draw.materialType === s.red.type) ||
                (draw.mesh === s.otherMesh && draw.materialType === s.blue.type)
                ? 1
                : 0,
        );
    }
    s.other.write(d.vec3u(s.mesh, s.red.type, s.red.material));
});
test("NotShadowCaster producer rows remain in colour and leave every sun-shadow slot", async () => {
    const world = apps()[1].world;
    const s = world.resource(state);
    world.step(0);
    const colour = (await captureTexture(world, s.camera)).rgba;
    expect((await shadow(world)).some((v) => v > 0)).toBe(true);
    s.flags.write(1);
    world.step(0);
    expect((await captureTexture(world, s.camera)).rgba).toEqual(colour);
    expect((await shadow(world)).every((v) => v === 0)).toBe(true);
    s.flags.write(0);
});

test("a second table registered after warm has a separate pair region and clamps its GPU count", async () => {
    const world = apps()[1].world;
    const s = world.resource(state);
    const root = world.gpu.root;
    const late = registerProducerTable(world, {
        name: "late",
        capacity: 1,
        meshes: [s.mesh],
        materials: [s.red],
    });
    const group = root.createBindGroup(layout, {
        placements: late.placements,
        instances: late.meshInstances,
        count: late.count,
        config: s.config,
        other: s.other,
        flags: s.flags,
    });
    const encoder = world.gpu.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(s.pipe);
    pass.setBindGroup(0, root.unwrap(group));
    pass.dispatchWorkgroups(1);
    pass.end();
    world.gpu.device.queue.submit([encoder.finish()]);
    world.step(0);
    expect((await producerArgs(world)).args[1]).toBe(2);
    const draw = [...world.resource(Draws)].find((draw) => draw.name.startsWith("producer:late:"))!;
    const args = new Uint32Array(
        (
            await probeBuffer(world, root.unwrap(draw.args.indirect), {
                offset: draw.args.offset!,
                size: 20,
            })
        ).bytes,
    );
    expect(args[1]).toBe(1);
    expect(args[4]).toBeGreaterThanOrEqual(2);
    expect(new Uint32Array((await probeBuffer(world, root.unwrap(late.count))).bytes)[0]).toBe(3);
});

test("registration refuses blended producer materials because transparent sorting is on the CPU", () => {
    const world = apps()[1].world;
    const blended = world
        .resource(material)
        .add({ color: d.vec4f(1) }, { alphaMode: AlphaMode.Blend });
    expect(() =>
        registerProducerTable(world, {
            name: "blended-proof",
            capacity: 1,
            meshes: [],
            materials: [blended],
        }),
    ).toThrow(
        `Producer table "blended-proof": material ${blended.type}:${blended.material} is blended; transparent sorting is on the CPU`,
    );
});
