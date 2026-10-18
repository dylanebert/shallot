import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { Meshes } from "../../core/mesh";
import { meshResourcesKey } from "../../core/mesh/mesh";
import {
    AlphaMode,
    Draws,
    Materials,
    MeshRenderPlugin,
    StandardMaterial,
    StandardRenderingPlugin,
} from "./index";
import { AlphaPipelineKey } from "./material-type";
import { MeshDrawBuffers } from "./preprocess";

setDefaultTimeout(TIMEOUT.node);
const config = { defaults: false, plugins: [MeshRenderPlugin, StandardRenderingPlugin] };
const subjects = gpuApps(import.meta.path, [config, config, config, config, config, config]);
function setup(index: number, count = 1) {
    const { world } = subjects()[index]!;
    const meshes = world.resource(Meshes);
    const geometry = meshes.values().next().value!;
    const buffer = world.gpu.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    world.own(buffer);
    world.gpu.device.queue.writeBuffer(buffer, 0, new Uint32Array([3, 9]));
    const spec = { ...geometry, name: "count", indirectIndexCount: { buffer }, indexCount: 12 };
    const handles = Array.from({ length: count }, () => meshes.add(spec));
    world.step(0);
    return { world, meshes, spec, handles, buffer };
}
test("a hundred warm count-bearing turnovers create no device buffers or bind groups", () => {
    const s = setup(0, 200);
    const device = s.world.gpu.device;
    for (const handle of s.handles.slice(1)) {
        const buffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
        s.world.own(buffer);
        s.meshes.remove(handle);
        expect(s.meshes.add({ ...s.spec, indirectIndexCount: { buffer } })).not.toBe(handle);
    }
    s.world.step(0);
    const buffer = device.createBuffer.bind(device),
        group = device.createBindGroup.bind(device);
    let buffers = 0,
        groups = 0;
    device.createBuffer = (d) => {
        buffers++;
        return buffer(d);
    };
    device.createBindGroup = (d) => {
        groups++;
        return group(d);
    };
    try {
        for (let i = 0; i < 100; i++) {
            const handle = s.handles[0]!;
            s.meshes.remove(handle);
            s.handles[0] = s.meshes.add({
                ...s.spec,
                indirectIndexCount: { buffer: s.buffer, offset: (i % 2) * 4 },
            });
            expect(s.handles[0]).not.toBe(handle);
            s.world.step(0);
        }
    } finally {
        device.createBuffer = buffer;
        device.createBindGroup = group;
    }
    expect({ buffers, groups }).toEqual({ buffers: 0, groups: 0 });
});
test("two hundred counts sharing a source encode one count dispatch and no per-record copies", async () => {
    const s = setup(1, 200);
    const target = s.world.gpu.root.unwrap(s.world.resource(MeshDrawBuffers).drawArgs!);
    const device = s.world.gpu.device;
    const create = device.createCommandEncoder.bind(device);
    let dispatches = 0,
        copies = 0;
    device.createCommandEncoder = (...args) => {
        const encoder = create(...args);
        const begin = encoder.beginComputePass.bind(encoder),
            copy = encoder.copyBufferToBuffer.bind(encoder);
        encoder.beginComputePass = (descriptor) => {
            const pass = begin(descriptor);
            if (descriptor?.label === "shallot-mesh-count") {
                const dispatch = pass.dispatchWorkgroups.bind(pass);
                pass.dispatchWorkgroups = (...args) => {
                    dispatches++;
                    dispatch(...args);
                };
            }
            return pass;
        };
        encoder.copyBufferToBuffer = (...args: unknown[]) => {
            if (args[1] === target || args[2] === target) copies++;
            Reflect.apply(copy, encoder, args);
        };
        return encoder;
    };
    try {
        s.world.step(0);
    } finally {
        device.createCommandEncoder = create;
    }
    expect({ dispatches, copies }).toEqual({ dispatches: 1, copies: 0 });
    const args = await s.world.resource(MeshDrawBuffers).drawArgs!.read();
    for (const draw of s.world.resource(Draws))
        if (s.handles.includes(draw.mesh)) expect(args[draw.args.offset! / 20]!.indexCount).toBe(3);
});
test("invalid count sources are refused before handle, membership or version changes", () => {
    const s = setup(2);
    const registry = s.world.resource(meshResourcesKey).meshes;
    const before = [registry.size, registry.capacity, registry.version];
    for (const offset of [NaN, Infinity, -4, 1, 16]) {
        expect(() =>
            s.meshes.add({ ...s.spec, indirectIndexCount: { buffer: s.buffer, offset } }),
        ).toThrow("indirectIndexCount");
        expect([registry.size, registry.capacity, registry.version]).toEqual(before);
    }
    const buffer = s.world.gpu.device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
    s.world.own(buffer);
    expect(() => s.meshes.add({ ...s.spec, indirectIndexCount: { buffer } })).toThrow(
        "indirectIndexCount",
    );
    expect([registry.size, registry.capacity, registry.version]).toEqual(before);
});
test("failed draw publication does not consume a new alpha key before its record upload succeeds", async () => {
    const s = setup(5);
    const handle = s.handles[0]!;
    s.meshes.remove(handle);
    const next = s.meshes.add({ ...s.spec, indexBase: 6 });
    s.world.step(0);
    s.world.resource(Materials).add(StandardMaterial(), { alphaMode: AlphaMode.Blend });
    const queue = s.world.gpu.device.queue,
        write = queue.writeBuffer.bind(queue);
    let failures = 0;
    queue.writeBuffer = (...args) => {
        if (args[0].label === "shallot-draw-args") {
            failures++;
            throw new Error("injected draw upload failure");
        }
        write(...args);
    };
    try {
        expect(() => s.world.step(0)).toThrow("injected draw upload failure");
    } finally {
        queue.writeBuffer = write;
    }
    expect(failures).toBe(1);
    s.world.step(0);
    const draw = [...s.world.resource(Draws)].find(
        (d) => d.mesh === next && d.alphaPipelineKey === AlphaPipelineKey.Blend,
    )!;
    const args = (await draw.args.indirect.read()) as { firstIndex: number }[];
    expect(args[draw.args.offset! / 20]!.firstIndex).toBe(6);
});
test("a failed existing count-table upload retries the new count instead of trusting its comparison image", async () => {
    const s = setup(4);
    const handle = s.handles[0]!;
    s.meshes.remove(handle);
    const next = s.meshes.add({ ...s.spec, indirectIndexCount: { buffer: s.buffer, offset: 4 } });
    expect(next).not.toBe(handle);
    const queue = s.world.gpu.device.queue,
        write = queue.writeBuffer.bind(queue);
    let failures = 0;
    queue.writeBuffer = (...args) => {
        if (args[0].label === "shallot-mesh-count-rows") {
            failures++;
            throw new Error("injected count upload failure");
        }
        write(...args);
    };
    try {
        expect(() => s.world.step(0)).toThrow("injected count upload failure");
    } finally {
        queue.writeBuffer = write;
    }
    expect(failures).toBe(1);
    s.world.step(0);
    const draw = [...s.world.resource(Draws)].find((d) => d.mesh === next)!;
    const args = (await draw.args.indirect.read()) as { indexCount: number }[];
    expect(args[draw.args.offset! / 20]!.indexCount).toBe(9);
});
test("a failed count binding rebuild retries the unconsumed membership on the next frame", async () => {
    const s = setup(3);
    const buffer = s.world.gpu.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    s.world.own(buffer);
    s.world.gpu.device.queue.writeBuffer(buffer, 0, new Uint32Array([6]));
    const mesh = s.meshes.add({ ...s.spec, indirectIndexCount: { buffer } });
    const previousArgs = s.world.resource(MeshDrawBuffers).drawArgs;
    const device = s.world.gpu.device,
        create = device.createBindGroup.bind(device);
    device.createBindGroup = (descriptor) => {
        if ([...descriptor.entries].some((e) => (e.resource as GPUBufferBinding).buffer === buffer))
            throw new Error("injected count binding failure");
        return create(descriptor);
    };
    try {
        expect(() => s.world.step(0)).toThrow("injected count binding failure");
    } finally {
        device.createBindGroup = create;
    }
    expect(s.world.resource(MeshDrawBuffers).drawArgs).toBe(previousArgs);
    s.world.step(0);
    const draw = [...s.world.resource(Draws)].find((d) => d.mesh === mesh)!;
    const args = (await draw.args.indirect.read()) as { indexCount: number }[];
    expect(args[draw.args.offset! / 20]!.indexCount).toBe(6);
});
