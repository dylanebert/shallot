import { expect, setDefaultTimeout, test } from "bun:test";
import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Meshes, type MeshHandle, MeshPlugin, registerMesh } from "./index";
import { clearMeshes, flushMeshes } from "./mesh";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [MeshPlugin] }]);
const vertices = new Float32Array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1]);
const spec = (name: string) => ({ name, vertices, indices });
async function words(buffer: TgpuBuffer<d.AnyData>) {
    const { world } = subjects()[0];
    const source = world.gpu.root.unwrap(buffer);
    const copy = world.gpu.root.createBuffer(d.arrayOf(d.u32, source.size / 4)).$usage("storage");
    const encoder = world.gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, world.gpu.root.unwrap(copy), 0, source.size);
    world.gpu.device.queue.submit([encoder.finish()]);
    try {
        return await copy.read();
    } finally {
        copy.destroy();
    }
}

test("a batch splits by attribute signature, with dense absolute addressing and unchanged default bytes", async () => {
    const { world } = subjects()[0];
    clearMeshes(world);
    const aloneHandle = registerMesh(world, spec("plain"));
    flushMeshes(world);
    const alone = world.resource(Meshes).get(aloneHandle)!;
    const bytes = await Promise.all([
        words(alone.vertices),
        words(alone.position!),
        words(alone.quant!),
    ]);
    clearMeshes(world);
    const plainHandle = registerMesh(world, spec("plain"));
    const handles = new Map<string, MeshHandle>();
    for (const [name, data] of [
        ["a", new Float32Array([1, 2])],
        ["b", new Float32Array([3, 4])],
    ] as const)
        handles.set(
            name,
            registerMesh(world, {
                ...spec(name),
                attributes: { weight: { element: d.f32, data } },
            }),
        );
    flushMeshes(world);
    const meshes = world.resource(Meshes);
    const plain = meshes.get(plainHandle)!;
    const a = meshes.get(handles.get("a")!)!;
    const b = meshes.get(handles.get("b")!)!;
    expect(a.vertices).toBe(b.vertices);
    expect(plain.vertices).not.toBe(a.vertices);
    expect(
        await Promise.all([words(plain.vertices), words(plain.position!), words(plain.quant!)]),
    ).toEqual(bytes);
    expect(await a.attributes!.weight.read()).toEqual([1, 2, 3, 4]);
    expect(await b.indices.read()).toEqual([0, 1, 2, 3]);
});

test("a wrong-length stream refuses naming mesh, stream and storage stride", () => {
    const { world } = subjects()[0];
    expect(() =>
        registerMesh(world, {
            ...spec("bad"),
            attributes: { weight: { element: d.vec3f, data: new Float32Array(6) } },
        }),
    ).toThrow('mesh "bad": attribute "weight"');
});

test("clearMeshes destroys every attribute buffer with its family", () => {
    const { world } = subjects()[0];
    clearMeshes(world);
    const handle = registerMesh(world, {
        ...spec("owned"),
        attributes: {
            weight: { element: d.f32, data: new Float32Array(2) },
            tint: { element: d.vec4f, data: new Float32Array(8) },
        },
    });
    flushMeshes(world);
    const mesh = world.resource(Meshes).get(handle)!;
    const streams = Object.values(mesh.attributes ?? {});
    expect(streams).toHaveLength(2);
    clearMeshes(world);
    expect(streams.every((buffer) => buffer.destroyed)).toBe(true);
});
