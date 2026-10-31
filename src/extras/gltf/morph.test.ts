import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { generateTangents } from "../../core/mesh";
import { computeNormals, generateSceneTangents, parse } from "./gltf";
import { DeformationRanges } from "./morph";
import { morphFixture } from "./morph.fixture";
import { VertexDeformation } from "./skinning";

test("deformation vertex data reuses the skin stride; ranges coalesce and refuse named capacity overflow without changing extent", () => {
    expect(d.sizeOf(VertexDeformation)).toBe(32);
    const ranges = new DeformationRanges();
    expect(ranges.allocate(2, 8, "weights")).toBe(0);
    expect(ranges.allocate(3, 8, "weights")).toBe(2);
    expect(ranges.allocate(2, 8, "weights")).toBe(5);
    ranges.release(2, 3);
    expect(ranges.allocate(2, 8, "weights")).toBe(2);
    expect(() => ranges.allocate(3, 8, "weights")).toThrow("weights exceeds WebGPU");
    expect(ranges.extent).toBe(7);
    ranges.release(2, 2);
    ranges.release(5, 2);
    expect(ranges.extent).toBe(2);
    ranges.release(0, 2);
    expect(ranges.extent).toBe(0);
    expect(() => ranges.allocate(0, 8, "weights")).toThrow("weights requires a positive count");
});

const close = (a: number, b: number) => expect(Math.abs(a - b)).toBeLessThanOrEqual(2e-6);

test("morph streams decode sparse and bufferless accessors, missing semantics are zero, optional semantics are reported by name", () => {
    const { json, buffers } = morphFixture();
    const primitive = json.meshes![0].primitives[0];
    const count = json.accessors!.length;
    const sparse = new Float32Array([2, 3, 4]);
    buffers.push(new Uint8Array([1]).buffer, sparse.buffer);
    const view = json.bufferViews!.length;
    json.bufferViews!.push(
        { buffer: buffers.length - 2, byteLength: 1 },
        { buffer: buffers.length - 1, byteLength: 12 },
    );
    json.accessors!.push(
        {
            componentType: 5126,
            count: 5,
            type: "VEC3",
            sparse: {
                count: 1,
                indices: { bufferView: view, componentType: 5121 },
                values: { bufferView: view + 1 },
            },
        },
        { componentType: 5126, count: 5, type: "VEC3" },
    );
    primitive.targets = [
        { POSITION: count },
        {},
        { NORMAL: count + 1 },
        { TEXCOORD_0: count + 1, COLOR_0: count + 1 },
    ];
    const scene = parse(json, buffers);
    expect(scene.unsupported).toEqual(["morph.COLOR_0", "morph.TEXCOORD_0"]);
    expect([...scene.meshes[0].morphTargets![0].position!]).toEqual([
        0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(scene.meshes[0].morphTargets![1]).toEqual({
        position: undefined,
        normal: undefined,
        tangent: undefined,
    });
    expect([...scene.meshes[0].morphTargets![2].normal!]).toEqual(Array(15).fill(0));
});

test("implicit flat target normals use displaced faces and every delta follows corner expansion", () => {
    const { json, buffers } = morphFixture();
    delete json.meshes![0].primitives[0].attributes.NORMAL;
    const mesh = parse(json, buffers).meshes[0];
    expect(mesh.vertices.length / 8).toBe(6);
    const positions = Float32Array.from(
        { length: 18 },
        (_, i) => mesh.vertices[Math.floor(i / 3) * 8 + (i % 3)],
    );
    for (const target of mesh.morphTargets!) {
        const displaced = positions.map((v, i) => v + (target.position?.[i] ?? 0));
        const normals = computeNormals(displaced, mesh.indices);
        for (let i = 0; i < normals.length; i++)
            close(
                target.normal![i],
                normals[i] - mesh.vertices[Math.floor(i / 3) * 8 + 4 + (i % 3)],
            );
    }
});

test("provided tangents keep provided deltas and absent tangent deltas remain zero", async () => {
    const { json, buffers } = morphFixture();
    const tangents = Float32Array.from({ length: 20 }, (_, i) =>
        i % 4 === 0 || i % 4 === 3 ? 1 : 0,
    );
    const view = json.bufferViews!.length;
    json.bufferViews!.push({ buffer: buffers.length, byteLength: tangents.byteLength });
    buffers.push(tangents.buffer);
    json.accessors!.push({ bufferView: view, componentType: 5126, type: "VEC4", count: 5 });
    const primitive = json.meshes![0].primitives[0];
    primitive.attributes.TANGENT = json.accessors!.length - 1;
    primitive.targets![0].TANGENT = primitive.targets![0].NORMAL;
    const scene = parse(json, buffers);
    await generateSceneTangents(scene, json, buffers);
    expect(scene.meshes[0].tangent).toEqual(tangents);
    expect(scene.meshes[0].morphTargets![0].tangent).toEqual(
        scene.meshes[0].morphTargets![0].normal,
    );
    expect(scene.meshes[0].morphTargets![1].tangent).toBeUndefined();
    delete primitive.attributes.NORMAL;
    delete json.materials![0].normalTexture;
    const flat = parse(json, buffers).meshes[0];
    expect(flat.tangent).toBeUndefined();
    expect(flat.morphTargets![0].tangent).toBeUndefined();
});

test("MikkTSpace splits on every target's corner tangent, remaps all streams, and ignores injected provided tangent deltas", async () => {
    const { json, buffers } = morphFixture();
    const primitive = json.meshes![0].primitives[0];
    // Invalid glTF is intentional: a validator-clean asset cannot carry target TANGENT without base TANGENT.
    for (const target of primitive.targets!) target.TANGENT = target.POSITION;
    primitive.attributes.TEXCOORD_1 = primitive.attributes.TEXCOORD_0;
    primitive.attributes.COLOR_0 = primitive.attributes.WEIGHTS_0;
    const scene = parse(json, buffers),
        source = scene.meshes[0];
    const base = await generateTangents(source);
    expect(base.indices[0]).toBe(base.indices[3]);
    await generateSceneTangents(scene, json, buffers);
    const result = scene.meshes[0];
    expect(result.indices[0]).not.toBe(result.indices[3]);
    for (let m = 0; m < source.morphTargets!.length; m++) {
        const target = source.morphTargets![m],
            vertices = source.vertices.slice();
        for (let v = 0; v < vertices.length / 8; v++) {
            for (let a = 0; a < 3; a++) {
                vertices[v * 8 + a] += target.position?.[v * 3 + a] ?? 0;
                vertices[v * 8 + 4 + a] += target.normal?.[v * 3 + a] ?? 0;
            }
            const length = Math.hypot(...vertices.subarray(v * 8 + 4, v * 8 + 7));
            for (let a = 0; a < 3; a++) vertices[v * 8 + 4 + a] /= length;
        }
        const generated = await generateTangents({ ...source, vertices });
        for (let corner = 0; corner < source.indices.length; corner++) {
            const original = source.indices[corner],
                final = result.indices[corner];
            expect(result.vertices.subarray(final * 8, final * 8 + 8)).toEqual(
                source.vertices.subarray(original * 8, original * 8 + 8),
            );
            for (const [semantic, width] of [
                ["joints", 4],
                ["weights", 4],
                ["uv1", 2],
                ["color", 4],
            ] as const)
                expect(result[semantic]!.subarray(final * width, (final + 1) * width)).toEqual(
                    source[semantic]!.subarray(original * width, (original + 1) * width),
                );
            for (let a = 0; a < 3; a++) {
                close(
                    result.morphTargets![m].tangent![final * 3 + a],
                    generated.tangent[generated.indices[corner] * 4 + a] -
                        base.tangent[base.indices[corner] * 4 + a],
                );
                close(
                    result.morphTargets![m].position![final * 3 + a],
                    target.position![original * 3 + a],
                );
                close(
                    result.morphTargets![m].normal![final * 3 + a],
                    target.normal![original * 3 + a],
                );
            }
        }
    }
    expect(Math.abs(result.morphTargets![0].tangent![result.indices[3] * 3])).toBeGreaterThan(1);
});

test("node weights override mesh defaults; missing defaults use the first primitive that actually has targets", () => {
    const { json, buffers } = morphFixture();
    const mesh = json.meshes![0];
    const primitive = structuredClone(mesh.primitives[0]);
    delete primitive.targets;
    mesh.primitives.unshift(primitive);
    expect(parse(json, buffers).nodes[0].weights).toEqual([0, 0]);
    delete json.nodes![0].weights;
    expect(parse(json, buffers).nodes[0].weights).toEqual([0.2, 0.1]);
    delete mesh.weights;
    const scene = parse(json, buffers);
    expect(scene.nodes[0].weights).toEqual([0, 0]);
    expect(scene.meshes[0].morphTargets).toBeUndefined();
});
