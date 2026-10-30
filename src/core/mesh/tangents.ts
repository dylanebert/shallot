import * as d from "typegpu/data";
import type { MeshData } from "./mesh";

interface Kernel {
    memory: WebAssembly.Memory;
    allocWords(words: number): number;
    freeWords(ptr: number, words: number): void;
    generate(ptr: number, words: number): number;
}
let kernel: Promise<Kernel> | undefined;

async function load(): Promise<Kernel> {
    const { MIKKTSPACE_WASM_BASE64 } = await import("./mikktspace.wasm");
    const bytes = Uint8Array.from(atob(MIKKTSPACE_WASM_BASE64), (char) => char.charCodeAt(0));
    const { instance } = await WebAssembly.instantiate(bytes);
    return instance.exports as unknown as Kernel;
}

/** Generates MikkTSpace tangents with glTF handedness (w negated), without a GPU.
 * Input is eight interleaved f32 lanes per vertex: position, normal, UV. uvSet selects
 * the base UV (0) or uv1 (1), which must be present. Indices must describe triangles.
 * Returns owned arrays; input arrays are unchanged. Only vertices whose corners have
 * different tangent bit patterns are split, duplicating every supplied stream, including
 * named attributes. The asynchronously instantiated WASM instance is reused across calls.
 */
export async function generateTangents(
    mesh: MeshData,
    uvSet: 0 | 1 = 0,
): Promise<MeshData & { tangent: Float32Array }> {
    const nv = mesh.vertices.length / 8;
    if (!Number.isInteger(nv) || mesh.indices.length % 3 !== 0)
        throw new Error("generateTangents requires eight-f32 vertices and triangle indices");
    if (uvSet !== 0 && uvSet !== 1) throw new Error("generateTangents uvSet must be 0 or 1");
    if (uvSet === 1 && !mesh.uv1) throw new Error("generateTangents uvSet 1 requires uv1");
    for (const index of mesh.indices)
        if (index >= nv) throw new Error("generateTangents index is outside the vertex array");
    if (!mesh.vertices.every(Number.isFinite) || (uvSet === 1 && !mesh.uv1!.every(Number.isFinite)))
        throw new Error(
            "generateTangents requires finite positions, normals and texture coordinates",
        );
    const streams: { key: string; width: number; data: Uint8Array }[] = [];
    const add = (key: string, data: ArrayBufferView, width: number) => {
        if (data.byteLength !== nv * width * 4)
            throw new Error(`generateTangents ${key}: wrong stream length`);
        streams.push({
            key,
            width,
            data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        });
    };
    add("vertices", mesh.vertices, 8);
    if (mesh.uv1) add("uv1", mesh.uv1, 2);
    if (mesh.color) add("color", mesh.color, 4);
    for (const [name, stream] of Object.entries(mesh.attributes ?? {}))
        add(name, stream.data, d.sizeOf(d.arrayOf(stream.element, 1)) / 4);
    const size =
        4 +
        streams.length +
        streams.reduce((sum, s) => sum + s.data.byteLength / 4, 0) +
        mesh.indices.length;
    const wasm = await (kernel ??= load());
    const ptr = wasm.allocWords(size);
    let output = 0;
    let outputSize = 0;
    try {
        const words = new Uint32Array(wasm.memory.buffer, ptr, size);
        words.set([nv, mesh.indices.length, streams.length, uvSet]);
        words.set(
            streams.map((s) => s.width),
            4,
        );
        let offset = 4 + streams.length;
        for (const stream of streams) {
            new Uint8Array(wasm.memory.buffer, ptr + offset * 4, stream.data.byteLength).set(
                stream.data,
            );
            offset += stream.data.byteLength / 4;
        }
        words.set(mesh.indices, offset);
        output = wasm.generate(ptr, size);
        if (!output) throw new Error("MikkTSpace could not generate tangents for the mesh");
        const header = new Uint32Array(wasm.memory.buffer, output, 2);
        outputSize = header[0];
        const count = header[1];
        const result = { ...mesh } as MeshData & { tangent: Float32Array };
        offset = 2;
        const arrays = streams.map((stream) => {
            const data = new Float32Array(
                wasm.memory.buffer,
                output + offset * 4,
                count * stream.width,
            ).slice();
            offset += data.length;
            return data;
        });
        result.vertices = arrays[0];
        let next = 1;
        if (mesh.uv1) result.uv1 = arrays[next++];
        if (mesh.color) result.color = arrays[next++];
        if (mesh.attributes)
            result.attributes = Object.fromEntries(
                Object.entries(mesh.attributes).map(([name, stream]) => [
                    name,
                    {
                        element: stream.element,
                        data: new Uint8Array(arrays[next++].buffer),
                    },
                ]),
            );
        result.indices = new Uint32Array(
            wasm.memory.buffer,
            output + offset * 4,
            mesh.indices.length,
        ).slice();
        offset += mesh.indices.length;
        result.tangent = new Float32Array(
            wasm.memory.buffer,
            output + offset * 4,
            count * 4,
        ).slice();
        return result;
    } finally {
        if (output) wasm.freeWords(output, outputSize);
        wasm.freeWords(ptr, size);
    }
}
