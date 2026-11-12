import { type DracoType, decodeDraco } from "../../src/extras/gltf/draco";

const types: DracoType[] = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64", "bool"];
/** A corpus stream with the counts and requests its glTF accessors give. */
export interface Stream {
    name: string;
    bytes: Uint8Array<ArrayBuffer>;
    faces: number;
    points: number;
    /** Unique id and Draco output type id per requested attribute. */
    requests: [number, number][];
}
interface Handle { ptr: number }
interface Attribute extends Handle { num_components(): number }
export interface Reference {
    Decoder: new () => Handle & {
        GetEncodedGeometryType(array: Int8Array): number;
        DecodeArrayToMesh(array: Int8Array, length: number, mesh: Handle): { ok(): boolean; error_msg(): string };
        GetAttributeByUniqueId(mesh: Handle, id: number): Attribute;
        GetTrianglesUInt32Array(mesh: Handle, length: number, ptr: number): boolean;
        GetAttributeDataArrayForAllPoints(mesh: Handle, attribute: Attribute, type: number, length: number, ptr: number): boolean;
    };
    Mesh: new () => Handle & { num_faces(): number; num_points(): number };
    TRIANGULAR_MESH: number;
    HEAPF32: Float32Array<ArrayBuffer>;
    _malloc(length: number): number;
    _free(ptr: number): void;
    destroy(handle: Handle): void;
}
const arrays = [undefined, Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array, undefined, undefined, Float32Array] as const;

/** Decodes as three.js DRACOLoader's worker does with draco_wasm_wrapper_gltf.js: a decoder per stream, then owned slices. */
export function referenceDecode(draco: Reference, stream: Stream) {
    const decoder = new draco.Decoder();
    try {
        const array = new Int8Array(stream.bytes.buffer, stream.bytes.byteOffset, stream.bytes.byteLength);
        if (decoder.GetEncodedGeometryType(array) !== draco.TRIANGULAR_MESH) throw new Error(`${stream.name}: not a mesh`);
        const mesh = new draco.Mesh();
        const status = decoder.DecodeArrayToMesh(array, array.byteLength, mesh);
        if (!status.ok() || mesh.ptr === 0) throw new Error(`${stream.name}: ${status.error_msg()}`);
        const attributes = stream.requests.map(([id, type]) => {
            const attribute = decoder.GetAttributeByUniqueId(mesh, id);
            const Type = arrays[type]!;
            const length = mesh.num_points() * attribute.num_components() * Type.BYTES_PER_ELEMENT;
            const ptr = draco._malloc(length);
            decoder.GetAttributeDataArrayForAllPoints(mesh, attribute, type, length, ptr);
            const data = new Type(draco.HEAPF32.buffer, ptr, length / Type.BYTES_PER_ELEMENT).slice();
            draco._free(ptr);
            return data;
        });
        const length = mesh.num_faces() * 12;
        const ptr = draco._malloc(length);
        decoder.GetTrianglesUInt32Array(mesh, length, ptr);
        const faces = new Uint32Array(draco.HEAPF32.buffer, ptr, length / 4).slice();
        draco._free(ptr);
        draco.destroy(mesh);
        return { faces, attributes };
    } finally { draco.destroy(decoder); }
}
export function ownDecode(stream: Stream) {
    const mesh = decodeDraco(stream.bytes, stream.faces, stream.points, stream.requests.map(([id, type]) => ({ id, type: types[type - 1] })));
    return { faces: mesh.faces, attributes: mesh.attributes.map((a) => a.data) };
}
function equal(a: ArrayBufferView, b: ArrayBufferView): boolean {
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    return x.length === y.length && x.every((value, index) => value === y[index]);
}
export function models(streams: Stream[]): string[] {
    return [...new Set(streams.map((s) => s.name.split("/")[0]))];
}
// meshoptimizer tools/codecbench.cpp keeps maximum throughput over attempts.
// With fixed work, that is minimum elapsed time, not a contention-heavy median.
export const SAMPLE_BATCHES = 256;

export function timing(reference: Reference, streams: Stream[], first: number) {
    for (const model of models(streams)) {
        const files = streams.filter((s) => s.name.startsWith(`${model}/`));
        for (const file of files) {
            const own = ownDecode(file), ref = referenceDecode(reference, file);
            if (!equal(own.faces, ref.faces) || own.attributes.some((a, i) => !equal(a, ref.attributes[i]))) throw new Error(`${file.name}: WASM reference mismatch`);
        }
        let observed = 0;
        const own = () => { for (const file of files) observed ^= ownDecode(file).faces[0]; };
        const ref = () => { for (const file of files) observed ^= referenceDecode(reference, file).faces[0]; };
        // Each batch decodes every stream of the model through its entry, with input and output copies.
        const functions = [own, ref];
        for (let round = 0; round < 4; round++) { functions[(first + round) & 1](); functions[(first + round + 1) & 1](); }
        function batch(index: number, iterations: number) {
            const start = performance.now();
            for (let i = 0; i < iterations; i++) functions[index]();
            return (performance.now() - start) / iterations;
        }
        let iterations = 1;
        for (let round = 0; ; round++) {
            const a = (first + round) & 1;
            const one = batch(a, iterations);
            const two = batch(1 - a, iterations);
            if (Math.min(one, two) * iterations >= 100) break;
            iterations *= 2;
        }
        const samples: number[][] = [[], []];
        for (let round = 0; round < SAMPLE_BATCHES; round++) {
            const a = (first + round) & 1;
            samples[a].push(batch(a, iterations)); samples[1 - a].push(batch(1 - a, iterations));
        }
        const minima = samples.map((values) => Math.min(...values));
        const firstHalf = samples.map((values) => Math.min(...values.slice(0, SAMPLE_BATCHES / 2)));
        console.log(JSON.stringify({ process: first + 1, model, streams: files.length, statistic: "minimum", batches: SAMPLE_BATCHES, iterations, ownFirstHalfMs: firstHalf[0], referenceFirstHalfMs: firstHalf[1], ownMs: minima[0], referenceMs: minima[1], ratio: minima[0] / minima[1], observed }));
    }
}
