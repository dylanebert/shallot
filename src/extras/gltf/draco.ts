import { DRACO_WASM_BASE64 } from "./draco.wasm";

interface Kernel {
    memory: WebAssembly.Memory;
    alloc(length: number): number;
    release(ptr: number, length: number): void;
    decode(
        stream: number,
        length: number,
        faces: number,
        points: number,
        request: number,
        requested: number,
    ): number;
    close(handle: number): void;
    info(handle: number, field: number): number;
}
let wasm: Kernel | undefined;
function kernel(): Kernel {
    if (!wasm) {
        const bytes = Uint8Array.from(atob(DRACO_WASM_BASE64), (c) => c.charCodeAt(0));
        wasm = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports as unknown as Kernel;
    }
    return wasm;
}

/** Draco output data types, in Draco's numbering from 1; 64-bit and bool are refused by the decoder. */
const types = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
    "bool",
] as const;
export type DracoType = (typeof types)[number];
const arrays = {
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    float32: Float32Array,
};
type Arrays = typeof arrays;
export type DracoArray = InstanceType<Arrays[keyof Arrays]>;

export interface DracoRequest {
    /** The attribute's unique id: in glTF, the extension's `attributes` value. */
    id: number;
    type: DracoType;
}
export interface DracoMesh {
    /** Three point indices per face. */
    faces: Uint32Array;
    /** One per request, in request order: a value per point and component. */
    attributes: { components: number; data: DracoArray }[];
}

function count(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff)
        throw new Error(`invalid Draco expected ${name} count`);
    return value;
}

/**
 * Decodes a Draco 2.2 triangle mesh whose face and point counts must equal the caller's, giving
 * each requested attribute for every point in its requested type. The results own their memory.
 */
export function decodeDraco(
    stream: Uint8Array,
    faces: number,
    points: number,
    attributes: readonly DracoRequest[],
): DracoMesh {
    count(faces, "face");
    count(points, "point");
    if (stream.length > 0x7fffffff) throw new Error("invalid Draco stream byteLength");
    const request = new Uint32Array(attributes.length * 2);
    attributes.forEach(({ id, type }, index) => {
        const code = types.indexOf(type) + 1;
        if (!code) throw new Error("invalid Draco output data type");
        request[index] = count(id, "attribute id");
        request[attributes.length + index] = code;
    });
    const k = kernel();
    const input = k.alloc(stream.length);
    if (!input) throw new Error("Draco input allocation failed");
    new Uint8Array(k.memory.buffer).set(stream, input);
    const ids = k.alloc(request.byteLength);
    if (!ids) {
        k.release(input, stream.length);
        throw new Error("Draco input allocation failed");
    }
    new Uint8Array(k.memory.buffer).set(new Uint8Array(request.buffer), ids);
    const handle = k.decode(input, stream.length, faces, points, ids, attributes.length);
    try {
        const memory = k.memory.buffer;
        if (k.info(handle, 1))
            throw new Error(
                new TextDecoder().decode(
                    new Uint8Array(memory, k.info(handle, 0), k.info(handle, 1)),
                ),
            );
        const at = k.info(handle, 2);
        return {
            faces: new Uint32Array(memory.slice(at, at + k.info(handle, 3) * 12)),
            attributes: attributes.map(({ type }, index) => {
                const field = 4 + index * 3;
                const start = k.info(handle, field);
                const bytes = memory.slice(start, start + k.info(handle, field + 1));
                return {
                    components: k.info(handle, field + 2),
                    data: new arrays[type as keyof Arrays](bytes),
                };
            }),
        };
    } finally {
        k.close(handle);
    }
}
