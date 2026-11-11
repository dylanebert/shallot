import { cpus } from "node:os";
import { resolve } from "node:path";
import { gltfArtifactIsFresh } from "../../crates/gltf/scripts/inputs";
import { GLTF_WASM_BASE64 } from "../../src/extras/gltf/gltf.wasm";
import { decodeMeshoptView } from "../../src/extras/gltf/kernel";
import { referenceDecoder, referenceInstance } from "./reference";
import { timePair } from "./timing";
import { loadRecords, assertDecoded, modes, filters } from "./records";

const root = resolve(".");
if (!gltfArtifactIsFresh(root)) throw new Error("rebuild with bun crates/gltf/scripts/build.ts");
const decoder = await referenceDecoder();
const reference = (await referenceInstance()).exports;
const { instance } = await WebAssembly.instantiate(Buffer.from(GLTF_WASM_BASE64, "base64"), {});
const kernel = instance.exports;
type WasmFunction = (...args: number[]) => number;
const allocate = kernel.meshoptAlloc as WasmFunction;
const free = kernel.meshoptFree as WasmFunction;
const memory = kernel.memory as WebAssembly.Memory;
const sbrk = reference.sbrk as WasmFunction;
const referenceMemory = reference.memory as WebAssembly.Memory;
const functions = ["meshopt_decodeVertexBuffer", "meshopt_decodeIndexBuffer", "meshopt_decodeIndexSequence"];
const ownFunctions = ["decodeMeshoptAttributes", "decodeMeshoptTriangles", "decodeMeshoptIndices"];
const filterFunctions = ["", "meshopt_decodeFilterOct", "meshopt_decodeFilterQuat", "meshopt_decodeFilterExp", "meshopt_decodeFilterColor"];
const selected = new Set([
    "MeshoptCubeTest/glTF:23", "MeshoptCubeTest/glTF:33",
    "BrainStem/glTF-Meshopt-EXT:4", "BrainStem/glTF-Meshopt-EXT:7",
    "DragonAttenuation/glTF-Meshopt-EXT:2", "DragonAttenuation/glTF-Meshopt-EXT:3",
]);
const records = await loadRecords();
console.log(`${cpus()[0]?.model}; ${process.versions.bun ? `Bun ${process.versions.bun}` : `Node ${process.version}`}`);
console.log("Microseconds/decode. Inner: WASM entry, validation, decode and filter; no allocation or boundary copies.");
console.log("Entry: production JavaScript entry including allocation, copies and free. Each pair uses 1000 warmups per side and common iteration counts; median of 8 batches, alternating first side.");
console.log("Scopes use separate instances and timed batches; entry-minus-inner is diagnostic, not an isolated boundary cost.");
console.log("asset\tview\tRust_inner\tref_inner\tRust_entry\tref_entry\tRust_entry_minus_inner\tref_entry_minus_inner");
for (const record of records) {
    const { asset, index, count, stride, mode, filter, khr, source, length: decoded } = record;
    if (!selected.has(`${asset}:${index}`)) continue;
    const compressed = source.length;
    const output = new Uint8Array(decoded);
    const ptr = allocate(compressed + decoded);
    const sourcePtr = ptr + decoded;
    new Uint8Array(memory.buffer, sourcePtr, compressed).set(source);
    const rp = sbrk(((count + 3) & ~3) * stride);
    const input = sbrk(compressed);
    new Uint8Array(referenceMemory.buffer, input, compressed).set(source);
    const referenceDecode = reference[functions[mode]] as WasmFunction;
    const referenceFilter = reference[filterFunctions[filter]] as WasmFunction | undefined;
    const decode = kernel[ownFunctions[mode]] as WasmFunction;
    const own = () => {
        const result = mode === 0 ? decode(ptr, count, stride, sourcePtr, compressed, filter, khr) : decode(ptr, count, stride, sourcePtr, compressed);
        if (result) throw new Error("Rust inner decode failed");
    };
    const ref = () => {
        if (referenceDecode(rp, count, stride, input, compressed)) throw new Error("reference inner decode failed");
        referenceFilter?.(rp, (count + 3) & ~3, stride);
    };
    try {
        own(); ref();
        await assertDecoded(record, new Uint8Array(memory.buffer, ptr, decoded));
        await assertDecoded(record, new Uint8Array(referenceMemory.buffer, rp, decoded));
        const [rustInner, referenceInner] = timePair(own, ref, index & 1);
        const metadata = { count, byteStride: stride, mode: modes[mode], filter: filters[filter] };
        const [rustEntry, referenceEntry] = timePair(
            () => decodeMeshoptView(index, metadata, source, output, khr !== 0),
            () => decoder.decodeGltfBuffer(output, count, stride, source, modes[mode], filters[filter]),
            index & 1,
        );
        console.log([asset, index, ...[rustInner, referenceInner, rustEntry, referenceEntry, rustEntry-rustInner, referenceEntry-referenceInner].map((v) => v.toFixed(3))].join("\t"));
    } finally {
        free(ptr, compressed + decoded);
        sbrk(rp - sbrk(0));
    }
}
