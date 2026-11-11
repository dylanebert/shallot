import { cpus } from "node:os";
import { resolve } from "node:path";
import { decodeMeshoptView } from "../../src/extras/gltf/kernel";
import { gltfArtifactIsFresh } from "../../crates/gltf/scripts/inputs";
import { referenceDecoder } from "./reference";
import { timePair } from "./timing";
import { loadRecords, assertDecoded, modes, filters } from "./records";
const trial = Number(process.argv[process.argv.indexOf("--trial") + 1]) || 0;
const focus = process.argv.includes("--focus") ? new Set([
    "MeshoptCubeTest/glTF:23", "MeshoptCubeTest/glTF:33",
    "BrainStem/glTF-Meshopt-EXT:7", "BrainStem/glTF-Meshopt-EXT:4",
    "DragonAttenuation/glTF-Meshopt-EXT:2", "DragonAttenuation/glTF-Meshopt-EXT:3",
]) : undefined;

const root = resolve(".");
if (!gltfArtifactIsFresh(root)) throw new Error("rebuild with bun crates/gltf/scripts/build.ts");
const decoder = await referenceDecoder();
const records = await loadRecords();
console.log(`${cpus()[0]?.model}; ${process.platform}/${process.arch}; ${process.versions.bun ? `Bun ${process.versions.bun}` : `Node ${process.version}`}`);
console.log(`Trial ${trial}: 1000 warmups per side; common iteration count calibrated until both batches take >=5ms; median of 8 batches with alternating first side. Microseconds/decode.`);
console.log("Both include WASM allocation, input copy, decoding/filter, output copy, free; exclude JS output allocation.");
console.log("asset\tview\tShallot_us\treference_us\tratio");
for (const record of records) {
    const { asset, index: view, count, stride, mode, filter, khr, source, length: decodedLength } = record;
    if (focus && !focus.has(`${asset}:${view}`)) continue;
    const output = new Uint8Array(decodedLength);
    const metadata = { count, byteStride: stride, mode: modes[mode], filter: filters[filter] };
    const decode = () => decodeMeshoptView(view, metadata, source, output, khr !== 0);
    decode();
    await assertDecoded(record, output);
    const reference = () => decoder.decodeGltfBuffer(output, count, stride, source, modes[mode], filters[filter]);
    reference();
    await assertDecoded(record, output);
    const [a, b] = timePair(decode, reference, (trial + view) & 1);
    console.log(`${asset}\t${view}\t${a.toFixed(3)}\t${b.toFixed(3)}\t${a / b}`);
}
