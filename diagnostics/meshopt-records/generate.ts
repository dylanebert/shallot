import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, relative } from "node:path";
import { createHash } from "node:crypto";
import { normalizePath } from "../../src/project/paths";
import { pinnedCheckout, referenceDecoder, REVISION } from "./reference";

const models = pinnedCheckout("GLTF_SAMPLE_MODELS", "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf");
const decoder = await referenceDecoder();
const destination = resolve(import.meta.dir, "../../crates/gltf/tests/meshopt");
mkdirSync(destination, { recursive: true });
const manifest = [`# meshoptimizer ${REVISION}; glTF-Sample-Assets 2bac6f8c`, "# asset view decoded sha256"];
const files = new Map<string, string>();
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
function input(path: string): Buffer {
    const bytes = readFileSync(path);
    files.set(normalizePath(relative(models, path)), digest(bytes));
    return bytes;
}
for (const [model, variants] of [
    ["MeshoptCubeTest", ["glTF", "glTF-Meshopt"]],
    ["BrainStem", ["glTF-Meshopt", "glTF-Meshopt-EXT"]],
    ["DragonAttenuation", ["glTF-Meshopt", "glTF-Meshopt-EXT"]],
] as const) {
    for (const variant of variants) {
        const path = join(models, model, variant, `${model}.gltf`);
        const gltf = JSON.parse(input(path).toString());
        for (const [index, view] of gltf.bufferViews.entries()) {
            const khr = view.extensions?.KHR_meshopt_compression;
            const ext = khr ?? view.extensions?.EXT_meshopt_compression;
            if (!ext) continue;
            const uri = gltf.buffers[ext.buffer].uri;
            const buffer = uri.startsWith("data:") ? Buffer.from(uri.split(",")[1], "base64") : input(join(dirname(path), uri));
            const compressed = buffer.subarray(ext.byteOffset ?? 0, (ext.byteOffset ?? 0) + ext.byteLength);
            const output = new Uint8Array(ext.count * ext.byteStride);
            decoder.decodeGltfBuffer(output, ext.count, ext.byteStride, compressed, ext.mode, ext.filter ?? "NONE");
            manifest.push([`${model}/${variant}`, index, output.length, digest(output)].join("\t"));
        }
    }
}
writeFileSync(join(destination, "views.tsv"), `${manifest.join("\n")}\n`);
writeFileSync(join(destination, "inputs.json"), `${JSON.stringify({
    commit: "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf", decoder: REVISION,
    files: [...files].map(([path, sha256]) => ({ path, sha256 })),
    oracle: { path: "js/meshopt_decoder.mjs", sha256: digest(readFileSync(resolve(process.env.MESHOPTIMIZER!, "js/meshopt_decoder.mjs"))) },
}, null, 2)}\n`);
console.log(`${manifest.length - 2} decoded length/SHA-256 records; ${files.size} pinned corpus files`);

// Each record contains 18 source bytes followed by three indices of the given width.
const fifo: Uint8Array[] = [];
for (const stride of [2, 4]) {
    for (const code of [0x00, 0x01, 0xf1]) {
        const source = new Uint8Array(18);
        source[0] = 0xe1;
        source[1] = code;
        source[3] = 0x11;
        const output = new Uint8Array(3 * stride);
        decoder.decodeGltfBuffer(output, 3, stride, source, "TRIANGLES", "NONE");
        fifo.push(source, output);
    }
}
writeFileSync(join(destination, "fifo.bin"), Buffer.concat(fifo));
console.log(`${fifo.length / 2} FIFO sentinel records from meshoptimizer ${REVISION}`);
