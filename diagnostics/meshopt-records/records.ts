import { createHash } from "node:crypto";
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, join, dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { syncFiles, type AssetFile } from "../../examples/gltf/sync";
import { referenceDecoder } from "./reference";
import inputs from "../../crates/gltf/tests/meshopt/inputs.json";
import { normalizePath } from "../../src/project/paths";

const root = resolve(import.meta.dir, "../..");
const cache = join(root, "test-results/meshopt-assets");
const source = process.env.GLTF_SAMPLE_MODELS
    ? pathToFileURL(`${resolve(process.env.GLTF_SAMPLE_MODELS)}/`)
    : new URL(`https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`);
export const modes = ["ATTRIBUTES", "TRIANGLES", "INDICES"];
export const filters = ["NONE", "OCTAHEDRAL", "QUATERNION", "EXPONENTIAL", "COLOR"];
export const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

export async function readInput(file: AssetFile, destination = cache, origin = source): Promise<Buffer> {
    try {
        await syncFiles([file], origin, destination);
        return readFileSync(join(destination, file.path));
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") throw new Error(`${file.path}: missing pinned corpus input`, { cause: error });
        throw error;
    }
}

export async function loadRecords() {
    const files = new Map(await Promise.all(inputs.files.map(async (file) => [file.path, await readInput(file)] as const)));
    const manifest = readFileSync(join(root, "crates/gltf/tests/meshopt/views.tsv"), "utf8");
    return manifest.split("\n").filter((line) => line && !line.startsWith("#")).map((line) => {
        const [asset, id, length, sha256] = line.split("\t");
        const index = Number(id);
        const gltfPath = `${asset}/${asset.split("/")[0]}.gltf`;
        const gltf = JSON.parse(files.get(gltfPath)!.toString());
        const view = gltf.bufferViews[index];
        const khr = view.extensions.KHR_meshopt_compression;
        const ext = khr ?? view.extensions.EXT_meshopt_compression;
        const uri = gltf.buffers[ext.buffer].uri;
        const buffer = uri.startsWith("data:") ? Buffer.from(uri.split(",")[1], "base64") : files.get(normalizePath(join(dirname(gltfPath), uri)))!;
        const source = buffer.subarray(ext.byteOffset ?? 0, (ext.byteOffset ?? 0) + ext.byteLength);
        if (source.length !== ext.byteLength || ext.count * ext.byteStride !== Number(length)) throw new Error(`${asset} bufferView ${index}: recorded byteLength mismatch`);
        return { asset, index, count: ext.count as number, stride: ext.byteStride as number,
            mode: modes.indexOf(ext.mode), filter: filters.indexOf(ext.filter ?? "NONE"), khr: khr ? 1 : 0,
            source, length: Number(length), sha256 };
    });
}
export type Record = Awaited<ReturnType<typeof loadRecords>>[number];

async function oracle() {
    if (process.env.MESHOPTIMIZER) return referenceDecoder();
    const destination = join(root, "test-results/meshopt-oracle");
    await syncFiles([inputs.oracle], new URL(`https://raw.githubusercontent.com/zeux/meshoptimizer/${inputs.decoder}/`), destination);
    const { MeshoptDecoder } = await import(pathToFileURL(join(destination, inputs.oracle.path)).href);
    await MeshoptDecoder.ready;
    return MeshoptDecoder;
}

export async function referenceOutput(record: Record): Promise<Uint8Array> {
    const decoder = await oracle();
    const output = new Uint8Array(record.length);
    decoder.decodeGltfBuffer(output, record.count, record.stride, record.source, modes[record.mode], filters[record.filter]);
    return output;
}

/** Only a mismatch loads the pinned generator's decoder; passing checks use recorded hashes. */
export async function assertDecoded(record: Record, output: Uint8Array): Promise<void> {
    const actual = digest(output);
    if (output.length === record.length && actual === record.sha256) return;
    const expected = await referenceOutput(record);
    let offset = 0;
    while (offset < Math.min(output.length, expected.length) && output[offset] === expected[offset]) offset++;
    const difference = offset === output.length && offset === expected.length ? "none (output matches pinned decoder; regenerate the hash record)" : String(offset);
    throw new Error(`${record.asset} bufferView ${record.index}: SHA-256 mismatch: expected ${record.sha256}, received ${actual}; first differing offset ${difference}`);
}

if (import.meta.main) {
    const records = await loadRecords();
    if (process.argv[2] === "--compare") {
        const record = records.find((r) => r.asset === process.argv[3] && r.index === Number(process.argv[4]));
        if (!record) throw new Error(`unknown recorded view: ${process.argv[3]} ${process.argv[4]}`);
        await assertDecoded(record, readFileSync(process.argv[5]));
    } else {
        const destination = join(root, "test-results/meshopt-native", String(process.pid));
        mkdirSync(destination, { recursive: true });
        let offset = 0;
        const lines = records.map((r) => {
            const line = [r.asset, r.index, r.count, r.stride, r.mode, r.filter, r.khr, offset, r.source.length, r.length, r.sha256].join("\t");
            offset += r.source.length;
            return line;
        });
        writeFileSync(join(destination, "sources.bin"), Buffer.concat(records.map((r) => r.source)));
        writeFileSync(join(destination, "views.tsv"), lines.join("\n"));
        console.log(destination);
    }
}
