import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import inputs from "../../crates/draco/tests/records/inputs.json";
import { syncFiles } from "../../examples/gltf/sync";

const destination = resolve(import.meta.dir, "../../test-results/draco-assets");
const streams = resolve(import.meta.dir, "../../test-results/draco-streams");
const source = process.env.GLTF_SAMPLE_MODELS
    ? pathToFileURL(`${resolve(process.env.GLTF_SAMPLE_MODELS)}/`)
    : new URL(`https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`);
await syncFiles(inputs.files, source, destination);
for (const stream of inputs.streams) {
    const bytes = (await readFile(join(destination, stream.file))).subarray(stream.offset, stream.offset + stream.length);
    const hash = createHash("sha256").update(bytes).digest("hex");
    if (bytes.length !== stream.length || hash !== stream.sha256) throw new Error(`${stream.name}: extracted stream SHA-256 mismatch`);
    const path = join(streams, `${stream.name}.drc`);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, bytes);
}
console.log(streams);
