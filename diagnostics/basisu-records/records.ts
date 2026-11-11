import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { syncFiles } from "../../examples/gltf/sync";
import inputs from "../../crates/basisu/tests/records/inputs.json";

const destination = resolve(import.meta.dir, "../../test-results/basisu-assets");
const source = process.env.GLTF_SAMPLE_MODELS
    ? pathToFileURL(`${resolve(process.env.GLTF_SAMPLE_MODELS)}/`)
    : new URL(`https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`);
await syncFiles(inputs.files, source, destination);
console.log(join(destination));
