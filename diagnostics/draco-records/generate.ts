import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { instrument } from "./instrument";

const decoder = "554628821c3448319d7d4ae48db170b8c63e1aaf";
const commit = "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf";
const root = resolve(import.meta.dir, "../..");
const build = join(root, "test-results/draco-reference");
const records = join(root, "crates/draco/tests/records");
const fixtures = join(root, "crates/draco/tests/fixtures");
function run(command: string, args: string[]) {
    const result = spawnSync(command, args, { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
    if (result.status !== 0) throw new Error(`${command}: ${result.stdout}${result.stderr}`);
    return result.stdout.trim();
}
function checkout(variable: string, pin: string) {
    const path = process.env[variable];
    if (!path || run("git", ["-C", path, "rev-parse", "HEAD"]) !== pin || run("git", ["-C", path, "status", "--porcelain", "--untracked-files=no"])) throw new Error(`set ${variable} to a clean checkout at ${pin}`);
    return resolve(path);
}
const draco = checkout("DRACO", decoder);
const models = checkout("GLTF_SAMPLE_MODELS", commit);
for (const dir of [build, records, fixtures]) mkdirSync(dir, { recursive: true });
const source = join(build, "source");
// Never patch the reference checkout. Every run starts with the pinned archive.
rmSync(source, { recursive: true, force: true });
mkdirSync(source);
run("git", ["-C", draco, "archive", "--format=tar", `--output=${build}/source.tar`, decoder]);
run("tar", ["-xf", join(build, "source.tar"), "-C", source]);
instrument(source);
appendFileSync(join(source, "CMakeLists.txt"), `
add_executable(records_reference "${join(import.meta.dir, "reference.cpp")}" "${source}/src/draco/javascript/emscripten/decoder_webidl_wrapper.cc")
target_link_libraries(records_reference PRIVATE draco_static)
target_include_directories(records_reference PRIVATE "${source}/src" "${build}/cmake")
add_executable(records_fixtures "${join(import.meta.dir, "fixtures.cpp")}")
target_link_libraries(records_fixtures PRIVATE draco_static)
target_include_directories(records_fixtures PRIVATE "${source}/src" "${build}/cmake")
`);
// Apple clang contracts a * b + c into fused multiply-adds by default, which
// changes dequantized normals in their last bit. Google's shipped
// draco_decoder_gltf.wasm does not contract; checkWasm below holds the records
// to it.
run("cmake", ["-S", source, "-B", join(build, "cmake"), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DDRACO_TESTS=OFF", "-DCMAKE_CXX_FLAGS=-ffp-contract=off"]);
run("cmake", ["--build", join(build, "cmake"), "--target", "records_reference", "records_fixtures", "-j", "8"]);
const reference = join(build, "cmake/records_reference");
// Google's committed glTF decoder build, the one DRACOLoader ships.
const wasm = await createRequire(import.meta.url)(join(source, "javascript/draco_wasm_wrapper_gltf.js"))({ wasmBinary: readFileSync(join(source, "javascript/draco_decoder_gltf.wasm")) });
function checkWasm(name: string, bytes: Uint8Array, expected: [number, number, string | undefined][]) {
    const decoder = new wasm.Decoder();
    const mesh = new wasm.Mesh();
    const status = decoder.DecodeArrayToMesh(bytes, bytes.length, mesh);
    if (!status.ok()) throw new Error(`${name}: draco_decoder_gltf.wasm refused: ${status.error_msg()}`);
    for (const [index, type, sha256] of expected) {
        const attribute = decoder.GetAttribute(mesh, index);
        const size = mesh.num_points() * attribute.num_components() * [0, 1, 1, 2, 2, 4, 4, 8, 8, 4][type];
        const pointer = wasm._malloc(size);
        const accepted = decoder.GetAttributeDataArrayForAllPoints(mesh, attribute, type, size, pointer);
        const actual = accepted ? digest(new Uint8Array(wasm.HEAPU8.buffer, pointer, size)) : undefined;
        wasm._free(pointer);
        if (actual !== sha256) throw new Error(`${name} attribute ${index} as data type ${type}: draco_decoder_gltf.wasm gives ${actual}, native gives ${sha256}`);
    }
    wasm.destroy(mesh);
    wasm.destroy(decoder);
}
const fixtureNames = run(join(build, "cmake/records_fixtures"), [fixtures]).split("\n").sort();
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const manifest = ["# name faces length sha256 connectivity_end input_sha256 paths points"];
const files = new Map<string, string>();
const streams: { name: string; file: string; offset: number; length: number; sha256: string; primitives: string[] }[] = [];
const coverage = new Map<string, string[]>();
const attributeRows = ["# name attribute unique_id type data_type components normalized output_type length sha256"];
// glTF accessor componentType to Draco DataType.
const componentTypes: Record<number, number> = { 5120: 1, 5121: 2, 5122: 3, 5123: 4, 5125: 6, 5126: 9 };
const outputTypes = [1, 2, 3, 4, 5, 6, 9];
/** `asked` maps unique ids to the Draco types the corpus's accessors request; fixtures request every output type. */
function record(name: string, bytes: Uint8Array, asked?: Map<number, Set<number>>) {
    const input = join(build, "stream.drc"), output = join(build, "faces.bin"), values = join(build, "attributes");
    writeFileSync(input, bytes);
    rmSync(values, { recursive: true, force: true });
    mkdirSync(values);
    const [head, ...attributes] = run(reference, [input, output, values]).split("\n");
    const [faces, end, paths, points] = head.split("\t");
    const wasmExpected: [number, number, string | undefined][] = [];
    for (const line of attributes) {
        const [index, uniqueId, type, dataType, components, normalized, accepted] = line.split("\t");
        const types = new Set([Number(dataType), ...(asked ? (asked.get(Number(uniqueId)) ?? []) : outputTypes)]);
        for (const outputType of types) {
            const accepts = accepted.split(",").filter(Boolean).map(Number).includes(outputType);
            const data = accepts ? readFileSync(join(values, `${index}-${outputType}.bin`)) : undefined;
            attributeRows.push([name, index, uniqueId, type, dataType, components, normalized, outputType, data ? data.length : "refused", data ? digest(data) : "refused"].join("\t"));
            if (outputTypes.includes(outputType)) wasmExpected.push([Number(index), outputType, data && digest(data)]);
        }
    }
    // The glTF build omits DRACO_BACKWARDS_COMPATIBILITY; fixtures of deprecated schemes are native-only.
    if (asked) checkWasm(name, bytes, wasmExpected);
    const decoded = readFileSync(output);
    if (Number(faces) * 12 !== decoded.length || Number(end) < 11 || !paths || !points || !Number.isSafeInteger(Number(points)) || Number(points) < 0) throw new Error(`${name}: invalid reference output`);
    manifest.push([name, faces, decoded.length, digest(decoded), end, digest(bytes), paths, points].join("\t"));
    for (const path of paths.split(",")) {
        const names = coverage.get(path) ?? [];
        names.push(name);
        coverage.set(path, names);
    }
}
function input(path: string) {
    const bytes = readFileSync(join(models, path));
    files.set(path, digest(bytes));
    return bytes;
}
const modelNames = new Set<string>();
for (const model of readdirSync(models).sort()) {
    if (!readdirSync(models, { withFileTypes: true }).some((entry) => entry.name === model && entry.isDirectory())) continue;
    for (const variant of readdirSync(join(models, model), { withFileTypes: true }).filter((e) => e.isDirectory() && e.name.includes("Draco")).map((e) => e.name).sort()) {
        const directory = `${model}/${variant}`;
        for (const file of readdirSync(join(models, directory)).filter((f) => /\.(gltf|glb)$/.test(f)).sort()) {
            const filePath = `${directory}/${file}`;
            const bytes = input(filePath);
            let json: Buffer, glbOffset = 0;
            if (file.endsWith(".glb")) {
                if (bytes.toString("ascii", 0, 4) !== "glTF" || bytes.readUInt32LE(4) !== 2 || bytes.readUInt32LE(8) !== bytes.length) throw new Error(`${file}: invalid GLB`);
                const jsonLength = bytes.readUInt32LE(12);
                json = bytes.subarray(20, 20 + jsonLength);
                glbOffset = 20 + jsonLength + 8;
                if (bytes.readUInt32LE(glbOffset - 4) !== 0x004e4942) throw new Error(`${file}: no BIN chunk`);
            } else json = bytes;
            const gltf = JSON.parse(json.toString("utf8"));
            const views = new Map<number, string[]>();
            const asked = new Map<number, Map<number, Set<number>>>();
            for (const [mi, mesh] of gltf.meshes.entries()) {
                for (const [pi, primitive] of mesh.primitives.entries()) {
                    const extension = primitive.extensions?.KHR_draco_mesh_compression;
                    if (!extension) continue;
                    const names = views.get(extension.bufferView) ?? [];
                    names.push(`meshes/${mi}/primitives/${pi}`);
                    views.set(extension.bufferView, names);
                    const types = asked.get(extension.bufferView) ?? new Map<number, Set<number>>();
                    for (const [semantic, uniqueId] of Object.entries(extension.attributes as Record<string, number>)) {
                        const accessor = gltf.accessors[primitive.attributes[semantic]];
                        const type = accessor && componentTypes[accessor.componentType];
                        if (!type) throw new Error(`${filePath} meshes/${mi}/primitives/${pi} ${semantic}: no accessor component type`);
                        types.set(uniqueId, (types.get(uniqueId) ?? new Set()).add(type));
                    }
                    asked.set(extension.bufferView, types);
                }
            }
            for (const [id, primitives] of [...views].sort((a, b) => a[0] - b[0])) {
                const view = gltf.bufferViews[id];
                const uri = gltf.buffers[view.buffer].uri;
                const container = uri ? `${directory}/${decodeURIComponent(uri)}` : filePath;
                const offset = (view.byteOffset ?? 0) + (uri ? 0 : glbOffset);
                const data = input(container).subarray(offset, offset + view.byteLength);
                if (data.length !== view.byteLength) throw new Error(`${file} bufferView ${id}: invalid byte range`);
                const name = `${directory}/${file}/view-${id}`;
                streams.push({ name, file: container, offset, length: data.length, sha256: digest(data), primitives });
                modelNames.add(model);
                record(name, data, asked.get(id));
            }
        }
    }
}
for (const name of fixtureNames.filter((name) => !name.startsWith("reject-"))) record(`fixtures/${name}`, readFileSync(join(fixtures, `${name}.drc`)));
const rejected = ["# name input_sha256 reason"];
for (const name of fixtureNames.filter((name) => name.startsWith("reject-"))) {
    const path = join(fixtures, `${name}.drc`);
    if (spawnSync(reference, [path, join(build, "invalid.bin"), build]).status !== 2) throw new Error(`${name}: pinned decoder unexpectedly accepted`);
    rejected.push([`fixtures/${name}`, digest(readFileSync(path)), "sequential face-count bound exceeds remaining bytes / 3"].join("\t"));
}
writeFileSync(join(records, "rejections.tsv"), `${rejected.join("\n")}\n`);
for (const model of [...modelNames].sort()) {
    const license = "LICENSE.md";
    const path = join(root, "crates/draco/licenses", model, license);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, readFileSync(join(models, model, license)));
}
writeFileSync(join(records, "faces.tsv"), `${manifest.join("\n")}\n`);
writeFileSync(join(records, "attributes.tsv"), `${attributeRows.join("\n")}\n`);
writeFileSync(join(records, "inputs.json"), `${JSON.stringify({ commit, decoder, files: [...files].map(([path, sha256]) => ({ path, sha256 })), streams }, null, 2)}\n`);
writeFileSync(join(records, "coverage.json"), `${JSON.stringify(Object.fromEntries([...coverage].sort(([a], [b]) => a.localeCompare(b))), null, 2)}\n`);
// Pin the decoder-wins zero-value refusal independently of Shallot.
const invalidMetadata = Buffer.concat([Buffer.from([68, 82, 65, 67, 79, 2, 2, 1, 0, 0, 128]), Buffer.from([0, 1, 1, 120, 0, 0]), readFileSync(join(fixtures, "sequential-u8.drc")).subarray(11)]);
writeFileSync(join(build, "empty-metadata.drc"), invalidMetadata);
if (spawnSync(reference, [join(build, "empty-metadata.drc"), join(build, "invalid.bin"), build]).status !== 2) throw new Error("pinned reference did not refuse zero-length metadata");
console.log(`${modelNames.size} models, ${streams.length} unique corpus streams, ${manifest.length - 1 - streams.length} fixtures; ${coverage.size} observed paths; empty metadata refused by pin`);
