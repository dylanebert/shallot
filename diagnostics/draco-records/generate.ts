import { spawnSync } from "node:child_process";
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
add_executable(records_reference "${join(import.meta.dir, "reference.cpp")}")
target_link_libraries(records_reference PRIVATE draco_static)
target_include_directories(records_reference PRIVATE "${source}/src" "${build}/cmake")
add_executable(records_fixtures "${join(import.meta.dir, "fixtures.cpp")}")
target_link_libraries(records_fixtures PRIVATE draco_static)
target_include_directories(records_fixtures PRIVATE "${source}/src" "${build}/cmake")
`);
run("cmake", ["-S", source, "-B", join(build, "cmake"), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DDRACO_TESTS=OFF"]);
run("cmake", ["--build", join(build, "cmake"), "--target", "records_reference", "records_fixtures", "-j", "8"]);
const reference = join(build, "cmake/records_reference");
const fixtureNames = run(join(build, "cmake/records_fixtures"), [fixtures]).split("\n").sort();
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const manifest = ["# name faces length sha256 connectivity_end input_sha256 paths points"];
const files = new Map<string, string>();
const streams: { name: string; file: string; offset: number; length: number; sha256: string; primitives: string[] }[] = [];
const coverage = new Map<string, string[]>();
function record(name: string, bytes: Uint8Array) {
    const input = join(build, "stream.drc"), output = join(build, "faces.bin");
    writeFileSync(input, bytes);
    const [faces, end, paths, points] = run(reference, [input, output]).split("\t");
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
            for (const [mi, mesh] of gltf.meshes.entries()) {
                for (const [pi, primitive] of mesh.primitives.entries()) {
                    const extension = primitive.extensions?.KHR_draco_mesh_compression;
                    if (!extension) continue;
                    const names = views.get(extension.bufferView) ?? [];
                    names.push(`meshes/${mi}/primitives/${pi}`);
                    views.set(extension.bufferView, names);
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
                record(name, data);
            }
        }
    }
}
for (const name of fixtureNames.filter((name) => !name.startsWith("reject-"))) record(`fixtures/${name}`, readFileSync(join(fixtures, `${name}.drc`)));
const rejected = ["# name input_sha256 reason"];
for (const name of fixtureNames.filter((name) => name.startsWith("reject-"))) {
    const path = join(fixtures, `${name}.drc`);
    if (spawnSync(reference, [path, join(build, "invalid.bin")]).status !== 2) throw new Error(`${name}: pinned decoder unexpectedly accepted`);
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
writeFileSync(join(records, "inputs.json"), `${JSON.stringify({ commit, decoder, files: [...files].map(([path, sha256]) => ({ path, sha256 })), streams }, null, 2)}\n`);
writeFileSync(join(records, "coverage.json"), `${JSON.stringify(Object.fromEntries([...coverage].sort(([a], [b]) => a.localeCompare(b))), null, 2)}\n`);
// Pin the decoder-wins zero-value refusal independently of Shallot.
const invalidMetadata = Buffer.concat([Buffer.from([68, 82, 65, 67, 79, 2, 2, 1, 0, 0, 128]), Buffer.from([0, 1, 1, 120, 0, 0]), readFileSync(join(fixtures, "sequential-u8.drc")).subarray(11)]);
writeFileSync(join(build, "empty-metadata.drc"), invalidMetadata);
if (spawnSync(reference, [join(build, "empty-metadata.drc"), join(build, "invalid.bin")]).status !== 2) throw new Error("pinned reference did not refuse zero-length metadata");
console.log(`${modelNames.size} models, ${streams.length} unique corpus streams, ${manifest.length - 1 - streams.length} fixtures; ${coverage.size} observed paths; empty metadata refused by pin`);
