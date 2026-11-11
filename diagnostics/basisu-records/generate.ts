import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { normalizePath } from "../../src/project/paths";

const revision = "073eb6a0a34a23d77e75e54f41dcd9653a52da83";
const assetsRevision = "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf";
function run(command: string, args: string[]) {
    const result = spawnSync(command, args, { encoding: "utf8" });
    if (result.status !== 0) throw new Error(`${command}: ${result.stdout}${result.stderr}`);
    return result.stdout.trim();
}
function checkout(variable: string, pin: string) {
    const path = process.env[variable];
    if (!path || run("git", ["-C", path, "rev-parse", "HEAD"]) !== pin || run("git", ["-C", path, "status", "--porcelain", "--untracked-files=no"])) throw new Error(`set ${variable} to a clean checkout at ${pin}`);
    return resolve(path);
}
const source = checkout("BASIS_UNIVERSAL", revision);
const assets = checkout("GLTF_SAMPLE_MODELS", assetsRevision);
const root = resolve(import.meta.dir, "../..");
const build = join(root, "test-results/basisu-reference");
const destination = join(root, "crates/basisu/tests/records");
const fixtures = join(root, "crates/basisu/tests/fixtures");
for (const path of [build, destination, fixtures]) mkdirSync(path, { recursive: true });
run("cmake", ["-S", source, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DBASISU_OPENCL=OFF"]);
run("cmake", ["--build", build, "-j", "8"]);
run("clang", ["-O2", "-c", join(source, "zstd/zstddeclib.c"), "-o", join(build, "zstd.o")]);
const reference = join(build, "reference");
// WebAssembly's scalar arithmetic does not contract multiply/add into FMA.
run("clang++", ["-std=c++17", "-O2", "-ffp-contract=off", `-I${source}/transcoder`, `-I${source}/zstd`, join(import.meta.dir, "reference.cpp"), join(source, "transcoder/basisu_transcoder.cpp"), join(build, "zstd.o"), "-o", reference]);

// Deterministic 8x8 TGA; the pinned encoder does not infer red/RG DFD channel
// IDs from swizzles, so only those metadata IDs are assigned after encoding.
const tga = Buffer.alloc(18 + 8 * 8 * 4);
tga[2] = 2; tga.writeUInt16LE(8, 12); tga.writeUInt16LE(8, 14); tga[16] = 32; tga[17] = 0x28;
for (let i = 0; i < 64; i++) {
    tga[18 + i * 4] = (i * 13) & 255;
    tga[19 + i * 4] = (i * 31) & 255;
    tga[20 + i * 4] = (i * 7) & 255;
    tga[21 + i * 4] = 255;
}
const image = join(build, "source.tga");
writeFileSync(image, tga);
const constant = Buffer.from(tga);
for (let i = 0; i < 64; i++) constant.set([43, 191, 127, 255], 18 + i * 4);
const constantImage = join(build, "constant.tga");
writeFileSync(constantImage, constant);
const fixtureNames: string[] = [];
for (const [name, args, channels] of [
    ["uastc-none", ["-uastc", "-ktx2_no_zstandard"], [0]],
    ["etc1s-red", ["-swizzle", "rrra", "-no_alpha"], [3]],
    ["etc1s-rg", ["-swizzle", "rrrg"], [3, 4]],
    ["uastc-red", ["-uastc", "-swizzle", "rrra", "-no_alpha"], [4]],
    ["uastc-rg", ["-uastc", "-swizzle", "rgba", "-no_alpha"], [6]],
] as const) {
    const path = join(fixtures, `${name}.ktx2`);
    run(join(source, "bin/basisu"), ["-file", name === "uastc-red" ? constantImage : image, "-output_file", path, "-ktx2", "-linear", ...args]);
    const bytes = readFileSync(path);
    const dfd = bytes.readUInt32LE(48);
    if (bytes.readUInt32LE(52) !== 28 + channels.length * 16) throw new Error(`${name}: unexpected sample count`);
    channels.forEach((channel, i) => { bytes[dfd + 31 + i * 16] = channel; });
    writeFileSync(path, bytes);
    fixtureNames.push(name);
}
const digest = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const manifest: string[] = ["# asset level length sha256"];
const transcodes: string[] = ["# asset level target length sha256"];
const files: { path: string; sha256: string }[] = [];
function record(path: string, name: string) {
    const bytes = readFileSync(path);
    const count = bytes.readUInt32LE(40);
    for (let level = 0; level < count; level++) {
        const output = join(build, "level.bin");
        run(reference, [path, String(level), output]);
        const decoded = readFileSync(output);
        manifest.push([name, level, decoded.length, digest(decoded)].join("\t"));
        unlinkSync(output);
        if (bytes.readUInt32LE(44) === 1) {
            for (const target of ["etc1", "etc2-rgba", "bc7", "rgba8"]) {
                run(reference, [path, String(level), output, target]);
                const decoded = readFileSync(output);
                transcodes.push([name, level, target, decoded.length, digest(decoded)].join("\t"));
                unlinkSync(output);
            }
        }
    }
}
for (const model of ["StainedGlassLamp", "AnisotropyBarnLamp"]) {
    const relative = `${model}/glTF-KTX-BasisU`;
    for (const file of readdirSync(join(assets, relative)).filter((file) => file.endsWith(".ktx2")).sort()) {
        const path = join(assets, relative, file);
        files.push({ path: normalizePath(join(relative, file)), sha256: digest(readFileSync(path)) });
        record(path, `${model}/${file.slice(0, -5)}`);
    }
}
for (const name of fixtureNames) record(join(fixtures, `${name}.ktx2`), `fixtures/${name}`);
// RFC 8878 §3.1.1.1.1: the decoder ignores bit 4 but refuses reserved bit 3.
// Both variants use the same encoded fixture and pinned C decompressor as its record.
const frameFixture = readFileSync(join(fixtures, "uastc-red.ktx2"));
const frameOffset = Number(frameFixture.readBigUInt64LE(80));
const frameInput = join(build, "frame-header.ktx2");
const frameOutput = join(build, "frame-header.bin");
const frameRecord = manifest.find((line) => line.startsWith("fixtures/uastc-red\t0\t"))!.split("\t");
for (const bit of [0x10, 0x08]) {
    const variant = Buffer.from(frameFixture);
    variant[frameOffset + 4] |= bit;
    writeFileSync(frameInput, variant);
    if (bit === 0x10) {
        run(reference, [frameInput, "0", frameOutput]);
        const decoded = readFileSync(frameOutput);
        if (decoded.length !== Number(frameRecord[2]) || digest(decoded) !== frameRecord[3]) throw new Error("unused frame bit changed the pinned reference output");
        unlinkSync(frameOutput);
    } else {
        const result = spawnSync(reference, [frameInput, "0", frameOutput]);
        if (result.status !== 3) throw new Error(`reserved frame bit: expected pinned ZSTD_decompress refusal, received ${result.status}`);
    }
}
unlinkSync(frameInput);
console.log("pinned reference: unused frame bit matches uastc-red level 0 digest; reserved bit refused");
writeFileSync(join(destination, "levels.tsv"), `${manifest.join("\n")}\n`);
writeFileSync(join(destination, "etc1s.tsv"), `${transcodes.join("\n")}\n`);
writeFileSync(join(destination, "inputs.json"), `${JSON.stringify({ commit: assetsRevision, decoder: revision, files }, null, 2)}\n`);
console.log(`${manifest.length - 1} recorded levels`);
