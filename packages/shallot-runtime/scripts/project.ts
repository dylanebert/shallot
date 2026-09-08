import { createHash } from "node:crypto";
import {
    cpSync,
    existsSync,
    lstatSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { dirname, relative, resolve } from "node:path";

const owner = resolve(import.meta.dir, "..");
const distribution = resolve(owner, "../shallot");
const manifest = JSON.parse(readFileSync(resolve(distribution, "package.json"), "utf8"));
const runtime = JSON.parse(readFileSync(resolve(owner, "package.json"), "utf8"));
const hash = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");
const carried = (file: string) =>
    !/\.(?:test|fixture)\.ts$|\.gold\.json$|^src\/extras\/gltf\/fixtures(?:\/|$)/.test(file);
const walk = (root: string): string[] =>
    !existsSync(root)
        ? []
        : readdirSync(root, { recursive: true, withFileTypes: true })
              .filter((entry) => entry.isFile() || entry.isSymbolicLink())
              .map((entry) => resolve(entry.parentPath, entry.name));
const sources = walk(resolve(owner, "src"))
    .map((file) => relative(owner, file))
    .filter(carried)
    .sort();
// Git's source population is independent of filesystem enumeration: missing input cannot shrink a pack.
const tracked = Bun.spawnSync(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src"],
    { cwd: owner },
);
if (tracked.exitCode !== 0) throw Error("runtime projection: source inventory unavailable");
const declared = [...new Set(tracked.stdout.toString().trim().split("\n").filter(carried))].sort();
if (!sources.length || JSON.stringify(sources) !== JSON.stringify(declared))
    throw Error("runtime projection: missing or unregistered canonical source");
const assets = [
    "rust/audio/pkg/shallot_audio.js",
    "rust/audio/pkg/shallot_audio.d.ts",
    "rust/audio/pkg/shallot_audio.wasm",
];
const copies = [...sources, ...assets].sort();
const targets = Object.values(manifest.exports)
    .filter(
        (target): target is string =>
            typeof target === "string" && target !== "./src/harness/index.ts",
    )
    .map((target) => target.replace(/^\.\//, ""));
const forwards = [
    ...new Set([
        ...targets,
        "src/harness/runtime.ts",
        "src/engine/runtime/gpu.ts",
        "src/engine/runtime/log.ts",
        "src/extras/profile/benchmark.ts",
        "src/harness/degraded-boot.ts",
    ]),
].sort();
if (
    targets.length !== 24 ||
    forwards.length !== 29 ||
    forwards.some((file) => !sources.includes(file))
)
    throw Error("runtime projection: declared forwarding population changed");
if (manifest.version !== runtime.version) throw Error("runtime projection: version mismatch");
const roots = [
    "src/index.ts",
    "src/engine",
    "src/standard",
    "src/extras",
    "src/types",
    "src/harness/runtime.ts",
    "src/harness/pixels.ts",
    "src/harness/motion.ts",
    "src/harness/degraded-boot.ts",
    "rust/audio/pkg",
];
const mode = process.argv.includes("--pack") ? "pack" : "development";
const files = mode === "pack" ? copies : [...forwards, ...assets].sort();
const forward = (file: string) => {
    let target = relative(dirname(resolve(distribution, file)), resolve(owner, file)).replace(
        /\.ts$/,
        "",
    );
    if (!target.startsWith(".")) target = `./${target}`;
    return `export * from ${JSON.stringify(target)};\n`;
};
const inputs = Object.fromEntries(
    [...copies, "package.json", "scripts/project.ts", "scripts/build.ts"].map((file) => [
        file,
        hash(resolve(owner, file)),
    ]),
);
inputs["../shallot/package.json"] = hash(resolve(distribution, "package.json"));
const record = resolve(distribution, "runtime-inputs.json");
if (process.argv.includes("--check")) {
    const expected = JSON.parse(readFileSync(record, "utf8"));
    if (expected.mode !== mode) throw Error(`runtime projection: expected ${mode} mode`);
    if (JSON.stringify(expected.inputs) !== JSON.stringify(inputs))
        throw Error("runtime projection: stale canonical inputs");
    if (JSON.stringify(Object.keys(expected.outputs).sort()) !== JSON.stringify(files))
        throw Error("runtime projection: output record population mismatch");
    const actual = roots
        .flatMap((path) => {
            const absolute = resolve(distribution, path);
            if (!existsSync(absolute)) return [];
            if (lstatSync(absolute).isSymbolicLink())
                throw Error(`runtime projection: unexpected link ${path}`);
            return lstatSync(absolute).isDirectory()
                ? walk(absolute).map((file) => relative(distribution, file))
                : [path];
        })
        .sort();
    if (JSON.stringify(actual) !== JSON.stringify(files))
        throw Error("runtime projection: missing or extra output");
    for (const file of files) {
        const output = resolve(distribution, file);
        if (lstatSync(output).isSymbolicLink())
            throw Error(`runtime projection: unexpected link ${file}`);
        const content =
            mode === "development" && forwards.includes(file)
                ? Buffer.from(forward(file))
                : readFileSync(resolve(owner, file));
        if (!readFileSync(output).equals(content) || hash(output) !== expected.outputs[file])
            throw Error(`runtime projection: stale or misbound output ${file}`);
    }
    console.log(
        `runtime projection: ${mode}, ${forwards.length} declared targets, ${files.length} outputs fresh`,
    );
} else {
    for (const path of roots) rmSync(resolve(distribution, path), { recursive: true, force: true });
    for (const file of files) {
        const dest = resolve(distribution, file);
        mkdirSync(dirname(dest), { recursive: true });
        if (mode === "development" && forwards.includes(file)) writeFileSync(dest, forward(file));
        else cpSync(resolve(owner, file), dest);
    }
    const outputs = Object.fromEntries(
        files.map((file) => [file, hash(resolve(distribution, file))]),
    );
    writeFileSync(record, JSON.stringify({ mode, inputs, outputs }, null, 2));
    console.log(
        `runtime projection: ${mode}, ${forwards.length} declared targets, ${files.length} outputs`,
    );
}
