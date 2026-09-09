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
import { runtimeRoots as roots, runtimeRecord } from "../../shallot/scripts/projections";

const owner = resolve(import.meta.dir, "..");
const distribution = resolve(owner, "../shallot");
const record = resolve(distribution, runtimeRecord);
if (process.argv.includes("--check") && !existsSync(record)) {
    console.error(
        "runtime projection: missing runtime-inputs.json; run `bun run build` from the repository root",
    );
    process.exit(1);
}
const solver = resolve(owner, "../shallot-tumble");
const engine = "src/standard/tumble/engine";
const canonical = (file: string) => resolve(file.startsWith(`${engine}/`) ? solver : owner, file);
const bridge = resolve(owner, engine, "index.ts");
const bridgeContent = `export * from ${JSON.stringify(relative(dirname(bridge), resolve(solver, engine, "index")).replaceAll("\\\\", "/"))};\n`;
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
const sources = [owner, solver]
    .flatMap((root) => {
        const accepts = (file: string) =>
            carried(file) && (root === solver || !file.startsWith(`${engine}/`));
        const actual = walk(resolve(root, "src"))
            .map((file) => relative(root, file))
            .filter(accepts)
            .sort();
        // Git is independent of the filesystem: missing source cannot shrink the expected pack.
        const tracked = Bun.spawnSync(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src"],
            { cwd: root },
        );
        if (tracked.exitCode !== 0) throw Error("runtime projection: source inventory unavailable");
        const declared = [
            ...new Set(tracked.stdout.toString().trim().split("\n").filter(accepts)),
        ].sort();
        if (!actual.length || JSON.stringify(actual) !== JSON.stringify(declared))
            throw Error("runtime projection: missing or unregistered canonical source");
        return actual;
    })
    .sort();
const declared = [...new Set(sources)].sort();
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
    [
        ...copies,
        "package.json",
        "scripts/project.ts",
        "scripts/build.ts",
        "../shallot/scripts/projections.ts",
    ].map((file) => [relative(owner, canonical(file)), hash(canonical(file))]),
);
inputs["../shallot-tumble/package.json"] = hash(resolve(solver, "package.json"));
inputs["../shallot/package.json"] = hash(resolve(distribution, "package.json"));
if (process.argv.includes("--check")) {
    const bridges = walk(resolve(owner, engine));
    if (
        bridges.length !== 1 ||
        bridges[0] !== bridge ||
        lstatSync(bridge).isSymbolicLink() ||
        readFileSync(bridge, "utf8") !== bridgeContent
    )
        throw Error("runtime projection: missing, duplicate or misbound solver bridge");
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
                : readFileSync(canonical(file));
        if (!readFileSync(output).equals(content) || hash(output) !== expected.outputs[file])
            throw Error(`runtime projection: stale or misbound output ${file}`);
    }
    console.log(
        `runtime projection: ${mode}, ${forwards.length} declared targets, ${files.length} outputs fresh`,
    );
} else {
    rmSync(resolve(owner, engine), { recursive: true, force: true });
    mkdirSync(dirname(bridge), { recursive: true });
    writeFileSync(bridge, bridgeContent);
    for (const path of roots) rmSync(resolve(distribution, path), { recursive: true, force: true });
    for (const file of files) {
        const dest = resolve(distribution, file);
        mkdirSync(dirname(dest), { recursive: true });
        if (mode === "development" && forwards.includes(file)) writeFileSync(dest, forward(file));
        else cpSync(canonical(file), dest);
    }
    const outputs = Object.fromEntries(
        files.map((file) => [file, hash(resolve(distribution, file))]),
    );
    writeFileSync(record, JSON.stringify({ mode, inputs, outputs }, null, 2));
    console.log(
        `runtime projection: ${mode}, ${forwards.length} declared targets, ${files.length} outputs`,
    );
}
