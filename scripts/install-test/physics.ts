import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    cpSync,
    existsSync,
    lstatSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    renameSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "../..");
const enginePath = "src/standard/physics";
const canonical = resolve(root, "packages/shallot-physics", enginePath);
const hash = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");

function exec(
    dir: string,
    name: string,
    cmd: string[],
    pass = true,
    message?: RegExp,
    cwd = dir,
): string {
    const result = Bun.spawnSync(cmd, {
        cwd,
        stdout: "pipe",
        stderr: "pipe",
        timeout: 120000,
    });
    const out = result.stdout.toString() + result.stderr.toString();
    writeFileSync(join(dir, `${name}.log`), out);
    writeFileSync(
        join(dir, `${name}.json`),
        JSON.stringify({ cmd, cwd, exit: result.exitCode, signal: result.signalCode }),
    );
    assert(!result.signalCode, `${name}: signal ${result.signalCode}`);
    assert.equal(result.exitCode === 0, pass, `${name}: ${out}`);
    if (message) assert.match(out, message, name);
    console.log(`physics: ${name} exit=${result.exitCode}`);
    return out;
}

// Libarchive normally consumes AppleDouble members even in list mode on macOS. Disable that
// interpretation, not those entries: the inventory must expose every physical archive member.
export function archiveMembers(file: string): { path: string; type: string }[] {
    const options = process.platform === "darwin" ? ["--options", "!mac-ext"] : [];
    const list = (verbose: boolean) => {
        const result = Bun.spawnSync(["tar", ...options, verbose ? "-tvf" : "-tf", file], {
            stdout: "pipe",
            stderr: "pipe",
        });
        assert.equal(result.exitCode, 0, result.stderr.toString());
        return result.stdout.toString().trimEnd().split("\n");
    };
    const names = list(false);
    const details = list(true);
    assert.equal(names.length, details.length, "archive inventory: names/types population");
    return names.map((path, i) => ({ path, type: details[i][0] }));
}

export function checkMembers(members: ReturnType<typeof archiveMembers>, expected: string[]): void {
    assert(members.length > 0 && expected.length > 0, "archive inventory: empty population");
    const seen = new Set<string>();
    const files: string[] = [];
    for (const { path, type } of members) {
        assert(!seen.has(path), `archive inventory: duplicate member ${path}`);
        seen.add(path);
        assert(type === "-" || type === "d", `archive inventory: unexpected type ${type} ${path}`);
        assert(
            !path.split("/").some((part) => part.startsWith("._")),
            `archive inventory: platform metadata ${path}`,
        );
        if (type === "d") {
            assert(
                path.endsWith("/") && expected.some((file) => file.startsWith(path)),
                `archive inventory: unexpected directory ${path}`,
            );
        } else files.push(path);
    }
    assert.deepEqual(
        files.sort(),
        [...expected].sort(),
        "archive inventory: regular member population",
    );
}

/** Pack the canonical private solver, then realize it once inside the public tarball. */
export function projectPhysics(work: string, publicTar: string): string {
    const dir = join(work, "physics-pack");
    const source = resolve(root, "packages/shallot-physics");
    const unpacked = join(dir, "private-unpacked");
    const assembled = join(dir, "public");
    for (const path of [unpacked, assembled]) mkdirSync(path, { recursive: true });
    const inventory = Bun.spawnSync(["git", "ls-files", "--", "."], { cwd: canonical });
    assert.equal(inventory.exitCode, 0);
    const files = inventory.stdout
        .toString()
        .trim()
        .split("\n")
        .filter((file) => !/\.(?:test|fixture)\.ts$|\.gold\.json$/.test(file))
        .sort();
    assert(files.includes("index.ts") && files.includes("kernel.shared.wasm.ts"));
    const hashes = Object.fromEntries(files.map((file) => [file, hash(join(canonical, file))]));
    for (const file of files) assert(lstatSync(join(canonical, file)).isFile());
    exec(dir, "private-pack", ["bun", "pm", "pack", "--destination", dir], true, undefined, source);
    const privateTar = readdirSync(dir).find((file) => file.endsWith(".tgz"));
    assert(privateTar, "private Physics tarball produced");
    const privateMembers = archiveMembers(join(dir, privateTar));
    writeFileSync(join(dir, "private-members.json"), JSON.stringify(privateMembers, null, 2));
    checkMembers(privateMembers, [
        "package/package.json",
        ...files.map((file) => `package/${enginePath}/${file}`),
    ]);
    const originalMembers = archiveMembers(publicTar);
    writeFileSync(join(dir, "original-members.json"), JSON.stringify(originalMembers, null, 2));
    const publicFiles = originalMembers.filter(({ type }) => type === "-").map(({ path }) => path);
    checkMembers(originalMembers, publicFiles);
    exec(dir, "private-unpack", ["tar", "-xzf", join(dir, privateTar), "-C", unpacked]);
    assert.deepEqual(
        JSON.parse(readFileSync(join(unpacked, "package/package.json"), "utf8")),
        JSON.parse(readFileSync(join(source, "package.json"), "utf8")),
        "private solver manifest fields",
    );
    exec(dir, "public-unpack", ["tar", "-xzf", publicTar, "-C", assembled]);
    const target = join(assembled, "package", enginePath);
    assert.deepEqual(readdirSync(target).sort(), files, "public solver projection population");
    for (const file of files) {
        assert.equal(hash(join(unpacked, "package", enginePath, file)), hashes[file]);
        assert.equal(hash(join(target, file)), hashes[file]);
    }
    rmSync(target, { recursive: true });
    cpSync(join(unpacked, "package", enginePath), target, { recursive: true });
    const tar = join(dir, "shallot-composed.tgz");
    exec(dir, "public-compose", [
        "env",
        "COPYFILE_DISABLE=1",
        "tar",
        "-czf",
        tar,
        "-C",
        assembled,
        "package",
    ]);
    const composedMembers = archiveMembers(tar);
    writeFileSync(join(dir, "composed-members.json"), JSON.stringify(composedMembers, null, 2));
    checkMembers(composedMembers, publicFiles);
    console.log(
        `physics: archive members ${publicFiles.length} regular, ${composedMembers.length - publicFiles.length} directories; no extras`,
    );
    writeFileSync(
        join(dir, "projection.json"),
        JSON.stringify(
            {
                canonical,
                sourceManifest: hash(join(source, "package.json")),
                packedManifest: hash(join(unpacked, "package/package.json")),
                files: hashes,
                privateTar: hash(join(dir, privateTar)),
                publicTar: hash(tar),
            },
            null,
            2,
        ),
    );
    // Installed consumers have neither a private package dependency nor extracted generation roots.
    for (const path of [unpacked, assembled]) rmSync(path, { recursive: true });
    return tar;
}

const consumer = `
import assert from "node:assert/strict";
import { Physics, PhysicsPlugin, State } from "@dylanebert/shallot";
import { BodyType, World, init, threads, makeBoxHull } from "@dylanebert/shallot/physics/core";
import { kernel, workers } from "./node_modules/@dylanebert/shallot/src/standard/physics/kernel/kernel.ts";
const mode = process.argv[2];
const instantiate = WebAssembly.instantiate;
let instances = 0;
WebAssembly.instantiate = (...args) => { instances++; return instantiate(...args); };
await init(mode === "st" ? { threads: 0 } : { threads: 2 });
let before;
try { before = { kernel: kernel(), pool: workers() }; }
catch { throw Error("PHYSICS_KERNEL_IDENTITY"); }
assert.equal(threads(), mode === "st" ? 1 : 2, "SHALLOT_PHYSICS_THREADS");
assert.equal(before.kernel.memory.buffer instanceof SharedArrayBuffer, mode !== "st", "PHYSICS_MEMORY");
assert.equal(before.pool?.size ?? 0, mode === "st" ? 0 : 1, "PHYSICS_POOL");
const state = new State();
await PhysicsPlugin.warm(state);
assert(Physics.world instanceof World, "PHYSICS_SOLVER_IDENTITY");
assert.strictEqual(kernel(), before.kernel, "PHYSICS_KERNEL_IDENTITY");
assert.strictEqual(workers(), before.pool, "PHYSICS_POOL_IDENTITY");
assert.equal(instances, 1, "PHYSICS_POOL_IDENTITY");
const world = Physics.world;
const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
body.createHull({ density: 1 }, makeBoxHull(.5, .5, .5));
for (let step = 0; step < 20; step++) world.step(1 / 60, 4);
assert(body.getPosition().y < 5, "PHYSICS_STEP");
PhysicsPlugin.dispose(state);
state.dispose();
console.log("PHYSICS_REALIZED threads=" + threads());
`;

const types = `
import { Physics, PhysicsPlugin } from "@dylanebert/shallot";
import { World, Body, defaultWorldDef, defaultBodyDef, type WorldDef, type BodyDef, type InitOptions, init, type Vec3 } from "@dylanebert/shallot/physics/core";
const options: InitOptions = { threads: 2 };
const definition: WorldDef = defaultWorldDef();
const bodyDefinition: BodyDef = defaultBodyDef();
const raw: World = new World(definition);
const adapter: World | null = Physics.world;
const body: Body = raw.createBody(bodyDefinition);
const position: Vec3 = body.getPosition();
const hatch: Body | null = Physics.body(0);
const same: typeof Physics.world = raw;
void [options, adapter, position, hatch, same, init, PhysicsPlugin];
`;

/** Installed adapter/raw identity, actual lazy payload loss, declarations and natural Node/Bun exit. */
export function physicsArms(project: string): void {
    const shipped = join(project, "node_modules/@dylanebert/shallot");
    const engine = join(shipped, enginePath);
    assert(!lstatSync(shipped).isSymbolicLink(), "physical public package");
    assert(!existsSync(join(project, "node_modules/shallot-physics")), "no private install");
    writeFileSync(join(project, "physics-consumer.ts"), consumer);
    writeFileSync(join(project, "physics-types.ts"), types);
    writeFileSync(
        join(project, "physics-preload.ts"),
        'import { installGpuGlobals } from "./node_modules/@dylanebert/shallot/src/cli/gpu-globals.ts"; installGpuGlobals();\n',
    );
    const command = (mode: string) => [
        "bun",
        "--preload",
        "./physics-preload.ts",
        "physics-consumer.ts",
        mode,
    ];
    const typecheck = [
        "bun",
        resolve(root, "node_modules/typescript/bin/tsc"),
        "--noEmit",
        "--skipLibCheck",
        "--moduleResolution",
        "bundler",
        "--module",
        "esnext",
        "--target",
        "esnext",
        "--types",
        "@webgpu/types,vite/client,node",
        "physics-types.ts",
    ];
    const missing = (file: string, body: () => void) => {
        assert(!existsSync(`${file}.absent`));
        renameSync(file, `${file}.absent`);
        try {
            body();
        } finally {
            renameSync(`${file}.absent`, file);
        }
    };
    const mutate = (file: string, from: string, to: string, body: () => void) => {
        const original = readFileSync(file, "utf8");
        assert(original.includes(from), `mutation reaches ${file}`);
        writeFileSync(file, original.replaceAll(from, to));
        try {
            body();
        } finally {
            writeFileSync(file, original);
        }
    };
    missing(join(engine, "kernel/kernel.wasm.ts"), () =>
        exec(project, "physics-missing-st", command("st"), false, /kernel\.wasm/),
    );
    missing(join(engine, "kernel/kernel.shared.wasm.ts"), () => {
        exec(project, "physics-lazy-st", command("st"), true, /PHYSICS_REALIZED threads=1/);
        exec(project, "physics-missing-mt", command("mt"), false, /kernel\.shared\.wasm/);
    });
    const copy = join(shipped, "src/standard/physics-copy");
    assert(!existsSync(copy));
    cpSync(engine, copy, { recursive: true });
    try {
        mutate(
            join(shipped, "src/standard/physics/index.ts"),
            'from "./api"',
            'from "../physics-copy/api"',
            () =>
                exec(
                    project,
                    "physics-duplicate-solver",
                    command("mt"),
                    false,
                    /AssertionError: PHYSICS_SOLVER_IDENTITY/,
                ),
        );
        mutate(
            join(shipped, "src/standard/physics/index.ts"),
            'import { init, type Body as SolverBody, World as SolverWorld } from "./api";',
            'import { type Body as SolverBody, World as SolverWorld } from "./api";\nimport { init } from "../physics-copy/kernel/kernel";',
            () =>
                exec(
                    project,
                    "physics-duplicate-pool",
                    command("mt"),
                    false,
                    /AssertionError: PHYSICS_POOL_IDENTITY/,
                ),
        );
    } finally {
        rmSync(copy, { recursive: true });
    }
    missing(join(engine, "world/world.ts"), () =>
        exec(project, "physics-missing-types", typecheck, false, /world/),
    );
    exec(project, "physics-types", typecheck);
    const fixtures = join(shipped, "tests/physics/fixtures");
    const reader = join(engine, "solver/step.fixture.ts");
    assert(!existsSync(fixtures) && !existsSync(reader), "fixtures are not public package payload");
    const truth = join(root, "src/standard/physics/solver/fixtures");
    const population = readdirSync(truth)
        .filter((file) => file.endsWith(".json"))
        .sort();
    assert.equal(population.length, 53);
    mkdirSync(fixtures, { recursive: true });
    for (const file of population) cpSync(join(truth, file), join(fixtures, file));
    cpSync(join(canonical, "solver/step.fixture.ts"), reader);
    try {
        const fixtureCommand = (threads: string) => [
            "env",
            `SHALLOT_PHYSICS_THREADS=${threads}`,
            "bun",
            "test",
            "./node_modules/@dylanebert/shallot/src/standard/physics/solver/step.fixture.ts",
        ];
        missing(join(fixtures, population[0]), () =>
            exec(project, "physics-missing-fixture", fixtureCommand("0"), false, /ENOENT/),
        );
        for (const threads of ["0", "2", "8", "auto"])
            exec(project, `physics-fixtures-${threads}`, fixtureCommand(threads), true, /55 pass/);
        for (const file of population)
            assert.equal(hash(join(fixtures, file)), hash(join(truth, file)));
    } finally {
        rmSync(reader);
        for (const file of population) rmSync(join(fixtures, file));
    }
    exec(project, "physics-st", command("st"), true, /PHYSICS_REALIZED threads=1/);
    exec(project, "physics-mt", command("mt"), true, /PHYSICS_REALIZED threads=2/);

    // Node's supported application path is a bundle, not raw TypeScript under node_modules.
    writeFileSync(
        join(project, "physics-exit.ts"),
        `
import { init, threads, World, BodyType, makeBoxHull } from "@dylanebert/shallot/physics/core";
await init();
const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
body.createHull({ density: 1 }, makeBoxHull(.5, .5, .5));
for (let i = 0; i < 20; i++) world.step(1 / 60, 4);
if (!(body.getPosition().y < 5) || threads() < 2) throw Error("PHYSICS_EXIT_NOT_THREADED");
console.log("PHYSICS_EXIT threads=" + threads());
`,
    );
    exec(project, "physics-exit-build", [
        "bun",
        "build",
        "physics-exit.ts",
        "--target",
        "node",
        "--splitting",
        "--outdir",
        "physics-exit",
    ]);
    for (const runtime of ["node", "bun"]) {
        const result = Bun.spawnSync([runtime, "physics-exit/physics-exit.js"], {
            cwd: project,
            stdout: "pipe",
            stderr: "pipe",
            timeout: 15000,
        });
        const out = result.stdout.toString() + result.stderr.toString();
        writeFileSync(join(project, `physics-exit-${runtime}.log`), out);
        writeFileSync(
            join(project, `physics-exit-${runtime}.json`),
            JSON.stringify({
                cmd: [runtime, "physics-exit/physics-exit.js"],
                cwd: project,
                exit: result.exitCode,
                signal: result.signalCode,
                timeout: 15000,
            }),
        );
        assert.equal(result.exitCode, 0, out);
        assert(!result.signalCode, `natural ${runtime} exit`);
        assert.match(out, /PHYSICS_EXIT threads=[2-8]/);
        console.log(`physics: natural ${runtime} exit=0 ${out.trim()}`);
    }
}
