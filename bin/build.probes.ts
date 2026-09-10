import { test } from "bun:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    existsSync,
    lstatSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    readFileSync,
    realpathSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { buildWeb } from "./build";

// G2's permanent command is:
//     bun test --timeout 120000 ./bin/build.probes.ts
// The source and installed arms both drive `bin/cli.ts build`'s buildWeb → viteBuild → projectPlugin
// `generateBundle` path; the capture plugin is ordered before `projectPlugin` to prove the unused
// tree-shaken new-URL asset existed before the pruning hook ran. This is deliberately a build probe,
// not a browser/native/default-suite gate.

const SHALL0T_ROOT = resolve(import.meta.dir, "..");
const ENGINE_PACKAGE = SHALL0T_ROOT;
const EVIDENCE = join(tmpdir(), "shallot-manifest-css-assets-build-probe");
const CHILD_BYTES = Buffer.alloc(5_001, 0x43);
const UNUSED_BYTES = Buffer.alloc(5_001, 0x55);
CHILD_BYTES.write("SHALLOT-CSS-CHILD-v1");
UNUSED_BYTES.write("SHALLOT-UNUSED-v1");

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const quote = JSON.stringify;

interface CapturedFile {
    type: string;
    fileName: string;
    bytes: number | null;
    sha256: string | null;
}

interface Capture {
    files: CapturedFile[];
}

interface BundleFacts {
    html: { bytes: number; sha256: string };
    css: { file: string; bytes: number; sha256: string };
    child: { file: string; bytes: number; sha256: string; prefix: string };
    distFiles: string[];
    distHashes: Record<string, string>;
    prunedUnused: boolean;
}

function writeJson(path: string, value: unknown): void {
    writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`);
}

function filesUnder(root: string): string[] {
    return readdirSync(root, { recursive: true, withFileTypes: true })
        .filter((entry) => entry.isFile())
        .map((entry) => join(entry.parentPath, entry.name));
}

function sourceFixture(parent: string) {
    const project = mkdtempSync(join(parent, "source-"));
    return createFixture(project);
}

function createFixture(project: string) {
    const src = join(project, "src");
    const dist = join(project, "dist");
    const plugin = join(src, "css-plugin.ts");
    const prePrune = join(project, "pre-prune.json");
    mkdirSync(src, { recursive: true });
    writeJson(join(project, "package.json"), { type: "module" });
    writeJson(join(project, "shallot.json"), {
        plugins: {
            Slab: false,
            Transforms: false,
            Input: false,
            Render: false,
            Part: false,
            Sear: false,
            Glaze: false,
            CssFixture: "./src/css-plugin.ts",
        },
    });
    writeFileSync(
        plugin,
        'import "./style.css";\nexport function unusedAsset() { return new URL("./unused.bin", import.meta.url); }\nexport default { name: "CssFixture" };\n',
    );
    writeFileSync(join(src, "style.css"), 'body{background-image:url("./child.bin")}\n');
    writeFileSync(join(src, "child.bin"), CHILD_BYTES);
    writeFileSync(join(src, "unused.bin"), UNUSED_BYTES);
    // This project plugin runs before Shallot's output hook. It records bytes/hashes, not only names, so
    // the post-build assertion proves a real unused asset was emitted and then removed.
    writeFileSync(
        join(project, "vite.config.ts"),
        `import { createHash } from "node:crypto";
import { writeFileSync } from "node:fs";
const receipt = ${quote(prePrune)};
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
export default { plugins: [{ name: "capture-css-assets-before-prune", enforce: "pre", generateBundle(_options, bundle) {
    writeFileSync(receipt, JSON.stringify({ files: Object.values(bundle).map((file) => {
        if (file.type !== "asset") return { type: file.type, fileName: file.fileName, bytes: null, sha256: null };
        const bytes = Buffer.from(file.source);
        return { type: file.type, fileName: file.fileName, bytes: bytes.byteLength, sha256: hash(bytes) };
    }) }, null, 2));
} }] };
`,
    );
    assert.equal(
        existsSync(join(project, "index.html")),
        false,
        "fixture must begin without source HTML",
    );
    return { project, src, plugin, prePrune, dist };
}

function run(name: string, argv: string[], cwd: string) {
    const result = Bun.spawnSync(argv, {
        cwd,
        stdout: "pipe",
        stderr: "pipe",
        timeout: 120_000,
    });
    const output = `${result.stdout.toString()}\n${result.stderr.toString()}`;
    writeFileSync(join(EVIDENCE, `${name}.log`), output);
    writeJson(join(EVIDENCE, `${name}.json`), {
        argv,
        cwd,
        exit: result.exitCode,
        signal: result.signalCode,
        runtime: Bun.version,
    });
    assert.equal(result.exitCode, 0, `${name} failed:\n${output}`);
    return output;
}

function outputPath(dist: string, reference: string): string {
    const withoutQuery = reference.split(/[?#]/, 1)[0];
    return resolve(dist, withoutQuery.replace(/^\.\//, ""));
}

function verifyBundle(fixture: ReturnType<typeof createFixture>): BundleFacts {
    assert(existsSync(fixture.prePrune), "the pre-prune output capture must run");
    const capture = JSON.parse(readFileSync(fixture.prePrune, "utf8")) as Capture;
    const unusedHash = sha256(UNUSED_BYTES);
    const unusedBeforePrune = capture.files.find(
        (file) =>
            file.type === "asset" &&
            file.bytes === UNUSED_BYTES.byteLength &&
            file.sha256 === unusedHash,
    );
    assert(
        unusedBeforePrune,
        "the tree-shaken new-URL asset must exist before generateBundle pruning",
    );

    const htmlPath = join(fixture.dist, "index.html");
    assert(existsSync(htmlPath), "build must emit synthesized HTML");
    const html = readFileSync(htmlPath, "utf8");
    const cssReferences = [...html.matchAll(/href=["']([^"']+\.css(?:\?[^"']*)?)["']/g)].map(
        (match) => match[1],
    );
    assert.equal(cssReferences.length, 1, "HTML must point to one emitted CSS asset");
    const cssFile = outputPath(fixture.dist, cssReferences[0]);
    assert(existsSync(cssFile), `HTML CSS reference must resolve: ${cssReferences[0]}`);
    const css = readFileSync(cssFile);
    assert(css.byteLength > 0, "emitted CSS must be nonempty");

    const childReference = /url\(["']?([^)"']+)["']?\)/.exec(css.toString())?.[1];
    assert(childReference, "emitted CSS must retain its child-asset reference");
    const childFile = outputPath(dirname(cssFile), childReference);
    assert(existsSync(childFile), `CSS child reference must resolve: ${childReference}`);
    const child = readFileSync(childFile);
    assert.deepEqual(child, CHILD_BYTES, "CSS child bytes/signature must survive unchanged");

    const distFiles = filesUnder(fixture.dist);
    const distHashes = Object.fromEntries(
        distFiles.map((file) => [relative(fixture.dist, file), sha256(readFileSync(file))]),
    );
    assert(
        !distFiles.some((file) => sha256(readFileSync(file)) === unusedHash),
        "the genuinely unused asset must be absent after generateBundle pruning",
    );
    assert(
        !distFiles.some((file) => readFileSync(file).includes(UNUSED_BYTES)),
        "the unused asset bytes must be absent after generateBundle pruning",
    );
    return {
        html: { bytes: Buffer.byteLength(html), sha256: sha256(Buffer.from(html)) },
        css: {
            file: relative(fixture.dist, cssFile),
            bytes: css.byteLength,
            sha256: sha256(css),
        },
        child: {
            file: relative(fixture.dist, childFile),
            bytes: child.byteLength,
            sha256: sha256(child),
            prefix: child.subarray(0, "SHALLOT-CSS-CHILD-v1".length).toString(),
        },
        distFiles: distFiles.map((file) => relative(fixture.dist, file)),
        distHashes,
        prunedUnused: true,
    };
}

function installedIdentity(consumer: string) {
    const installed = join(consumer, "node_modules/@dylanebert/shallot");
    const bin = join(consumer, "node_modules/.bin/shallot");
    assert(existsSync(installed), "packed engine must be installed");
    assert(existsSync(bin), "installed public shallot bin must exist");
    assert.equal(
        lstatSync(installed).isSymbolicLink(),
        false,
        "installed engine must not be a symlink",
    );
    const packageRealpath = realpathSync(installed);
    const binRealpath = realpathSync(bin);
    assert.equal(packageRealpath, installed, "installed package realpath must be physical");
    assert.equal(
        binRealpath,
        resolve(packageRealpath, "bin/cli.ts"),
        "public bin must target installed CLI",
    );
    return { packageRealpath, binRealpath };
}

test("manifest build retains imported CSS/child assets and prunes a tree-shaken new-URL asset in source and pack installs", async () => {
    rmSync(EVIDENCE, { recursive: true, force: true });
    mkdirSync(EVIDENCE, { recursive: true });
    const sourceParent = mkdtempSync(join(SHALL0T_ROOT, ".shallot-css-build-"));
    const consumer = realpathSync(mkdtempSync(join(tmpdir(), "shallot-css-consumer-")));
    let sourceFixtureValue: ReturnType<typeof createFixture> | undefined;
    let consumerFixture: ReturnType<typeof createFixture> | undefined;
    try {
        sourceFixtureValue = sourceFixture(sourceParent);
        await buildWeb(sourceFixtureValue.project);
        const sourceBundle = verifyBundle(sourceFixtureValue);

        const originalPlugin = readFileSync(sourceFixtureValue.plugin, "utf8");
        writeFileSync(
            sourceFixtureValue.plugin,
            'import "./this-local-plugin-does-not-exist.css"; export default { name: "CssFixture" };\n',
        );
        let invalidError: unknown;
        try {
            await buildWeb(sourceFixtureValue.project);
        } catch (error) {
            invalidError = error;
        } finally {
            writeFileSync(sourceFixtureValue.plugin, originalPlugin);
        }
        assert(invalidError, "an invalid local plugin must fail the build");
        assert.equal(
            existsSync(join(sourceFixtureValue.project, "index.html")),
            false,
            "failed build must remove synthesized source HTML",
        );

        consumerFixture = createFixture(join(consumer, "game"));
        assert.equal(
            existsSync(join(consumer, "node_modules")),
            false,
            "consumer starts absent-only",
        );
        writeJson(join(consumer, "package.json"), {
            name: "shallot-css-asset-consumer",
            private: true,
            type: "module",
            dependencies: {
                "@dylanebert/shallot": "PACK_TARBALL",
                typegpu: "~0.12.5",
            },
        });

        const packDir = join(EVIDENCE, "pack");
        mkdirSync(packDir, { recursive: true });
        const packOutput = run(
            "pack",
            ["bun", "pm", "pack", "--destination", packDir, "--quiet"],
            ENGINE_PACKAGE,
        );
        const tarballs = readdirSync(packDir).filter((file) => file.endsWith(".tgz"));
        assert.equal(tarballs.length, 1, `pack must produce exactly one tarball:\n${packOutput}`);
        const tarball = join(packDir, tarballs[0]);
        const packageJson = JSON.parse(readFileSync(join(consumer, "package.json"), "utf8")) as {
            dependencies: Record<string, string>;
        };
        packageJson.dependencies["@dylanebert/shallot"] = `file:${tarball}`;
        writeJson(join(consumer, "package.json"), packageJson);

        run("install", ["bun", "install", "--no-progress"], consumer);
        const lockPath = join(consumer, "bun.lock");
        assert(existsSync(lockPath), "first physical install must write bun.lock");
        const lockHash = sha256(readFileSync(lockPath));
        rmSync(join(consumer, "node_modules"), { recursive: true, force: true });
        run(
            "install-frozen-repeat",
            ["bun", "install", "--frozen-lockfile", "--no-progress"],
            consumer,
        );
        assert.equal(
            sha256(readFileSync(lockPath)),
            lockHash,
            "frozen repeat must retain lock bytes",
        );
        const identity = installedIdentity(consumer);

        const physicalBuildOutput = run(
            "physical-build",
            ["bunx", "shallot", "build", "."],
            consumerFixture.project,
        );
        const physicalBundle = verifyBundle(consumerFixture);
        assert.match(physicalBuildOutput, /done\./, "public bin build must complete");

        writeJson(join(EVIDENCE, "receipt.json"), {
            command: "bun test --timeout 120000 ./bin/build.probes.ts",
            runtime: Bun.version,
            source: {
                project: sourceFixtureValue.project,
                bundle: sourceBundle,
                failedLocalPluginRemovedIndex: true,
            },
            pack: {
                tarball,
                sha256: sha256(readFileSync(tarball)),
                output: packOutput,
            },
            consumer: {
                project: consumerFixture.project,
                lock: lockPath,
                lockSha256: lockHash,
                packageRealpath: identity.packageRealpath,
                binRealpath: identity.binRealpath,
                bundle: physicalBundle,
            },
        });
        console.log(
            `G2 source+pack passed: tarball ${sha256(readFileSync(tarball))}; lock ${lockHash}; ` +
                `bin ${identity.binRealpath}`,
        );
    } finally {
        rmSync(sourceParent, { recursive: true, force: true });
        rmSync(consumer, { recursive: true, force: true });
    }
});
