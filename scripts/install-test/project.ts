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
import { gunzipSync } from "node:zlib";
import { inspectRuntime } from "./runtime";

const root = resolve(import.meta.dir, "../..");
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const json = (path: string, value: unknown) => writeFileSync(path, JSON.stringify(value, null, 2));
const quote = JSON.stringify;

function execute(evidence: string, name: string, argv: string[], cwd: string) {
    mkdirSync(evidence, { recursive: true });
    const result = Bun.spawnSync(argv, { cwd, stdout: "pipe", stderr: "pipe" });
    const out = result.stdout.toString() + result.stderr.toString();
    writeFileSync(join(evidence, `${name}.log`), out);
    json(join(evidence, `${name}.json`), { argv, cwd, exit: result.exitCode });
    assert.equal(result.exitCode, 0, `${name}: ${out}`);
    return out;
}

function packageAt(dir: string, name: string, code: string, exports?: object) {
    const path = join(dir, "node_modules", name);
    mkdirSync(path, { recursive: true });
    json(join(path, "package.json"), {
        name,
        version: "1.0.0",
        type: "module",
        main: "index.js",
        ...(exports ? { exports } : {}),
    });
    writeFileSync(join(path, "index.js"), code);
    return path;
}

/** Exercise project-root identity and all-entry preflight in fresh Bun processes, without a device. */
function projectArms(
    command: string,
    features: string,
    parent: string,
    evidence: string,
    identity = false,
) {
    const external = mkdtempSync(join(tmpdir(), "shallot-project-external-"));
    const nested = mkdtempSync(join(parent, "game-"));
    const name = "shallot-project-binding-fixture";
    const decoyRoot = resolve(dirname(features), "..");
    const decoy = join(decoyRoot, "node_modules", name);
    assert(!existsSync(decoy), "owned decoy destination must be absent");
    const decoySentinel = join(nested, "decoy-evaluated");
    try {
        for (const [layout, project] of [
            ["external", external],
            ["nested", nested],
        ]) {
            if (layout === "nested")
                packageAt(
                    decoyRoot,
                    name,
                    `import { writeFileSync } from 'node:fs'; writeFileSync(${quote(decoySentinel)}, 'yes'); export default { name: 'Decoy', features: ['decoy-only'] };`,
                );
            const disabled = join(project, "disabled");
            const code = `export default { name: 'Project', features: ['timestamp-query'], preferredFeatures: ['subgroups'] };`;
            packageAt(project, name, code);
            packageAt(
                project,
                "@fixture/scoped",
                `export default { name: 'Scoped', features: ['shader-f16'] };`,
                { "./entry": "./index.js" },
            );
            packageAt(
                project,
                "disabled-binding",
                `import { writeFileSync } from 'node:fs'; writeFileSync(${quote(disabled)}, 'yes'); export default { name: 'Disabled' };`,
            );
            writeFileSync(join(project, "relative.js"), `export default { name: 'Relative' };`);
            writeFileSync(
                join(project, "disabled.js"),
                `import { writeFileSync } from 'node:fs'; writeFileSync(${quote(disabled)}, 'yes'); export default { name: 'Disabled' };`,
            );
            const manifest = {
                plugins: {
                    Bare: name,
                    Scoped: "@fixture/scoped/entry",
                    Relative: "./relative.js",
                    Off: ["disabled-binding", false],
                    OffFile: ["./disabled.js", false],
                    Absent: ["uninstalled-disabled-binding", false],
                },
            };
            json(join(project, "shallot.json"), manifest);
            const prelude = `import assert from 'node:assert/strict'; import { existsSync, realpathSync } from 'node:fs';
                const dir = ${quote(project)};
                const { planProject, readProject, loadLocalPlugins } = await import(${quote(command)});
                const { requiredFeatures } = await import(${quote(features)});`;
            execute(
                evidence,
                `${layout}-identity`,
                [
                    process.execPath,
                    "-e",
                    `${prelude}
                const planned = planProject(dir); assert.equal(planned.code, 0);
                assert.deepEqual(planned.plan.locals.map(l => l.path), [${quote(name)}, '@fixture/scoped/entry', dir + '/relative.js']);
                let loaded; try { loaded = await loadLocalPlugins(planned.plan); } catch { loaded = []; }
                assert.equal(loaded.length, 3, 'all project entries load');
                for (const [i, spec] of [${quote(name)}, '@fixture/scoped/entry', './relative.js'].entries())
                    assert.strictEqual(loaded[i], (await import(Bun.resolveSync(spec, dir))).default, 'project object identity');
                assert.equal(existsSync(${quote(disabled)}), false, 'disabled modules untouched');
                assert.equal(existsSync(${quote(decoySentinel)}), false, 'CLI decoy untouched');
                console.log('PASS project object identity, authored paths, disabled/decoy non-evaluation');
            `,
                ],
                parent,
            );
            execute(
                evidence,
                `${layout}-features`,
                [
                    process.execPath,
                    "-e",
                    `${prelude}
                assert.deepEqual(await requiredFeatures(dir), ['timestamp-query', 'shader-f16'], 'project required union excludes preferred and decoy');
                assert.equal(existsSync(${quote(disabled)}), false);
                assert.equal(existsSync(${quote(decoySentinel)}), false);
                console.log('PASS project required union and non-evaluation');
            `,
                ],
                parent,
            );
            for (const consumer of ["command", "features"]) {
                const sentinel = join(project, `${consumer}-early`);
                writeFileSync(
                    join(project, "early.js"),
                    `import { writeFileSync } from 'node:fs'; writeFileSync(${quote(sentinel)}, 'yes'); export default { name: 'Early' };`,
                );
                json(join(project, "shallot.json"), {
                    plugins: { Early: "./early.js", Gone: "missing-enabled-binding" },
                });
                execute(
                    evidence,
                    `${layout}-${consumer}-missing`,
                    [
                        process.execPath,
                        "-e",
                        `${prelude}
                    const setup = planProject(dir); assert.equal(setup.code, 2);
                    let error; try { await ${consumer === "command" ? "loadLocalPlugins(readProject(dir))" : "requiredFeatures(dir)"}; } catch (e) { error = e; }
                    assert.equal(existsSync(${quote(sentinel)}), false, 'all entries resolve before earlier effects');
                    assert.match(String(error), /plugin "Gone".*missing-enabled-binding/);
                    assert.ok(String(error).includes(dir));
                    console.log('PASS setup exit 2, direct loader preflight, diagnostic identity');
                `,
                    ],
                    parent,
                );
            }
            if (layout === "nested") {
                execute(
                    evidence,
                    "decoy-foil",
                    [
                        process.execPath,
                        "-e",
                        `${prelude}
                    const wrong = (await import(Bun.resolveSync(${quote(name)}, ${quote(decoyRoot)}))).default;
                    const right = (await import(Bun.resolveSync(${quote(name)}, dir))).default;
                    assert.throws(() => assert.strictEqual(wrong, right), /same|equal|reference/i);
                    assert.equal(existsSync(${quote(decoySentinel)}), true);
                    console.log('PASS deliberately CLI-root import fails project identity');
                `,
                    ],
                    parent,
                );
                if (identity) {
                    const plugin = join(project, "node_modules", name, "index.js");
                    writeFileSync(
                        plugin,
                        `import * as engine from '@dylanebert/shallot'; import tgpu from 'typegpu'; export { engine, tgpu }; export default { name: 'Identity' };`,
                    );
                    execute(
                        evidence,
                        "engine-typegpu-identity",
                        [
                            process.execPath,
                            "-e",
                            `${prelude}
                        const { installGpuGlobals } = await import(${quote(join(dirname(features), "gpu-globals.ts"))}); installGpuGlobals();
                        const origin = ${quote(dirname(plugin))};
                        const bindings = {};
                        for (const spec of ['@dylanebert/shallot', 'typegpu']) {
                            const cli = realpathSync(Bun.resolveSync(spec, ${quote(decoyRoot)}));
                            const child = realpathSync(Bun.resolveSync(spec, origin));
                            assert.equal(cli, child, 'one dependency realpath'); bindings[spec] = { cli, child };
                        }
                        const plugin = await import(${quote(plugin)});
                        assert.strictEqual(plugin.engine, await import(bindings['@dylanebert/shallot'].cli));
                        assert.strictEqual(plugin.tgpu, (await import(bindings.typegpu.cli)).default);
                        console.log(JSON.stringify(bindings));
                    `,
                        ],
                        parent,
                    );
                }
            }
        }
        execute(
            evidence,
            "purity",
            [
                process.execPath,
                "-e",
                `import assert from 'node:assert/strict'; await import(${quote(command)});
            assert.equal(Object.keys(require.cache).some(p => ['vite', 'rollup', 'esbuild'].some(n => p.includes('/' + n + '/'))), false);
            for (const key of ['window','document','GPUShaderStage','GPUBufferUsage']) assert.equal(key in globalThis, false);
            console.log('PASS pure command import');`,
            ],
            parent,
        );
    } finally {
        rmSync(decoy, { recursive: true, force: true });
    }
}

/** Build and evaluate the actual virtual project with browser exports; Bun must choose its own entry. */
function browserArm(command: string, viteEntry: string, parent: string, evidence: string) {
    const project = mkdtempSync(join(parent, "conditions-"));
    const pkg = packageAt(project, "conditional-binding", `export default { name: 'BunEntry' };`, {
        ".": { browser: "./browser.js", bun: "./index.js", default: "./default.js" },
    });
    writeFileSync(join(pkg, "browser.js"), `export default { name: 'BrowserEntry' };`);
    writeFileSync(join(pkg, "default.js"), `export default { name: 'WrongDefault' };`);
    const script = `import assert from 'node:assert/strict';
        import { writeFileSync } from 'node:fs';
        const { readProject, loadLocalPlugins } = await import(${quote(command)});
        const { projectPlugin } = await import(${quote(viteEntry)});
        const { build } = await import(Bun.resolveSync('vite', ${quote(dirname(viteEntry))}));
        const dir = ${quote(project)};
        const defaults = readProject(dir).engine;
        writeFileSync(dir + '/shallot.json', JSON.stringify({ plugins: { ...Object.fromEntries(defaults.map(n => [n, false])), Conditional: 'conditional-binding' } }));
        assert.equal((await loadLocalPlugins(readProject(dir)))[0].name, 'BunEntry');
        const result = await build({ configFile: false, root: dir, logLevel: 'silent', plugins: [projectPlugin(dir)],
            build: { write: false, minify: false, lib: { entry: dir + '/entry.js', formats: ['es'] } } });
        const outputs = [result].flat().flatMap(r => r.output);
        assert.equal(outputs.filter(o => o.type === 'chunk').length, 1);
        const code = outputs.find(o => o.type === 'chunk').code;
        writeFileSync(dir + '/built.mjs', code);
        console.log('PASS Bun marker and one browser output chunk');`;
    writeFileSync(
        join(project, "entry.js"),
        `import project from 'virtual:project'; export default project.locals[0].plugin.name;`,
    );
    execute(evidence, "browser-conditions", [process.execPath, "-e", script], parent);
    execute(
        evidence,
        "browser-evaluate",
        [
            process.execPath,
            "-e",
            `
        import assert from 'node:assert/strict';
        import marker from ${quote(join(project, "built.mjs"))};
        assert.equal(marker, 'BrowserEntry', 'Vite project browser condition');
        console.log('PASS evaluated browser marker differs from Bun marker');
    `,
        ],
        parent,
    );
}

/** Qualify one physical candidate installation, not the display/install roster. */
export function projectFlow(tarball: string, evidence: string) {
    assert(existsSync(tarball), "candidate tarball required");
    assert(!existsSync(join(evidence, "binding.json")), "fresh evidence destination required");
    mkdirSync(evidence, { recursive: true });
    const bytes = readFileSync(tarball);
    const tar = gunzipSync(bytes);
    const archive: Record<string, string> = {};
    let directories = 0;
    let pax = 0;
    for (let offset = 0; offset + 512 <= tar.length; ) {
        const header = tar.subarray(offset, offset + 512);
        if (header.every((b) => b === 0)) break;
        const text = (start: number, length: number) =>
            header
                .subarray(start, start + length)
                .toString()
                .replace(/\0.*$/s, "");
        const size = Number.parseInt(text(124, 12).trim(), 8) || 0;
        const type = text(156, 1);
        const name = [text(345, 155), text(0, 100)].filter(Boolean).join("/");
        const content = tar.subarray(offset + 512, offset + 512 + size);
        if (type === "x" || type === "g") {
            pax++;
            assert(
                !/\d+ (?:path|linkpath)=/.test(content.toString()),
                "unsupported PAX path override",
            );
        } else if (type === "5") directories++;
        else {
            assert(type === "0" || type === "", `unexpected archive type ${type}`);
            assert(
                name.startsWith("package/") &&
                    !name.split("/").some((p) => p === ".." || p.startsWith("._")),
            );
            const file = name.slice(8);
            assert(!Object.hasOwn(archive, file), `duplicate ${file}`);
            archive[file] = hash(content);
        }
        offset += 512 + Math.ceil(size / 512) * 512;
    }
    assert(Object.keys(archive).length > 250, "nonempty physical archive population");
    const fixture = join(evidence, "fixture.json");
    const prior = existsSync(fixture) ? JSON.parse(readFileSync(fixture, "utf8")) : null;
    if (prior)
        assert.equal(prior.sha256, hash(bytes), "retained install must bind the same tarball");
    const parent: string = prior?.parent ?? mkdtempSync(join(tmpdir(), "shallot-project-pack-"));
    if (!prior) {
        json(join(parent, "package.json"), {
            private: true,
            type: "module",
            dependencies: { typegpu: "~0.12.4", vite: "^8.0.16" },
        });
        execute(evidence, "peers-install", [process.execPath, "install"], parent);
        execute(evidence, "candidate-install", [process.execPath, "add", resolve(tarball)], parent);
        json(fixture, { parent, sha256: hash(bytes) });
    }
    const bin = realpathSync(join(parent, "node_modules/.bin/shallot"));
    const shipped = resolve(dirname(bin), "..");
    const installed: Record<string, string> = {};
    for (const entry of readdirSync(shipped, { recursive: true, withFileTypes: true })) {
        const path = join(entry.parentPath, entry.name);
        assert(!lstatSync(path).isSymbolicLink(), `package symlink ${path}`);
        if (entry.isFile()) installed[relative(shipped, path)] = hash(readFileSync(path));
    }
    assert.deepEqual(installed, archive, "archive/install exact file hashes");
    inspectRuntime(shipped);
    const runtime = JSON.parse(readFileSync(join(shipped, "runtime-inputs.json"), "utf8"));
    const cli = JSON.parse(readFileSync(join(shipped, "dist/cli-inputs.json"), "utf8"));
    for (const [file, expected] of Object.entries(cli.inputs))
        assert.equal(
            hash(readFileSync(resolve(root, "packages/shallot", file))),
            expected,
            `canonical CLI input ${file}`,
        );
    for (const [file, expected] of Object.entries(cli.outputs))
        assert.equal(installed[file], expected, `installed CLI output ${file}`);
    const command = join(shipped, "src/project/command.ts");
    projectArms(command, join(shipped, "bin/features.ts"), parent, evidence, true);
    browserArm(command, Bun.resolveSync("@dylanebert/shallot/vite", parent), parent, evidence);
    const restored = Object.fromEntries(
        readdirSync(shipped, { recursive: true, withFileTypes: true })
            .filter((entry) => !entry.isDirectory())
            .map((entry) => {
                const file = join(entry.parentPath, entry.name);
                assert(
                    entry.isFile() && !lstatSync(file).isSymbolicLink(),
                    `post-control physical file ${file}`,
                );
                return [relative(shipped, file), hash(readFileSync(file))];
            }),
    );
    assert.deepEqual(restored, archive, "post-control archive closure");
    const git = (args: string[]) =>
        execute(
            evidence,
            `git-${args.join("-").replaceAll("/", "_")}`,
            ["git", ...args],
            root,
        ).trim();
    json(join(evidence, "binding.json"), {
        tarball,
        driver: { path: import.meta.path, sha256: hash(readFileSync(import.meta.path)) },
        sha256: hash(bytes),
        parent,
        shipped,
        bin,
        commit: git(["rev-parse", "HEAD"]),
        tree: git(["rev-parse", "HEAD^{tree}"]),
        status: git(["status", "--short"]),
        regular: Object.keys(archive).length,
        directories,
        pax,
        archive,
        installed,
        runtime,
        cli,
        arms: 13,
    });
    console.log(
        `project seam: 13 subprocess arms passed; ${Object.keys(archive).length} regular files, ${directories} directories, ${pax} PAX headers; ${hash(bytes)}`,
    );
}

if (import.meta.main) {
    const args = process.argv.slice(2);
    assert.equal(args.length, 4, "usage: --tarball <path> --evidence <dir>");
    assert.equal(args[0], "--tarball");
    assert.equal(args[2], "--evidence");
    projectFlow(resolve(args[1]), resolve(args[3]));
}
