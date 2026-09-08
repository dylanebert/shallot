import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { cpSync, mkdirSync, readdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

export const SUBPATH_MIGRATIONS: Record<string, string | null> = {
    "./document": null,
    "./tween/core": "./animation/core",
};
const RETIRED_VALUES = [
    "Document",
    "Readback",
    "ReadbackSystem",
    "Sequence",
    "Session",
    "Tween",
    "TweenPlugin",
    "TweenState",
    "sequence",
    "tween",
];
const root = resolve(import.meta.dir, "../..");
const hash = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");
const files = (dir: string) =>
    readdirSync(dir, { recursive: true, withFileTypes: true })
        .filter((e) => e.isFile())
        .map((e) => join(e.parentPath, e.name));

/** Execute the immutable previous scaffold and plugin inputs against both physical installations. */
export function compatibilityFlow(work: string, candidate: string): void {
    const evidence = join(work, "compatibility");
    mkdirSync(evidence, { recursive: true });
    let sequence = 0;
    const exec = (name: string, cmd: string[], cwd: string, pass: boolean | null = true) => {
        const p = Bun.spawnSync(cmd, { cwd, stdout: "pipe", stderr: "pipe" });
        const text = p.stdout.toString() + p.stderr.toString();
        const stem = join(evidence, `${++sequence}-${name}`);
        writeFileSync(`${stem}.log`, text);
        writeFileSync(
            `${stem}.json`,
            JSON.stringify({
                cmd,
                cwd,
                exit: p.exitCode,
                signal: p.signalCode,
                runtime: Bun.version,
            }),
        );
        if (pass !== null) assert.equal(p.exitCode === 0, pass, `${name}: ${text.slice(-2000)}`);
        else
            assert(
                p.exitCode === 0 || (!p.signalCode && text.includes("error TS")),
                `${name}: no attributable compiler result`,
            );
        console.log(`compatibility: ${name} exit=${p.exitCode}`);
        return text;
    };
    const baseline = join(root, "scripts/install-test/compat-0.9.5");
    const previous = join(baseline, "tarballs/shallot-0.9.5.tgz");
    const scaffold = join(baseline, "tarballs/create-shallot-0.9.5.tgz");
    const original = join(evidence, "original");
    mkdirSync(original);
    exec(
        "extract-previous",
        ["tar", "-xzf", previous, "--strip-components=1", "-C", original],
        evidence,
    );
    const create = join(evidence, "create");
    mkdirSync(create);
    writeFileSync(
        join(create, "package.json"),
        JSON.stringify({ private: true, dependencies: { "create-shallot": `file:${scaffold}` } }),
    );
    exec("install-previous-scaffold", ["bun", "install"], create);
    assert.equal(
        realpathSync(join(create, "node_modules/create-shallot")),
        join(create, "node_modules/create-shallot"),
    );
    exec(
        "execute-previous-scaffold",
        ["bun", "node_modules/.bin/create-shallot", "compat-app"],
        create,
    );
    const emitted = join(create, "compat-app");
    const frozen = join(baseline, "scaffold");
    const inputs = Object.fromEntries(
        files(frozen).map((file) => [file.slice(frozen.length + 1), hash(file)]),
    );
    assert.deepEqual(
        files(emitted)
            .map((file) => file.slice(emitted.length + 1))
            .sort(),
        Object.keys(inputs).sort(),
        "actual previous scaffold population",
    );
    for (const [file, expected] of Object.entries(inputs))
        assert.equal(hash(join(emitted, file)), expected, `unchanged previous scaffold ${file}`);
    writeFileSync(join(evidence, "frozen-inputs.json"), JSON.stringify(inputs, null, 2));
    const migration = readFileSync(join(original, "MIGRATION.md"), "utf8");
    const config = /```ts\n(\/\/ vite\.config\.ts\n[\s\S]*?)\n```/.exec(migration)?.[1];
    assert(config, "previous documented ejected configuration");
    writeFileSync(join(evidence, "previous-vite.config.ts.txt"), `${config}\n`);
    let baselineDiagnostics: string[] | undefined;
    let baselineValues: string[] | undefined;
    let baselineTiming: unknown;
    let baselineClip: unknown;
    for (const [label, tar] of [
        ["previous", previous],
        ["candidate", candidate],
    ] as const) {
        const app = join(evidence, `${label}-scaffold`);
        cpSync(emitted, app, { recursive: true });
        const manifest = JSON.parse(readFileSync(join(app, "package.json"), "utf8"));
        manifest.dependencies["@dylanebert/shallot"] = `file:${tar}`;
        // Only artifact selection and documented optional verification/native prerequisites differ.
        manifest.devDependencies.playwright = "1.62.1";
        manifest.devDependencies["bun-webgpu"] = "0.1.7";
        writeFileSync(join(app, "package.json"), JSON.stringify(manifest, null, 2));
        for (const [file, expected] of Object.entries(inputs))
            if (file !== "package.json")
                assert.equal(
                    hash(join(app, file)),
                    expected,
                    `${label}: original source/config ${file}`,
                );
        exec(`${label}-install`, ["bun", "install"], app);
        const installed = join(app, "node_modules/@dylanebert/shallot");
        assert.equal(realpathSync(installed), installed);
        writeFileSync(
            join(evidence, `${label}-artifact.json`),
            JSON.stringify({
                tar,
                sha256: hash(tar),
                manifest: JSON.parse(readFileSync(join(installed, "package.json"), "utf8")),
            }),
        );
        const originalTypes = exec(
            `${label}-original-types`,
            ["bun", "node_modules/typescript/bin/tsc", "--noEmit"],
            app,
            label === "previous" ? false : null,
        );
        const diagnostics = originalTypes
            .trim()
            .split("\n")
            .filter(Boolean)
            .map((line) => line.replace(/\(\d+,\d+\)/, ""));
        if (label === "previous") {
            assert.equal(diagnostics.length, 6, "frozen scaffold type-context failure population");
            assert(
                diagnostics.every((line) => /error TS(?:2339|2591|7006):/.test(line)),
                "only the pinned original type-context failures",
            );
            baselineDiagnostics = diagnostics;
        } else {
            const remaining = [...baselineDiagnostics!];
            for (const diagnostic of diagnostics) {
                const index = remaining.indexOf(diagnostic);
                assert(index >= 0, `candidate introduced a compiler diagnostic: ${diagnostic}`);
                remaining.splice(index, 1);
            }
        }
        writeFileSync(
            join(app, "surface.mjs"),
            'import * as api from "@dylanebert/shallot"; console.log(JSON.stringify(Object.keys(api).sort()));',
        );
        const values = JSON.parse(
            exec(`${label}-public-values`, ["bun", "surface.mjs"], app),
        ) as string[];
        if (label === "previous") baselineValues = values;
        else
            assert.deepEqual(
                baselineValues!.filter((name) => !values.includes(name)),
                RETIRED_VALUES,
                "all removed author values have an explicit migration or retirement",
            );
        writeFileSync(
            join(app, "document-check.mjs"),
            'import {Document} from "@dylanebert/shallot/document"; const doc=new Document("<scene><a transform /></scene>"); if(!doc.serialize().includes("transform"))throw Error("document roundtrip"); console.log("DOCUMENT_OK");',
        );
        const document = exec(
            `${label}-document-contract`,
            ["bun", "document-check.mjs"],
            app,
            label === "previous",
        );
        if (label === "previous") assert.match(document, /DOCUMENT_OK/);
        else assert.match(document, /document|exports/);
        writeFileSync(
            join(app, "timing.mjs"),
            `import {Composite,Fill,owns,sample,EASING_FUNCTIONS,getEasingName,getEasingIndex} from "@dylanebert/shallot/${label === "previous" ? "tween" : "animation"}/core"; const rows=[];for(const duration of [0,1])for(const elapsed of [-1,0,0.5,1,2])for(const fill of Object.values(Fill))for(const composite of Object.values(Composite))for(let easing=0;easing<EASING_FUNCTIONS.length;easing++)rows.push([duration,elapsed,fill,composite,getEasingName(easing),getEasingIndex(getEasingName(easing)),owns(elapsed,duration,fill),sample(elapsed,duration,easing,0,10,composite,3)]);console.log(JSON.stringify(rows));`,
        );
        const timing = JSON.parse(exec(`${label}-timing-migration`, ["bun", "timing.mjs"], app));
        assert.equal(timing.length, 2480);
        if (label === "previous") baselineTiming = timing;
        else
            assert.deepEqual(
                timing,
                baselineTiming,
                "timing/easing migration preserves the full sampled parameter population",
            );
        writeFileSync(
            join(app, "clip.mjs"),
            label === "previous"
                ? 'import {Fill,owns,sample} from "@dylanebert/shallot/tween/core";console.log(JSON.stringify([-1,0,0.5,1,2].map(t=>owns(t,1,Fill.Forwards)?sample(t,1,0,0,10,0,0):null)));'
                : 'import {keyframes,Pose} from "@dylanebert/shallot/animation/core";const clip=keyframes({x:[{value:0},{value:10}]},{duration:1});const pose=new Pose();console.log(JSON.stringify([-1,0,0.5,1,2].map(t=>{clip.evaluate(t,pose);return pose.get("x")??null;})));',
        );
        const clip = JSON.parse(exec(`${label}-clip-migration`, ["bun", "clip.mjs"], app));
        if (label === "previous") baselineClip = clip;
        else assert.deepEqual(clip, baselineClip);
        exec(`${label}-type-context-install`, ["bun", "add", "-d", "@types/node@^26.0.0"], app);
        exec(
            `${label}-migrated-types`,
            [
                "bun",
                "node_modules/typescript/bin/tsc",
                "--noEmit",
                "--types",
                "@webgpu/types,node,vite/client",
            ],
            app,
        );
        exec(`${label}-build`, ["bun", "node_modules/.bin/shallot", "build", "."], app);
        const verdict = exec(
            `${label}-built-verify`,
            [
                "bun",
                "node_modules/.bin/shallot",
                "verify",
                ".",
                "--dist",
                "--json",
                "--timeout",
                "30000",
            ],
            app,
        );
        assert(/"pass"\s*:\s*true/.test(verdict));
        exec(`${label}-help`, ["bun", "node_modules/.bin/shallot", "tui", "--help"], app);
        const tui = exec(
            `${label}-external-root-tui`,
            [
                "bun",
                join(app, "node_modules/.bin/shallot"),
                "tui",
                app,
                "--frames",
                "1",
                "--tier",
                "plain",
            ],
            evidence,
            false,
        );
        if (label === "previous")
            assert.match(tui, /unknown option: --frames/, "0.9.5 predates the bounded TUI command");
        else {
            assert.match(
                tui,
                /does not enable "Cells"/,
                "ordinary scaffold retains the explicit terminal configuration refusal",
            );
            const terminal = join(app, "terminal");
            mkdirSync(join(terminal, "public"), { recursive: true });
            cpSync(join(app, "src"), join(terminal, "src"), { recursive: true });
            writeFileSync(
                join(terminal, "shallot.json"),
                JSON.stringify({ scene: "main.scene", plugins: { Cells: true } }),
            );
            writeFileSync(
                join(terminal, "public/main.scene"),
                '<scene><a camera sear cells transform="pos: 0 0 5" /><a part transform /></scene>',
            );
            exec(
                `${label}-configured-external-tui`,
                [
                    "bun",
                    join(app, "node_modules/.bin/shallot"),
                    "tui",
                    terminal,
                    "--frames",
                    "1",
                    "--tier",
                    "plain",
                ],
                evidence,
            );
        }
        writeFileSync(join(app, "vite.config.ts"), `${config}\n`);
        writeFileSync(
            join(app, "index.html"),
            '<!doctype html><html><body style="margin:0"><canvas style="display:block;width:100vw;height:100vh"></canvas><script type="module" src="/src/ejected.ts"></script></body></html>',
        );
        writeFileSync(
            join(app, "src/ejected.ts"),
            'import {run} from "@dylanebert/shallot";import project from "virtual:project";await run({plugins:project.plugins,scene:project.scene??undefined,defaults:false});\n',
        );
        exec(
            `${label}-documented-ejected-build`,
            ["bun", "node_modules/.bin/shallot", "build", "."],
            app,
        );
        const ejected = exec(
            `${label}-documented-ejected-verify`,
            [
                "bun",
                "node_modules/.bin/shallot",
                "verify",
                ".",
                "--dist",
                "--json",
                "--timeout",
                "30000",
            ],
            app,
        );
        assert(/"pass"\s*:\s*true/.test(ejected));
        const recipe = join(evidence, `${label}-original-particles`);
        cpSync(join(original, "examples/recipes/gpu-particles"), recipe, { recursive: true });
        const recipePkg = JSON.parse(readFileSync(join(recipe, "package.json"), "utf8"));
        recipePkg.dependencies = {
            ...recipePkg.dependencies,
            "@dylanebert/shallot": `file:${tar}`,
            typegpu: "~0.12.4",
        };
        recipePkg.devDependencies = { ...recipePkg.devDependencies, playwright: "1.62.1" };
        writeFileSync(join(recipe, "package.json"), JSON.stringify(recipePkg, null, 2));
        for (const file of files(join(original, "examples/recipes/gpu-particles"))) {
            const rel = file.slice(join(original, "examples/recipes/gpu-particles").length + 1);
            if (rel !== "package.json")
                assert.equal(
                    hash(join(recipe, rel)),
                    hash(file),
                    `${label}: original plugin ${rel}`,
                );
        }
        exec(`${label}-original-plugin-install`, ["bun", "install"], recipe);
        exec(
            `${label}-original-plugin-build`,
            ["bun", "node_modules/.bin/shallot", "build", "."],
            recipe,
        );
        const plugin = exec(
            `${label}-original-plugin-verify`,
            [
                "bun",
                "node_modules/.bin/shallot",
                "verify",
                ".",
                "--dist",
                "--json",
                "--timeout",
                "30000",
            ],
            recipe,
        );
        assert(/"pass"\s*:\s*true/.test(plugin));
    }
}

if (import.meta.main) {
    const [work, tar] = process.argv.slice(2);
    assert(work && tar, "compatibility.ts <owned evidence directory> <candidate tarball>");
    compatibilityFlow(resolve(work), resolve(tar));
}
