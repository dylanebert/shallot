import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
    existsSync,
    lstatSync,
    readdirSync,
    readFileSync,
    renameSync,
    writeFileSync,
} from "node:fs";
import { builtinModules } from "node:module";
import { dirname, relative, resolve } from "node:path";
import { parse } from "@babel/parser";
import { SUBPATH_MIGRATIONS } from "./compatibility";

const root = resolve(import.meta.dir, "../..");
const hash = (file: string) => createHash("sha256").update(readFileSync(file)).digest("hex");
const walk = (dir: string) =>
    readdirSync(dir, { recursive: true, withFileTypes: true })
        .filter((e) => e.isFile() || e.isSymbolicLink())
        .map((e) => resolve(e.parentPath, e.name));
const carried = (file: string) =>
    !/\.(?:test|fixture)\.ts$|\.gold\.json$|^src\/extras\/gltf\/fixtures(?:\/|$)/.test(file);

/** Resolve the actual installed graph before comparing projection hashes, so an escape cannot hide behind its record. */
export function inspectRuntime(shipped: string): void {
    const pkg = JSON.parse(readFileSync(resolve(shipped, "package.json"), "utf8"));
    assert(!pkg.exports["./src/*"], "installed runtime: source wildcard removed");
    const old = JSON.parse(
        readFileSync(
            resolve(root, "scripts/install-test/compat-0.9.5/engine-package.json"),
            "utf8",
        ),
    );
    for (const key of Object.keys(old.exports).filter((key) => key !== "./src/*")) {
        if (Object.hasOwn(SUBPATH_MIGRATIONS, key)) {
            const replacement = SUBPATH_MIGRATIONS[key];
            assert(
                !pkg.exports[key],
                `installed runtime: retired entry unexpectedly restored ${key}`,
            );
            if (replacement)
                assert(
                    pkg.exports[replacement],
                    `installed runtime: missing migration target ${replacement}`,
                );
        } else assert(pkg.exports[key], `installed runtime: missing public entry ${key}`);
    }
    const dependencies = new Set(
        Object.keys({ ...pkg.dependencies, ...pkg.peerDependencies, ...pkg.optionalDependencies }),
    );
    const resolveLocal = (file: string, spec: string) => {
        const path = resolve(dirname(file), spec.split("?")[0]);
        assert(
            path.startsWith(shipped + "/"),
            `installed runtime: outside-package reach ${relative(shipped, file)} -> ${spec}`,
        );
        const candidates = [
            path,
            `${path}.ts`,
            `${path}.js`,
            `${path}/index.ts`,
            ...(path.endsWith(".js")
                ? [path.replace(/\.js$/, ".ts"), path.replace(/\.js$/, ".d.ts")]
                : []),
        ];
        assert(
            candidates.some((p) => existsSync(p) && lstatSync(p).isFile()),
            `installed runtime: missing target ${relative(shipped, file)} -> ${spec}`,
        );
    };
    const files = walk(shipped).filter(
        (file) => !file.includes("/node_modules/", shipped.length + 1),
    );
    assert(files.length > 250, "installed runtime: complete physical package population");
    let imports = 0;
    for (const file of files) {
        assert(!lstatSync(file).isSymbolicLink(), `installed runtime: link ${file}`);
        if (!/\.(?:[cm]?js|ts)$/.test(file)) continue;
        const tree = parse(readFileSync(file, "utf8"), {
            sourceType: "unambiguous",
            plugins: file.endsWith(".ts") ? [["typescript", { dts: file.endsWith(".d.ts") }]] : [],
            sourceFilename: file,
        });
        const visit = (node: any) => {
            if (!node || typeof node !== "object") return;
            if (Array.isArray(node)) {
                node.forEach(visit);
                return;
            }
            let spec: unknown;
            if (
                [
                    "ImportDeclaration",
                    "ExportNamedDeclaration",
                    "ExportAllDeclaration",
                    "ImportExpression",
                ].includes(node.type)
            )
                spec = node.source?.value;
            if (node.type === "TSImportType") spec = node.argument?.value;
            if (
                node.type === "CallExpression" &&
                (node.callee?.type === "Import" ||
                    (node.callee?.type === "Identifier" && node.callee.name === "require"))
            )
                spec =
                    node.arguments[0]?.type === "StringLiteral"
                        ? node.arguments[0].value
                        : undefined;
            if (typeof spec === "string") {
                imports++;
                if (spec.startsWith(".")) resolveLocal(file, spec);
                else if (
                    spec === "@dylanebert/shallot" ||
                    spec.startsWith("@dylanebert/shallot/")
                ) {
                    const key =
                        spec === "@dylanebert/shallot"
                            ? "."
                            : `.${spec.slice("@dylanebert/shallot".length)}`;
                    const entry = pkg.exports[key];
                    assert(entry, `installed runtime: unresolved public specifier ${spec}`);
                    for (const target of typeof entry === "string" ? [entry] : Object.values(entry))
                        resolveLocal(resolve(shipped, "package.json"), String(target));
                } else if (
                    !spec.startsWith("node:") &&
                    spec !== "bun" &&
                    !builtinModules.includes(spec)
                ) {
                    const name = spec.startsWith("@")
                        ? spec.split("/").slice(0, 2).join("/")
                        : spec.split("/")[0];
                    assert(
                        dependencies.has(name),
                        `installed runtime: undeclared/private specifier ${spec}`,
                    );
                }
            }
            if (
                node.type === "NewExpression" &&
                node.callee?.name === "URL" &&
                node.arguments[0]?.type === "StringLiteral" &&
                node.arguments[1]?.type === "MemberExpression" &&
                node.arguments[1].object?.type === "MetaProperty"
            ) {
                const url = node.arguments[0].value;
                if (!/^[a-z]+:/i.test(url))
                    resolveLocal(file, url.startsWith(".") ? url : `./${url}`);
            }
            for (const [key, child] of Object.entries(node))
                if (!["loc", "start", "end", "comments"].includes(key)) visit(child);
        };
        visit(tree);
    }
    assert(imports > 300, "installed runtime: nonempty resolved import population");
    const owner = resolve(root);
    const solver = resolve(root, "packages/shallot-physics");
    const engine = "src/standard/physics/engine/";
    const canonical = (file: string) => resolve(file.startsWith(engine) ? solver : owner, file);
    const sources = [owner, solver].flatMap((dir) => {
        const inventory = Bun.spawnSync(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "src"],
            { cwd: dir },
        );
        assert.equal(inventory.exitCode, 0);
        const files = [
            ...new Set(
                inventory.stdout
                    .toString()
                    .trim()
                    .split("\n")
                    .filter(
                        (file) => carried(file) && (dir === solver || !file.startsWith(engine)),
                    ),
            ),
        ];
        assert(files.length > 0, "installed runtime: nonempty canonical owner");
        return files;
    });
    const outputs = [
        ...sources,
        "rust/audio/pkg/shallot_audio.js",
        "rust/audio/pkg/shallot_audio.d.ts",
        "rust/audio/pkg/shallot_audio.wasm",
    ].sort();
    assert(outputs.includes("src/types/env.d.ts"));
    for (const file of outputs)
        assert.equal(
            hash(resolve(shipped, file)),
            hash(canonical(file)),
            `installed runtime: canonical bytes ${file}`,
        );
    console.log(
        `runtime: ${files.length} physical files, ${imports} literal imports, ${outputs.length} canonical files`,
    );
}

/** Artifact-level refusal through the same installed reader used for acceptance. */
export function runtimeArms(project: string): void {
    const shipped = resolve(project, "node_modules/@dylanebert/shallot");
    inspectRuntime(shipped);
    const entry = resolve(shipped, "src/index.ts");
    const source = readFileSync(entry, "utf8");
    try {
        writeFileSync(entry, 'export * from "../../shallot-runtime/src/index";\n');
        assert.throws(() => inspectRuntime(shipped), /outside-package reach/);
        console.log("runtime: installed development-forwarder refusal");
    } finally {
        writeFileSync(entry, source);
    }
    const manifest = resolve(shipped, "package.json");
    const original = readFileSync(manifest, "utf8");
    try {
        const pkg = JSON.parse(original);
        delete pkg.exports["./runtime"];
        writeFileSync(manifest, JSON.stringify(pkg));
        assert.throws(() => inspectRuntime(shipped), /missing public entry \.\/runtime/);
        console.log("runtime: missing public entry refusal");
    } finally {
        writeFileSync(manifest, original);
    }
    writeFileSync(
        resolve(project, "runtime-types.ts"),
        'import {State} from "@dylanebert/shallot";import {installHarness} from "@dylanebert/shallot/harness";import {StepSystem} from "@dylanebert/shallot/physics/core";const state=new State();const extension:Parameters<NonNullable<typeof StepSystem.update>>[0]=state;installHarness(extension);state.dispose();\n',
    );
    const check = (name: string, pass: boolean) => {
        const command = [
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
            "runtime-types.ts",
        ];
        const p = Bun.spawnSync(command, { cwd: project, stdout: "pipe", stderr: "pipe" });
        const output = p.stdout.toString() + p.stderr.toString();
        writeFileSync(resolve(project, `${name}.log`), output);
        assert.equal(p.exitCode === 0, pass, `${name}: ${output}`);
        if (!pass) assert.match(output, /env\.d\.ts/);
        console.log(`runtime: ${name} exit=${p.exitCode}`);
    };
    check("installed-type-interoperability", true);
    const ambient = resolve(shipped, "src/types/env.d.ts");
    assert(!existsSync(`${ambient}.absent`));
    try {
        renameSync(ambient, `${ambient}.absent`);
        check("missing-runtime-ambient", false);
    } finally {
        renameSync(`${ambient}.absent`, ambient);
    }
    check("restored-runtime-types", true);
    inspectRuntime(shipped);
}
