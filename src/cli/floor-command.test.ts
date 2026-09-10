import { afterEach, expect, test } from "bun:test";
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    realpathSync,
    rmSync,
    writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { parse, parseExpression } from "@babel/parser";

const dirs: string[] = [];
afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function fixture(required: boolean, gap: boolean) {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "shallot-floor-command-")));
    dirs.push(dir);
    writeFileSync(
        join(dir, "shallot.json"),
        JSON.stringify({ plugins: { Physics: true, Local: "external-floor-plugin" } }),
    );
    const plugin = join(dir, "node_modules/external-floor-plugin");
    mkdirSync(plugin, { recursive: true });
    writeFileSync(
        join(plugin, "package.json"),
        JSON.stringify({ name: "external-floor-plugin", main: "index.js" }),
    );
    writeFileSync(
        join(plugin, "index.js"),
        `console.log("PLUGIN_LOADED"); export default { name: "Local", ${required ? "features" : "preferredFeatures"}: ["timestamp-query", "subgroups"] };`,
    );
    const preload = join(dir, "preload.ts");
    writeFileSync(
        preload,
        `import { WEBVIEW_UNSUPPORTED } from ${JSON.stringify(resolve(import.meta.dir, "../engine/runtime/floor.ts"))}; ${gap ? 'WEBVIEW_UNSUPPORTED.mac = ["timestamp-query"];' : ""}`,
    );
    const id = crypto.randomUUID();
    const receipt = join(dir, "config-receipt.json");
    // The real native bundle functions load project configuration before cargo or app launch. This
    // fixture exits at that boundary, so the receipt proves the production path without compiling or
    // launching a native target.
    writeFileSync(
        join(dir, "vite.config.ts"),
        `
import { writeFileSync } from "node:fs";
export default function config({ command, mode }) {
    writeFileSync(${JSON.stringify(receipt)}, JSON.stringify({ id: ${JSON.stringify(id)}, command, mode, project: process.cwd(), configDir: import.meta.dirname }));
    console.log("PROJECT_CONFIG_REACHED");
    process.exit(0);
}
`,
    );
    return { dir, preload, receipt, id };
}

function commandReceipt(
    command: string,
    target: string,
    portable: boolean,
    project: ReturnType<typeof fixture>,
) {
    expect(existsSync(join(project.dir, "index.html"))).toBe(false);
    const result = Bun.spawnSync(
        [
            process.execPath,
            "--preload",
            project.preload,
            resolve(import.meta.dir, "../../bin/shallot.ts"),
            command,
            project.dir,
            "--target",
            target,
            ...(portable ? ["--portable"] : []),
        ],
        { cwd: project.dir, env: process.env },
    );
    return { result, output: result.stdout.toString() + result.stderr.toString() };
}

function reached(project: ReturnType<typeof fixture>) {
    expect(JSON.parse(readFileSync(project.receipt, "utf8"))).toEqual({
        id: project.id,
        command: "build",
        mode: "production",
        project: project.dir,
        configDir: project.dir,
    });
    expect(existsSync(join(project.dir, "build"))).toBe(false);
    expect(existsSync(join(project.dir, "dist"))).toBe(false);
    expect(existsSync(join(project.dir, "index.html"))).toBe(false);
}

test("qualified real project config receipt precedes native tooling", () => {
    const project = fixture(false, false);
    const { result, output } = commandReceipt("build", "mac", false, project);
    expect({ exit: result.exitCode, output }).toEqual({
        exit: 0,
        output: expect.stringContaining("PROJECT_CONFIG_REACHED"),
    });
    expect(output).toContain("PLUGIN_LOADED");
    reached(project);
});

type Syntax = { type: string; [key: string]: unknown };
type Site = { node: Syntax; parents: Syntax[] };

// The receipt cannot observe native output/options, so this independently parses the live CLI
// binding and follows the imported function, branch and argument cardinality rather than matching text.
function sites(value: unknown, parents: Syntax[] = []): Site[] {
    if (!value || typeof value !== "object") return [];
    if (Array.isArray(value)) return value.flatMap((child) => sites(child, parents));
    const node = value as Syntax;
    if (typeof node.type !== "string") return [];
    return [
        { node, parents },
        ...Object.entries(node).flatMap(([key, child]) =>
            [
                "loc",
                "start",
                "end",
                "extra",
                "comments",
                "leadingComments",
                "trailingComments",
                "innerComments",
            ].includes(key)
                ? []
                : sites(child, [...parents, node]),
        ),
    ];
}

function semantic(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(semantic);
    if (!value || typeof value !== "object") return value;
    return Object.fromEntries(
        Object.entries(value)
            .filter(
                ([key]) =>
                    ![
                        "loc",
                        "start",
                        "end",
                        "extra",
                        "errors",
                        "comments",
                        "leadingComments",
                        "trailingComments",
                        "innerComments",
                    ].includes(key),
            )
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, child]) => [key, semantic(child)]),
    );
}

const ast = (file: string) =>
    parse(readFileSync(resolve(import.meta.dir, file), "utf8"), {
        sourceType: "module",
        plugins: ["typescript"],
    });
const expression = (source: string) => semantic(parseExpression(source));
const isName = (node: unknown, name: string) =>
    (node as Syntax)?.type === "Identifier" && (node as Syntax).name === name;

function imported(tree: ReturnType<typeof ast>, name: string, from: string): string {
    const matches = tree.program.body
        .filter((node) => node.type === "ImportDeclaration")
        .flatMap((node) =>
            node.specifiers
                .filter(
                    (specifier) =>
                        specifier.type === "ImportSpecifier" &&
                        specifier.imported.type === "Identifier" &&
                        specifier.imported.name === name,
                )
                .map((specifier) => ({ local: specifier.local.name, from: node.source.value })),
        );
    expect(matches).toHaveLength(1);
    expect(matches[0].from).toBe(from);
    const local = matches[0].local;
    // A same-spelled local would hide the imported binding. Refuse that ambiguous production shape
    // rather than claiming the AST check resolved a different symbol.
    for (const { node } of sites(tree)) {
        const declarations =
            node.type === "VariableDeclarator"
                ? [node.id]
                : ["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression"].includes(
                        node.type,
                    )
                  ? [node.id, node.params]
                  : node.type === "CatchClause"
                    ? [node.param]
                    : [];
        expect(
            declarations.flatMap((decl) => sites(decl)).some(({ node }) => isName(node, local)),
        ).toBe(false);
        if (node.type === "AssignmentExpression") expect(isName(node.left, local)).toBe(false);
    }
    return local;
}

function calls(tree: unknown, name: string): Site[] {
    return sites(tree).filter(
        ({ node }) => node.type === "CallExpression" && isName(node.callee, name),
    );
}

function owner(site: Site, name: string): Syntax {
    const fn = site.parents.findLast((node) => node.type === "FunctionDeclaration");
    expect((fn?.id as Syntax)?.name).toBe(name);
    return fn!;
}

function declaration(site: Site, name: string): Syntax {
    for (const block of site.parents.toReversed()) {
        if (block.type !== "BlockStatement" && block.type !== "Program") continue;
        const matches = (block.body as Syntax[])
            .filter((node) => node.type === "VariableDeclaration")
            .flatMap((node) =>
                (node.declarations as Syntax[])
                    .filter((decl) => isName(decl.id, name))
                    .map((decl) => ({ kind: node.kind, decl })),
            );
        if (!matches.length) continue;
        expect(matches).toHaveLength(1);
        expect(matches[0].kind).toBe("const");
        expect(matches[0].decl.start as number).toBeLessThan(site.node.start as number);
        return matches[0].decl.init as Syntax;
    }
    throw new Error(`missing local binding ${name}`);
}

function argumentsAre(site: Site, args: string): void {
    expect(semantic(site.node.arguments)).toEqual(
        semantic((parseExpression(`[${args}]`) as unknown as Syntax).elements),
    );
}

for (const command of ["build", "run"] as const) {
    test(`CLI ${command} dispatch binding preserves project and options`, () => {
        const tree = ast("index.ts");
        const local = imported(tree, `${command}Project`, `./${command}`);
        const branch = sites(tree).filter(
            ({ node }) =>
                node.type === "IfStatement" &&
                JSON.stringify(semantic(node.test)) ===
                    JSON.stringify(expression(`parsed.subcmd === "${command}"`)),
        );
        expect(branch).toHaveLength(1);
        const invokes = calls(branch[0].node.consequent, local);
        expect(invokes).toHaveLength(1);
        expect(
            calls(branch[0].node.consequent, command === "build" ? "runProject" : "buildProject"),
        ).toHaveLength(0);
        const full = calls(tree, local).find((site) => site.node === invokes[0].node)!;
        expect(semantic(declaration(full, "projectDir"))).toEqual(
            expression("resolve(parsed.dir)"),
        );
        imported(tree, "resolve", "node:path");
        argumentsAre(
            full,
            command === "build"
                ? "projectDir, { target: parsed.target, release: parsed.release, portable: parsed.portable }"
                : "projectDir, { target: parsed.target, port: parsed.port, release: parsed.release, portable: parsed.portable }",
        );
    });

    for (const target of ["windows", "mac", "linux"] as const) {
        test(`${command} ${target} native binding preserves target, output and options`, () => {
            const tree = ast(`${command}.ts`);
            const bundle = `bundleNative${target[0].toUpperCase()}${target.slice(1)}`;
            const local = imported(tree, bundle, "../native");
            const output = imported(tree, "nativeOutDir", "../native");
            const invokes = calls(tree, local);
            expect(invokes).toHaveLength(1);
            const call = invokes[0];
            const fn = owner(call, `${command}Project`);
            expect((fn.params as Syntax[]).map((node) => node.name)).toEqual([
                "projectDir",
                "opts",
            ]);
            argumentsAre(
                call,
                command === "build"
                    ? "projectDir, outputDir, bundleOpts"
                    : "projectDir, outputDir, { release, portable }",
            );
            expect(semantic(declaration(call, "release"))).toEqual(
                expression("opts.release ?? false"),
            );
            expect(semantic(declaration(call, "portable"))).toEqual(
                expression("opts.portable ?? false"),
            );
            const out = declaration(call, "outputDir");
            expect(isName(out.callee, output)).toBe(true);
            expect(semantic(out.arguments)).toEqual(
                semantic(
                    (
                        parseExpression(
                            command === "build"
                                ? "[projectDir, target, release, portable]"
                                : `[projectDir, "${target}", release, portable]`,
                        ) as unknown as Syntax
                    ).elements,
                ),
            );
            if (command === "build") {
                expect(semantic(declaration(call, "target"))).toEqual(expression("opts.target"));
                expect(semantic(declaration(call, "bundleOpts"))).toEqual(
                    expression("({ release, portable })"),
                );
                const branches = call.parents.filter((node) => node.type === "IfStatement");
                const last = branches.at(-1)!;
                expect(semantic(last.test)).toEqual(
                    expression(`target === "${target === "linux" ? "mac" : target}"`),
                );
                expect(
                    sites(target === "linux" ? last.alternate : last.consequent).some(
                        ({ node }) => node === call.node,
                    ),
                ).toBe(true);
            } else {
                expect(semantic(declaration(call, "runTarget"))).toEqual(
                    expression("resolveRunTarget(opts.target)"),
                );
                const branches = call.parents.filter((node) => node.type === "IfStatement");
                if (target === "windows") {
                    expect(branches).toHaveLength(0);
                    const guards = (fn.body as Syntax).body as Syntax[];
                    for (const other of ["web", "unknown", "mac", "linux"]) {
                        const guard = guards.find(
                            (node) =>
                                node.type === "IfStatement" &&
                                JSON.stringify(semantic(node.test)) ===
                                    JSON.stringify(expression(`runTarget.kind === "${other}"`)),
                        );
                        expect(guard).toBeDefined();
                        expect(guard!.start as number).toBeLessThan(call.node.start as number);
                    }
                } else {
                    expect(semantic(branches.at(-1)!.test)).toEqual(
                        expression(`runTarget.kind === "${target}"`),
                    );
                }
            }
        });
    }
}

for (const target of ["Windows", "Mac", "Linux"]) {
    test(`native ${target} binding reaches real buildWeb with its project`, () => {
        const tree = ast("../native/index.ts");
        const local = imported(tree, "buildWeb", "../cli/build");
        const all = calls(tree, local);
        expect(all).toHaveLength(3);
        const invokes = all.filter(
            (site) =>
                (site.parents.findLast((node) => node.type === "FunctionDeclaration")?.id as Syntax)
                    ?.name === `bundleNative${target}`,
        );
        expect(invokes).toHaveLength(1);
        const fn = owner(invokes[0], `bundleNative${target}`);
        expect((fn.params as Syntax[]).map((node) => node.name)).toEqual([
            "projectDir",
            "outputDir",
            "opts",
        ]);
        argumentsAre(invokes[0], "projectDir");
    });
}

for (const command of ["build", "run"]) {
    for (const [target, portable, allowed, gap, required] of [
        ["linux", false, false, false, false],
        ["linux", true, true, false, false],
        ["mac", false, true, false, false],
        ["windows", false, true, false, false],
        ["mac", false, false, true, true],
        ["mac", false, true, true, false],
        ["mac", true, true, true, true],
    ] as const) {
        test(`${command} ${target} portable=${portable} gap=${gap} required=${required}: ${allowed ? "reaches project config" : "refuses before config"}`, () => {
            const project = fixture(required, gap);
            const { result, output } = commandReceipt(command, target, portable, project);
            if (allowed) {
                expect({ exit: result.exitCode, output }).toEqual({
                    exit: 0,
                    output: expect.stringContaining("PROJECT_CONFIG_REACHED"),
                });
                reached(project);
            } else {
                expect(output).toContain("Cannot build");
                expect(output).toContain(gap ? "timestamp-query" : "WebGPU base floor");
                expect(output).toContain("--portable");
                expect(result.exitCode).toBe(1);
                expect(output).not.toContain("PROJECT_CONFIG_REACHED");
                expect(existsSync(project.receipt)).toBe(false);
                expect(existsSync(join(project.dir, "build"))).toBe(false);
                expect(existsSync(join(project.dir, "dist"))).toBe(false);
            }
            expect(output).toContain("PLUGIN_LOADED");
        });
    }
}
