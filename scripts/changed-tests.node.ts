import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { normalizePath } from "../src/project/paths";
import {
    allocationGates,
    changedPaths,
    changedTests,
    gateEntries,
    gateInputs,
    importClosures,
    nodeTests,
    selectTests,
} from "./changed-tests";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

const DEMO = "diagnostics/first-person-allocation/demo.node.ts";
const scratch = mkdtempSync(join(tmpdir(), "shallot-changed-tests-"));
let inputs: Map<string, Set<string>>;
let closures: Map<string, Set<string>>;

beforeAll(async () => {
    inputs = await gateInputs(allocationGates());
    closures = importClosures(nodeTests());
}, TIMEOUT.startup);

afterAll(() => removeTestDirectory(scratch));

const transpiler = new Bun.Transpiler({ loader: "tsx" });

function importers(specifier: string): string[] {
    return nodeTests().filter((path) =>
        transpiler
            .scanImports(readFileSync(resolve(import.meta.dir, "..", path), "utf8"))
            .some((entry) => entry.path === specifier),
    );
}

test("a module reached only through the package's own name selects every test importing it", () => {
    const gpu = importers("@dylanebert/shallot/testing/gpu");
    expect(gpu.length).toBeGreaterThan(0);
    const gpuSelected = selectTests(closures, ["src/project/testing-gpu.ts"]);
    expect(gpu.filter((path) => !gpuSelected.includes(path))).toEqual([]);
    const testing = importers("@dylanebert/shallot/testing");
    expect(testing.length).toBeGreaterThan(0);
    const testingSelected = selectTests(closures, ["src/project/testing.js"]);
    expect(testing.filter((path) => !testingSelected.includes(path))).toEqual([]);
});

test("non-ASCII paths in git listings select their tests", () => {
    const root = realpathSync(mkdtempSync(join(scratch, "unicode-")));
    // Inherited git state stays out: an inherited commit.gpgsign hangs commit, a GIT_TEMPLATE_DIR
    // installs its hooks, and GIT_DIR redirects every command.
    const hooks = mkdtempSync(join(scratch, "hooks-"));
    const config = join(scratch, "unicode.gitconfig");
    writeFileSync(
        config,
        `[user]\n\tname = test\n\temail = test@example.com\n[core]\n\thooksPath = ${normalizePath(hooks)}\n`,
    );
    const env = {
        ...Object.fromEntries(
            Object.entries(process.env).filter(([name]) => !name.toUpperCase().startsWith("GIT_")),
        ),
        GIT_CONFIG_GLOBAL: config,
        GIT_CONFIG_NOSYSTEM: "1",
    };
    const run = (...args: string[]) => {
        const result = Bun.spawnSync(["git", ...args], { cwd: root, env, stderr: "pipe" });
        if (result.exitCode !== 0) throw new Error(result.stderr.toString());
    };
    mkdirSync(join(root, "scripts"));
    writeFileSync(
        join(root, "package.json"),
        '{ "name": "@dylanebert/shallot", "exports": { "./testing": "./testing.js" } }\n',
    );
    writeFileSync(join(root, "testing.js"), "export const TIMEOUT = { node: 1 };\n");
    writeFileSync(join(root, "scripts/réview-helper.ts"), "export const value = 1;\n");
    writeFileSync(
        join(root, "scripts/réview.node.ts"),
        `import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { value } from "./réview-helper";

setDefaultTimeout(TIMEOUT.node);
value;
`,
    );
    run("init", "-q");
    run("add", ".");
    run("commit", "-qm", "fixture");
    writeFileSync(join(root, "scripts/réview-helper.ts"), "export const value = 2;\n");
    writeFileSync(
        join(root, "scripts/naïve.node.ts"),
        `import { setDefaultTimeout } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);
`,
    );

    const tests = nodeTests(root);
    expect(tests).toEqual(["scripts/naïve.node.ts", "scripts/réview.node.ts"]);
    const changed = changedPaths("HEAD", root);
    expect(changed.sort()).toEqual(["scripts/naïve.node.ts", "scripts/réview-helper.ts"]);
    expect(selectTests(importClosures(tests, root), changed, root).sort()).toEqual(tests);
});

test("a changed node test selects itself", () => {
    expect(selectTests(closures, ["scripts/changed-tests.node.ts"])).toContain(
        "scripts/changed-tests.node.ts",
    );
});

test("a gate a change reaches through both its imports and its bundle is printed once", async () => {
    const changed = ["diagnostics/first-person-allocation/allocation.ts"];
    const imported = selectTests(closures, changed);
    const both = selectTests(inputs, changed).filter((gate) => imported.includes(gate));
    expect(both.length).toBeGreaterThan(0);
    const selected = await changedTests(changed);
    for (const gate of both) expect(selected.filter((path) => path === gate)).toEqual([gate]);
});

test("a changed example selects the tests that read examples from disk, each once", async () => {
    const readers = [
        "scripts/icons-packed.node.ts",
        "scripts/icons.node.ts",
        "scripts/packed-install.node.ts",
        "src/cli/create.node.ts",
    ];
    expect(await changedTests(["examples/physics/src/main.ts"])).toEqual(
        expect.arrayContaining(readers),
    );
    const selected = await changedTests(["src/project/testing.js", "examples/physics/src/main.ts"]);
    for (const test of readers) expect(selected.filter((path) => path === test)).toEqual([test]);
});

test("a change no test imports selects none", async () => {
    expect(await changedTests(["README.md"])).toEqual([]);
});

test("an import that does not resolve fails, naming its file", () => {
    const file = join(scratch, "unresolved.node.ts");
    writeFileSync(file, 'import "./absent";\n');
    expect(() => importClosures([file])).toThrow(
        /unresolved\.node\.ts: cannot resolve "\.\/absent"/,
    );
});

test("every allocation gate names an entry that exists", async () => {
    const gates = allocationGates();
    expect(gates).toContain(DEMO);
    for (const gate of gates) expect((await gateEntries(gate)).length).toBeGreaterThan(0);
});

test("an engine module reached only through the package's own name selects the gate that bundles it", () => {
    expect(selectTests(inputs, ["src/core/input/index.ts"])).toContain(DEMO);
});

test("an example module an entry bundles selects its gate", () => {
    expect(selectTests(inputs, ["examples/first-person/src/demo.ts"])).toEqual([DEMO]);
});

test("an asset the harness copies into every bundle selects every gate", () => {
    expect(selectTests(inputs, ["src/extras/text/Inter-Regular.ttf"])).toEqual(allocationGates());
});

test("a source no gate bundles selects no gate", () => {
    expect(selectTests(inputs, ["src/cli/create.ts"])).toEqual([]);
});

test("a gate naming no existing entry fails, naming the gate", async () => {
    const gate = join(scratch, "unnamed.node.ts");
    writeFileSync(gate, 'sampleAllocation("missing.entry.ts");\n');
    await expect(gateEntries(gate)).rejects.toThrow(`${gate}: names no .entry.ts file that exists`);
});

test("a gate whose entry fails to bundle fails, naming the gate", async () => {
    const gate = join(scratch, "broken.node.ts");
    writeFileSync(gate, 'sampleAllocation("broken.entry.ts");\n');
    writeFileSync(join(scratch, "broken.entry.ts"), 'import "./absent";\n');
    writeFileSync(join(scratch, "package.json"), "{}\n");
    await expect(gateInputs([gate])).rejects.toThrow(
        `${gate}: bundle failed: ResolveMessage: Could not resolve: "./absent"`,
    );
});
