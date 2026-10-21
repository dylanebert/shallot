import { afterAll, beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { allocationGates, gateEntries, gateInputs, selectGates } from "./allocation-gates";
import { removeTestDirectory } from "./test-directory";

setDefaultTimeout(TIMEOUT.node);

const DEMO = "diagnostics/first-person-allocation/demo.node.ts";
const scratch = mkdtempSync(join(tmpdir(), "shallot-allocation-gates-"));
let inputs: Map<string, Set<string>>;

beforeAll(async () => {
    inputs = await gateInputs(allocationGates());
}, TIMEOUT.startup);

afterAll(() => removeTestDirectory(scratch));

test("every allocation gate names an entry that exists", async () => {
    const gates = allocationGates();
    expect(gates).toContain(DEMO);
    for (const gate of gates) expect((await gateEntries(gate)).length).toBeGreaterThan(0);
});

test("an engine module reached only through the package's own name selects the gate that bundles it", () => {
    expect(selectGates(inputs, ["src/core/input/index.ts"])).toContain(DEMO);
});

test("an example module an entry bundles selects its gate", () => {
    expect(selectGates(inputs, ["examples/first-person/src/demo.ts"])).toEqual([DEMO]);
});

test("an asset the harness copies into every bundle selects every gate", () => {
    expect(selectGates(inputs, ["src/extras/text/Inter-Regular.ttf"])).toEqual(allocationGates());
});

test("a source no gate bundles selects no gate", () => {
    expect(selectGates(inputs, ["src/cli/create.ts"])).toEqual([]);
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
