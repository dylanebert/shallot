import { expect, test } from "bun:test";

const ORACLE_COMMAND = "bun test ./packages/shallot-ocean/tests/realization.oracle.ts";

test("the real-space realization oracle remains reachable through its package script", async () => {
    const packageJson = await Bun.file(new URL("../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-realization"]).toBe(ORACLE_COMMAND);
    expect(await Bun.file(new URL("./realization.oracle.ts", import.meta.url)).exists()).toBe(true);
});
