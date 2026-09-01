import { expect, test } from "bun:test";

const ORACLE_COMMAND = "bun test ./tests/slope.oracle.ts";

test("the slope N-invariance oracle remains reachable through its package script", async () => {
    const packageJson = await Bun.file(new URL("../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:slope"]).toBe(ORACLE_COMMAND);
    expect(await Bun.file(new URL("./slope.oracle.ts", import.meta.url)).exists()).toBe(true);
});
