import { expect, test } from "bun:test";

const ROOT_ORACLE_COMMAND = "bun test ./packages/shallot-ocean/tests/slope.oracle.ts";

test("the slope N-invariance oracle remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-slope"]).toBe(ROOT_ORACLE_COMMAND);
    expect(await Bun.file(new URL("./slope.oracle.ts", import.meta.url)).exists()).toBe(true);
});
