import { expect, test } from "bun:test";

export const SLOPE_REACH_COMMAND = "bun test ./examples/showcase/ocean/test/slope.oracle.ts";

test("the slope cascade oracle file remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-slope"]).toBe(SLOPE_REACH_COMMAND);
    expect(await Bun.file(new URL("./slope.oracle.ts", import.meta.url)).exists()).toBe(true);
});
