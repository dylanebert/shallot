import { expect, test } from "bun:test";

const ROOT_ORACLE_COMMAND =
    "FOLD_ENSEMBLE_MODE=full bun test ./examples/showcase/ocean/test/fold-anchor.oracle.ts";
export const FOLD_REACH_COMMAND =
    "env FOLD_ENSEMBLE_MODE=reduced bun test ./examples/showcase/ocean/test/fold-anchor.oracle.ts";

test("the choppiness fold-anchor oracle remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-fold"]).toBe(ROOT_ORACLE_COMMAND);
    expect(await Bun.file(new URL("./fold-anchor.oracle.ts", import.meta.url)).exists()).toBe(true);
});
