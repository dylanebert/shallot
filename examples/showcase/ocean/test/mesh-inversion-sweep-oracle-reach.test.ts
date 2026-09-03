import { expect, test } from "bun:test";

export const MESH_INVERSION_REACH_COMMAND =
    "bun test ./examples/showcase/ocean/test/mesh-inversion-sweep.oracle.ts";

test("the mesh-inversion-sweep oracle file remains reachable through the root script", async () => {
    const packageJson = await Bun.file(new URL("../../../../package.json", import.meta.url)).json();
    expect(packageJson.scripts["test:ocean-mesh-inversion"]).toBe(MESH_INVERSION_REACH_COMMAND);
    expect(
        await Bun.file(new URL("./mesh-inversion-sweep.oracle.ts", import.meta.url)).exists(),
    ).toBe(true);
});
