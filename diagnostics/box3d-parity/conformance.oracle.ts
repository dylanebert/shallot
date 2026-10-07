// BOX3D=~/kex/reference/box3d bun test ./diagnostics/box3d-parity/conformance.oracle.ts
// Every subject uses the feature-only WASM build and the x86_64 Box3D build under Rosetta.
import { expect, setDefaultTimeout, test } from "bun:test";
import { join } from "node:path";

setDefaultTimeout(180_000);
for (const subject of ["functions", "public-queries", "joint-defaults", "compound-depth"]) {
    test(`${subject}: native boundary comparisons have only their named todo reds`, () => {
        const process = Bun.spawnSync([
            "bun",
            "test",
            "--preload",
            join(import.meta.dir, "oracle-preload.ts"),
            "--todo",
            join(import.meta.dir, `${subject}.oracle.ts`),
        ]);
        console.info(process.stdout.toString(), process.stderr.toString());
        expect(process.exitCode).toBe(0);
    });
}
