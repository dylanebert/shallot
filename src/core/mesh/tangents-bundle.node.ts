import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";

setDefaultTimeout(TIMEOUT.node);

import { mkdtempSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { removeTestDirectory } from "../../../scripts/test-directory";

test("a mesh import without tangent generation bundles no MikkTSpace bytes", async () => {
    const scratch = mkdtempSync(resolve(import.meta.dir, ".tangent-bundle-"));
    try {
        const entry = resolve(scratch, "entry.ts");
        writeFileSync(
            entry,
            'import { cube } from "@dylanebert/shallot/mesh"; console.log(cube());',
        );
        const result = await Bun.build({ entrypoints: [entry], target: "browser", minify: true });
        expect(result.success).toBe(true);
        const text = await Promise.all(result.outputs.map((output) => output.text()));
        expect(text.join("")).not.toContain("AGFzbQ");
        expect(text.join("")).not.toContain("MikkTSpace");
        expect(result.outputs.reduce((sum, output) => sum + output.size, 0)).toBeLessThan(60_000);
    } finally {
        removeTestDirectory(scratch);
    }
});
