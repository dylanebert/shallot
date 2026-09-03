import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const manifest = JSON.parse(readFileSync(join(import.meta.dir, "../shallot.json"), "utf8")) as {
    plugins: Record<string, true | string>;
};

describe("ocean showcase composition", () => {
    test("contains only the separate Sky and Ocean custom plugins", () => {
        const custom = Object.entries(manifest.plugins).filter(
            ([, value]) => typeof value === "string",
        );
        expect(custom.filter(([name]) => name !== "Capture")).toEqual([
            ["Sky", "./src/sky"],
            ["Ocean", "./src/ocean"],
        ]);
    });
});
