import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary, run } from "./native";

setDefaultTimeout(0);
test("public native 2D hull cases repeat their gold and reject changed circle inputs", () => {
    const expected = readFileSync(new URL("../../src/standard/physics/hull2d.gold.json", import.meta.url), "utf8");
    const binary = nativeBinary("hull2d.c");
    expect(run([binary])).toBe(expected);
    expect(run([binary])).toBe(expected);
    const directory = mkdtempSync(join(tmpdir(), "shallot-hull2d-mutation-"));
    try {
        const source = readFileSync(new URL("./hull2d.c", import.meta.url), "utf8");
        const mutated = source.replace("float radius=5.0f;", "float radius=5.1f;");
        expect(mutated).not.toBe(source);
        const file = join(directory, "hull2d.c");
        writeFileSync(file, mutated);
        expect(run([nativeBinary(file)])).not.toBe(expected);
    } finally { rmSync(directory, { recursive: true, force: true }); }
});
