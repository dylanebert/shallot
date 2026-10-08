import { expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { nativeBinary, run } from "./native";

setDefaultTimeout(180_000);
const dir = mkdtempSync(join(tmpdir(), "box3d-callbacks-"));
const build = await Bun.build({ entrypoints: [join(import.meta.dir, "callbacks-scene.ts")], outdir: dir, target: "node", format: "esm" });
if (!build.success) throw new Error(build.logs.join("\n"));
process.on("exit", () => rmSync(dir, { recursive: true, force: true }));
for (const mode of ["filter", "pre", "pressure"]) {
    for (const threads of [1, 4]) {
        test(`${mode} callback contacts, impacts and sensor events equal native every step at ${threads} threads`, () => {
            const expected = run([nativeBinary("callbacks.c"), String(threads), mode]).trim();
            const actual = run(["node", join(dir, "callbacks-scene.js"), String(threads), mode]).trim();
            expect(actual.split("\n").sort()).toEqual(expected.split("\n").sort());
        });
    }
}
