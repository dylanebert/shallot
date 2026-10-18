import { expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { run } from "./native";
import { sampleBinary } from "./sample";

const steps = 600;
for (const scene of ["box_stack", "bridge"]) {
    test(`${scene}: upstream sample hashes repeat and a source constant changes them`, () => {
        const binary = sampleBinary();
        const hashes = run([binary, scene, String(steps)]);
        expect(run([binary, scene, String(steps)])).toBe(hashes);
        expect(readFileSync(new URL(`../../examples/physics/tests/${scene === "box_stack" ? "boxStack" : scene}.gold.txt`, import.meta.url), "utf8")).toBe(hashes);
        const copy = mkdtempSync(join(tmpdir(), "box3d-sample-mutation-"));
        try {
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp"]) {
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, name));
            }
            const path = join(copy, scene === "box_stack" ? "sample_stacking.cpp" : "sample_joint.cpp");
            const source = readFileSync(path, "utf8");
            const start = source.indexOf(scene === "box_stack" ? "class BoxStack :" : "class Bridge :");
            const prefix = source.slice(0, start);
            const selected = source.slice(start);
            const constant = scene === "box_stack" ? "float a = 0.5f;" : "shapeDef.density = 20.0f;";
            expect(selected.includes(constant)).toBe(true);
            writeFileSync(path, prefix + selected.replace(constant, scene === "box_stack" ? "float a = 0.6f;" : "shapeDef.density = 21.0f;"));
            const mutated = run([sampleBinary(copy), scene, String(steps)]);
            expect(mutated).not.toBe(hashes);
            expect(hashes.trim().split("\n")).toHaveLength(steps);
            console.log(`${scene}: ${hashes.split("\n")[0]} … ${hashes.trim().split("\n").at(-1)}`);
            console.log(`${scene} mutation: ${mutated.split("\n")[0]} … ${mutated.trim().split("\n").at(-1)}`);
        } finally {
            rmSync(copy, { recursive: true, force: true });
        }
    });
}
