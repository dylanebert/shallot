import { expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./native";
import { sampleBinary } from "./sample";

test("grab sequence reproduces the example's native gold", () => {
    const hashes = run([sampleBinary(), "box_stack", "45", "grab"]);
    expect(run([sampleBinary(), "box_stack", "45", "grab"])).toBe(hashes);
    expect(readFileSync(new URL("../../examples/physics/tests/grab.gold.txt", import.meta.url), "utf8")).toBe(hashes);
});

const steps = 600;
for (const scene of ["box_stack", "bridge", "revolute", "prismatic", "wheel", "filter", "parallel"]) {
    test(`${scene}: upstream sample hashes repeat and a source constant changes them`, () => {
        const binary = sampleBinary();
        const hashes = run([binary, scene, String(steps)]);
        expect(run([binary, scene, String(steps)])).toBe(hashes);
        expect(readFileSync(new URL(`../../examples/physics/tests/${scene === "box_stack" ? "boxStack" : scene}.gold.txt`, import.meta.url), "utf8")).toBe(hashes);
        if (["revolute", "prismatic", "wheel", "parallel"].includes(scene)) {
            const sequence = fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url));
            const controlled = run([binary, scene, String(steps), sequence]);
            expect(run([binary, scene, String(steps), sequence])).toBe(controlled);
            expect(readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`, import.meta.url), "utf8")).toBe(controlled);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-sample-mutation-"));
        try {
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp"]) {
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, name));
            }
            const path = join(copy, scene === "box_stack" ? "sample_stacking.cpp" : "sample_joint.cpp");
            const source = readFileSync(path, "utf8");
            const classes: Record<string, string> = { box_stack: "BoxStack", bridge: "Bridge", revolute: "RevoluteJoint", prismatic: "PrismaticJoint", wheel: "WheelJoint", filter: "FilterJoint", parallel: "ParallelJoint" };
            const start = source.indexOf(`class ${classes[scene]} :`);
            const prefix = source.slice(0, start);
            const selected = source.slice(start);
            const [constant, replacement] = scene === "box_stack" ? ["float a = 0.5f;", "float a = 0.6f;"] : scene === "bridge" ? ["shapeDef.density = 20.0f;", "shapeDef.density = 21.0f;"] : ["bodyDef.position = { 0.0f, 4.0f, 0.0f };", "bodyDef.position = { 0.0f, 4.1f, 0.0f };"];
            const before = scene === "wheel" ? "bodyDef.position = { 0.0f, 2.0f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.0f, 4.0f, 0.0f };" : constant!;
            const after = scene === "wheel" ? "bodyDef.position = { 0.0f, 2.1f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.1f, 4.0f, 0.0f };" : replacement!;
            expect(selected.includes(before)).toBe(true);
            writeFileSync(path, prefix + selected.replace(before, after));
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
