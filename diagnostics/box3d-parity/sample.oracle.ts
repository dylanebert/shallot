import { expect, test } from "bun:test";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./native";
import { sampleBinary } from "./sample";
import { sampleGold } from "../../examples/physics/tests/gold";

function equalSampleSteps(actual: string, expected: string, label: string): void {
    const a = sampleGold(actual), b = sampleGold(expected);
    const actualLines = actual.split("\n"), expectedLines = expected.split("\n");
    expect(a.length, `${label} step ${Math.min(a.length, b.length)} count`).toBe(b.length);
    expect(actualLines.length, `${label} step ${Math.min(actualLines.length, expectedLines.length)} encoding`).toBe(expectedLines.length);
    for (let step = 0; step < expectedLines.length; step++) {
        if (step < b.length) {
            expect(a[step]!.hash, `${label} step ${step} hash`).toBe(b[step]!.hash);
            expect(a[step]!.events, `${label} step ${step} events`).toEqual(b[step]!.events);
        }
        expect(actualLines[step], `${label} step ${step} encoding`).toBe(expectedLines[step]);
    }
}

test("grab sequence reproduces the example's native gold", () => {
    const hashes = run([sampleBinary(), "box_stack", "45", "grab"]);
    equalSampleSteps(run([sampleBinary(), "box_stack", "45", "grab"]), hashes, "grab repeat");
    equalSampleSteps(hashes, readFileSync(new URL("../../examples/physics/tests/grab.gold.txt", import.meta.url), "utf8"), "grab");
});

const steps = 600;
for (const scene of ["box_stack", "bridge", "revolute", "prismatic", "wheel", "filter", "parallel", "motor", "joint", "sensor", "contact", "hit"]) {
    test(`${scene}: upstream sample hashes repeat and a source constant changes them`, () => {
        const binary = sampleBinary();
        const hashes = run([binary, scene, String(steps)]);
        equalSampleSteps(run([binary, scene, String(steps)]), hashes, `${scene} repeat`);
        equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${scene === "box_stack" ? "boxStack" : scene}.gold.txt`, import.meta.url), "utf8"), scene);
        if (["revolute", "prismatic", "wheel", "parallel", "motor", "joint", "contact"].includes(scene)) {
            const sequence = fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url));
            const controlledSteps = scene === "contact" ? 1204 : steps;
            const controlled = run([binary, scene, String(controlledSteps), sequence]);
            equalSampleSteps(run([binary, scene, String(controlledSteps), sequence]), controlled, `${scene} controls repeat`);
            equalSampleSteps(controlled, readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`, import.meta.url), "utf8"), `${scene} controls`);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-sample-mutation-"));
        try {
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp"]) {
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, name));
            }
            const path = join(copy, scene === "box_stack" ? "sample_stacking.cpp" : ["joint", "sensor", "contact", "hit"].includes(scene) ? "sample_events.cpp" : "sample_joint.cpp");
            const source = readFileSync(path, "utf8");
            const classes: Record<string, string> = { box_stack: "BoxStack", bridge: "Bridge", revolute: "RevoluteJoint", prismatic: "PrismaticJoint", wheel: "WheelJoint", filter: "FilterJoint", parallel: "ParallelJoint", motor: "MotorJoint", joint: "JointEvent", sensor: "SensorVisit", contact: "ContactEvent", hit: "HitEvent" };
            const start = source.indexOf(`class ${classes[scene]} :`);
            const prefix = source.slice(0, start);
            const selected = source.slice(start);
            const [constant, replacement] = scene === "box_stack" ? ["float a = 0.5f;", "float a = 0.6f;"] : scene === "bridge" ? ["shapeDef.density = 20.0f;", "shapeDef.density = 21.0f;"] : ["bodyDef.position = { 0.0f, 4.0f, 0.0f };", "bodyDef.position = { 0.0f, 4.1f, 0.0f };"];
            const eventConstants: Record<string, [string, string]> = {
                motor: ["bodyDef.position = { -2.0, 2.0, 0.0 };", "bodyDef.position = { -2.1, 2.0, 0.0 };"],
                joint: ["b3Vec3 position = { -12.5f, 10.0, 0.0f };", "b3Vec3 position = { -12.4f, 10.0, 0.0f };"],
                sensor: ["bodyDef.position = { 0.0f, 12.5f, 0.0f };", "bodyDef.position = { 0.0f, 12.6f, 0.0f };"],
                contact: ["bodyDef.position = { 0.0f, 1.0f, 0.0f };", "bodyDef.position = { 0.0f, 1.1f, 0.0f };"],
                hit: ["float r = 0.75f;", "float r = 0.8f;"],
            };
            const before = eventConstants[scene]?.[0] ?? (scene === "wheel" ? "bodyDef.position = { 0.0f, 2.0f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.0f, 4.0f, 0.0f };" : constant!);
            const after = eventConstants[scene]?.[1] ?? (scene === "wheel" ? "bodyDef.position = { 0.0f, 2.1f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.1f, 4.0f, 0.0f };" : replacement!);
            expect(selected.includes(before)).toBe(true);
            writeFileSync(path, prefix + selected.replace(before, after));
            const mutated = run([sampleBinary(copy), scene, String(steps)]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            expect(hashes.trim().split("\n")).toHaveLength(steps);
            console.log(`${scene}: ${hashes.split("\n")[0]} … ${hashes.trim().split("\n").at(-1)}`);
            console.log(`${scene} mutation: ${mutated.split("\n")[0]} … ${mutated.trim().split("\n").at(-1)}`);
        } finally {
            rmSync(copy, { recursive: true, force: true });
        }
    });
}
