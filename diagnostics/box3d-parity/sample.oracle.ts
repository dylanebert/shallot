import { expect, setDefaultTimeout, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./native";
import { sampleBinary } from "./sample";
import { sampleGold } from "../../examples/physics/tests/gold";

setDefaultTimeout(0);

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
            mkdirSync(join(copy, "samples"));
            mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp"]) {
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            }
            const path = join(copy, "samples", scene === "box_stack" ? "sample_stacking.cpp" : ["joint", "sensor", "contact", "hit"].includes(scene) ? "sample_events.cpp" : "sample_joint.cpp");
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
            const mutated = run([sampleBinary(join(copy, "samples")), scene, String(steps)]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            expect(hashes.trim().split("\n")).toHaveLength(steps);
            console.log(`${scene}: ${hashes.split("\n")[0]} … ${hashes.trim().split("\n").at(-1)}`);
            console.log(`${scene} mutation: ${mutated.split("\n")[0]} … ${mutated.trim().split("\n").at(-1)}`);
        } finally {
            rmSync(copy, { recursive: true, force: true });
        }
    });
}

const benchmarkMutations: Record<string, [string, string]> = {
    fallingTrees: ["int bodyCount = BENCHMARK_DEBUG ? 10 : 50;", "int bodyCount = BENCHMARK_DEBUG ? 10 : 49;"],
    candyCups: ["constexpr int n = m_isDebug ? 4 : 16;", "constexpr int n = m_isDebug ? 4 : 15;"],
    convexPile: ["float amplitude = 2.0f;", "float amplitude = 2.1f;"],
    washer: ["float motorSpeed = 25.0f;", "float motorSpeed = 26.0f;"],
    junkyard: ["g_junkyardData.radius = 35.0f;", "g_junkyardData.radius = 36.0f;"],
    rain: ["position.y = 20.0f;", "position.y = 21.0f;"],
    largePyramid: ["int baseCount = BENCHMARK_DEBUG ? 20 : 100;", "int baseCount = BENCHMARK_DEBUG ? 20 : 99;"],
    widePyramid: ["const float boxSeparation = 0.5f;", "const float boxSeparation = 0.6f;"],
    manyPyramids: ["int baseCount = 10;", "int baseCount = 9;"],
    jointGrid: ["0.0f, 0.0f, 0.0f }, 0.4f", "0.0f, 0.0f, 0.0f }, 0.5f"],
    fallingBoxes: ["float a = 0.5f;", "float a = 0.6f;"],
};
for (const [scene, [before, after]] of Object.entries(benchmarkMutations)) {
    test(`${scene}: release sample repeats its gold and rejects a source mutation`, () => {
        const hashes = run([sampleBinary(), scene, String(steps)]);
        equalSampleSteps(run([sampleBinary(), scene, String(steps)]), hashes, `${scene} repeat`);
        equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url), "utf8"), scene);
        if (scene === "fallingTrees") {
            const sequence = fileURLToPath(new URL("../../examples/physics/tests/fallingTrees.controls.txt", import.meta.url));
            const controlled = run([sampleBinary(), scene, String(steps), sequence]);
            equalSampleSteps(run([sampleBinary(), scene, String(steps), sequence]), controlled, `${scene} controls repeat`);
            equalSampleSteps(controlled, readFileSync(new URL("../../examples/physics/tests/fallingTrees.controls.gold.txt", import.meta.url), "utf8"), `${scene} controls`);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-benchmark-mutation-"));
        try {
            mkdirSync(join(copy, "samples"));
            mkdirSync(join(copy, "shared"));
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            const path = join(copy, ["fallingBoxes", "candyCups"].includes(scene) ? "samples/sample_benchmark.cpp" : "shared/benchmarks.c");
            const source = readFileSync(path, "utf8");
            const start = scene === "fallingBoxes" ? source.indexOf("class FallingBoxes :") : scene === "candyCups" ? source.indexOf("class CandyCups :") : 0;
            expect(source.slice(start).includes(before)).toBe(true);
            writeFileSync(path, source.slice(0, start) + source.slice(start).replace(before, after));
            const mutated = run([sampleBinary(join(copy, "samples")), scene, String(steps)]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
        } finally { rmSync(copy, { recursive: true, force: true }); }
    });
}
