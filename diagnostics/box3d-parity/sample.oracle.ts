import { expect, setDefaultTimeout, test } from "bun:test";
import { readdirSync, existsSync, copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./native";
import { sampleBinary } from "./sample";
import { nativeCompoundRecords } from "./compound-records";
import { nativeMeshRecords } from "./mesh-records";
import { nativeMeshFrames } from "./mesh-frames";
import { sampleGold } from "../../examples/physics/tests/gold";
import { digestMismatch, drawingDigest, eventDigests, querySteps, recordDigest } from "../../examples/physics/tests/digest";

setDefaultTimeout(0);

function copyRobustnessSources(directory: string): void {
    for (const file of ["samples/sample_issues.cpp", "samples/sample_tree.cpp", "samples/sample_world.cpp", "samples/sample_robustness.cpp", "samples/sample_determinism.cpp", "shared/overflow_color.c", "shared/determinism.c", "shared/stability.c"])
        copyFileSync(join(process.env.BOX3D!, file), join(directory, file));
}

for (const [scene, name, before, after] of [
    ["ghostCollisions", "SBoxGhostCollisions", "m_characterMass = 500.0f", "m_characterMass = 501.0f"],
    ["wheelStack", "WheelStack", "friction = 0.6f", "friction = 0.7f"],
    ["restitutionOvershoot", "RestitutionOvershoot", "m_dropHeight = 10.0f", "m_dropHeight = 11.0f"],
    ["crash", "Crash", "{ 2.0f, 4.0f, 0.0f }", "{ 2.1f, 4.0f, 0.0f }"],
    ["multiplePrismatic", "MultiplePrismatic", "{ 0.0f, -0.6f, 0.0f }", "{ 0.0f, -0.7f, 0.0f }"],
    ["hullCrash", "HullCrash", "{ 100.000000, -142.292389, 130.826111 }", "{ 101.000000, -142.292389, 130.826111 }"],
    ["convexJitter", "ConvexJitter", "rollingResistance = 0.1f;", "rollingResistance = 0.2f;"],
    ["sboxMover", "SBoxMover", "{ 0.0f, 3.5f, 0.0f }", "{ 0.0f, 3.6f, 0.0f }"],
    ["capsuleMesh", "CapsuleMeshBug", "{ 0.0f, 4.0f, 10.0f }", "{ 0.0f, 4.1f, 10.0f }"],
    ["slideTwistOffCenterShape", "SlideTwistOffCenterShape", "b3Body_SetAngularVelocity( boxBody, 25.0f", "b3Body_SetAngularVelocity( boxBody, 26.0f"],
    ["hugeBox", "HugeBox", "float a = 100.0f;", "float a = 101.0f;"],
    ["issueHeightfield", "HeightfieldIssue", "{ 50, 20, 50 }", "{ 50, 21, 50 }"],
] as const) {
    test(`${scene}: Issues native steps repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start = performance.now(), binary = sampleBinary();
        const directory = mkdtempSync(join(tmpdir(), "box3d-issues-"));
        const base = new URL("../../examples/physics/tests/issues/", import.meta.url), records = join(directory, "geometry.jsonl");
        const hasReadouts = scene === "ghostCollisions" || scene === "restitutionOvershoot";
        const env: Record<string, string> = scene === "hullCrash" ? { GEOMETRY_RECORDS: records } : hasReadouts ? { ISSUE_READOUTS: records } : {};
        const readouts = () => readFileSync(records, "utf8");
        const geometry = () => readFileSync(records, "utf8").trim().split("\n").map(line => drawingDigest(JSON.parse(line))).join("\n") + "\n";
        try {
            const hashes = run([binary, scene, "600"], env);
            equalSampleSteps(hashes, readFileSync(new URL(`${scene}.gold.txt`, base), "utf8"), scene);
            if (hasReadouts) expect(readouts()).toBe(readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8"));
            const hull = scene === "hullCrash" ? geometry() : "";
            if (scene === "hullCrash") expect(hull).toBe(readFileSync(new URL("hullCrash.geometry.gold.txt", base), "utf8"));
            equalSampleSteps(run([binary, scene, "600"], env), hashes, `${scene} repeat`);
            if (scene === "hullCrash") expect(geometry()).toBe(hull);
            if (hasReadouts) expect(readouts()).toBe(readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8"));
            if (scene === "crash" || scene === "ghostCollisions") {
                const args = [binary, scene, "600", fileURLToPath(new URL(`${scene}.controls.txt`, base))];
                const controlled = run(args, env);
                if(hasReadouts) expect(readouts()).toBe(readFileSync(new URL(`${scene}.controls.readouts.gold.txt`,base),"utf8"));
                equalSampleSteps(controlled, readFileSync(new URL(`${scene}.controls.gold.txt`, base), "utf8"), `${scene} controls`);
                equalSampleSteps(run(args, env), controlled, `${scene} controls repeat`);
                if(hasReadouts) expect(readouts()).toBe(readFileSync(new URL(`${scene}.controls.readouts.gold.txt`,base),"utf8"));
                expect(controlled).not.toBe(hashes);
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const file of readdirSync(join(process.env.BOX3D!, "samples")))
                if (file.endsWith(".cpp") || file.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", file), join(samples, file));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(samples, "sample_issues.cpp"), source = readFileSync(path, "utf8"), offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThanOrEqual(0);
            expect(source.slice(offset)).toContain(before);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            const mutated = run([sampleBinary(samples), scene, "600"], env);
            if (scene === "hullCrash") expect(geometry()).not.toBe(hull);
            else expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            console.log(`${scene} native parity, repeat, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

test("treeBenchmark: native tree records repeat, controls use DrawControls, and a source mutation fails parity", () => {
    const start = performance.now(), binary = sampleBinary();
    const directory = mkdtempSync(join(tmpdir(), "box3d-tree-sample-"));
    const base = new URL("../../examples/physics/tests/", import.meta.url), records = join(directory, "records.txt"), bounds = join(directory, "bounds.txt");
    const observe = (binary: string, controlled: boolean) => {
        const args = [binary, "treeBenchmark", "600", ...(controlled ? [fileURLToPath(new URL("treeBenchmark.controls.txt", base))] : [])];
        const hashes = run(args, { TREE_RECORDS: records, TREE_BOUNDS: bounds, TREE_SAVE_SLOT: join(directory, "tree.dt") });
        return { hashes, records: readFileSync(records, "utf8").trim().split("\n").map(line => drawingDigest(JSON.parse(line))), bounds: readFileSync(bounds, "utf8").trim().split("\n").map(line => drawingDigest(JSON.parse(line))) };
    };
    try {
        let plain: string[] = [];
        for (const controlled of [false, true]) {
            const stem = `treeBenchmark${controlled ? ".controls" : ""}`, actual = observe(binary, controlled);
            equalSampleSteps(actual.hashes, readFileSync(new URL(`${stem}.gold.txt`, base), "utf8"), stem);
            expect(actual.records).toHaveLength(600);
            expect(actual.records).toEqual(readFileSync(new URL(`${stem}.tree.gold.txt`, base), "utf8").trim().split("\n"));
            expect(actual.bounds).toHaveLength(600);
            expect(actual.bounds).toEqual(readFileSync(new URL(`${stem}.bounds.gold.txt`, base), "utf8").trim().split("\n"));
            expect(observe(binary, controlled)).toEqual(actual);
            if (!controlled) plain = actual.records;
            else expect(actual.records).not.toEqual(plain);
        }
        const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
        for (const name of readdirSync(join(process.env.BOX3D!, "samples")))
            if (name.endsWith(".cpp") || name.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", name), join(samples, name));
        copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
        copyRobustnessSources(directory);
        const path = join(samples, "sample_tree.cpp"), source = readFileSync(path, "utf8");
        const before = "m_config[0] = { 1.0f, false };";
        expect(source).toContain(before);
        writeFileSync(path, source.replace(before, "m_config[0] = { 1.1f, false };"));
        expect(observe(sampleBinary(samples), false).records).not.toEqual(plain);
    } finally { rmSync(directory, { recursive: true, force: true }); }
    console.log(`Tree Benchmark native repeat, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
});

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

for (const [scene, file, before, after] of [
    ["fallingRagdolls", "determinism.c", "float frictionTorque = 5.0f;", "float frictionTorque = 6.0f;"],
    ["wavePile", "determinism.c", "float spacing = 1.7f;", "float spacing = 1.8f;"],
    ["querySpawn", "determinism.c", "bodyDef.linearDamping = 1.0f;", "bodyDef.linearDamping = 1.1f;"],
    ["meshDropDeterminism", "stability.c", "float groundAmplitude = 0.5f;", "float groundAmplitude = 0.6f;"],
] as const) {
    test(`${scene}: native sample readouts repeat and a shared source mutation fails parity`, () => {
        const binary = sampleBinary(), directory = mkdtempSync(join(tmpdir(), "box3d-determinism-"));
        const base = new URL("../../examples/physics/tests/", import.meta.url), rows = join(directory, "rows.txt");
        try {
            const args = [binary, scene, "600"], env = { DETERMINISM_READOUTS: rows };
            const hashes = run(args, env), readouts = readFileSync(rows, "utf8");
            equalSampleSteps(hashes, readFileSync(new URL(`${scene}.gold.txt`, base), "utf8"), scene);
            expect(readouts).toBe(readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8"));
            equalSampleSteps(run(args, env), hashes, `${scene} repeat`);
            expect(readFileSync(rows, "utf8")).toBe(readouts);
            if (scene === "querySpawn") {
                run([binary, scene, "681"], env);
                expect(`${readFileSync(rows, "utf8").trim().split("\n").at(-1)}\n`).toBe(readFileSync(new URL("querySpawn.sleep.gold.txt", base), "utf8"));
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const name of readdirSync(join(process.env.BOX3D!, "samples")))
                if (name.endsWith(".cpp") || name.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", name), join(samples, name));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(directory, "shared", file), source = readFileSync(path, "utf8");
            expect(source).toContain(before);
            writeFileSync(path, source.replace(before, after));
            const mutated = run([sampleBinary(samples), scene, "600"]);
            expect(sampleGold(mutated).map(s => s.hash)).not.toEqual(sampleGold(hashes).map(s => s.hash));
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

for (const [scene, name, before, after] of [
    ["farStack", "FarStack", "m_columnCount = 6;", "m_columnCount = 7;"],
    ["farPyramid", "FarPyramid", "int baseCount = 40;", "int baseCount = 39;"],
    ["farRagdolls", "FarRagdolls", "m_count = 20;", "m_count = 19;"],
    ["farMeshDrop", "FarMeshDrop", "m_offsetKilometers = 1000.0f;", "m_offsetKilometers = 1001.0f;"],
] as const) {
    test(`${scene}: native steps and readouts repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start = performance.now(), binary = sampleBinary();
        const directory = mkdtempSync(join(tmpdir(), "box3d-world-"));
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const readouts = join(directory, "readouts.txt");
        try {
            const hashes = run([binary, scene, "600"], { WORLD_READOUTS: readouts });
            const rows = readFileSync(readouts, "utf8");
            equalSampleSteps(hashes, readFileSync(new URL(`${scene}.gold.txt`, base), "utf8"), scene);
            equalSampleSteps(run([binary, scene, "600"], { WORLD_READOUTS: readouts }), hashes, `${scene} repeat`);
            expect(readFileSync(readouts, "utf8")).toBe(rows);
            expect(rows).toBe(readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8"));
            expect(sampleGold(hashes)).toHaveLength(600);
            if (scene === "farStack") {
                const args = [binary, scene, "600", fileURLToPath(new URL(`${scene}.controls.txt`, base))];
                const controlled = run(args, { WORLD_READOUTS: readouts });
                const controlledRows = readFileSync(readouts, "utf8");
                equalSampleSteps(controlled, readFileSync(new URL(`${scene}.controls.gold.txt`, base), "utf8"), `${scene} controls`);
                equalSampleSteps(run(args, { WORLD_READOUTS: readouts }), controlled, `${scene} controls repeat`);
                expect(readFileSync(readouts, "utf8")).toBe(controlledRows);
                expect(controlledRows).toBe(readFileSync(new URL(`${scene}.controls.readouts.gold.txt`, base), "utf8"));
                expect(controlled).not.toBe(hashes);
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const file of readdirSync(join(process.env.BOX3D!, "samples")))
                if (file.endsWith(".cpp") || file.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", file), join(samples, file));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(samples, "sample_world.cpp"), source = readFileSync(path, "utf8"), offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThanOrEqual(0);
            expect(source.slice(offset)).toContain(before);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            const mutated = run([sampleBinary(samples), scene, "600"]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            console.log(`${scene} native parity, readouts, repeats, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

for (const [scene, name, before, after] of [
    ["highMassRatio1", "HighMassRatio1", "float extent = 1.0f;", "float extent = 1.1f;"],
    ["tinyPyramid", "TinyPyramid", "m_extent = 0.025f;", "m_extent = 0.026f;"],
    ["overlapRecovery", "OverlapRecovery", "m_overlap = 0.25f;", "m_overlap = 0.26f;"],
    ["overflowColorPile", "", "shapeDef.density = 50.0f;", "shapeDef.density = 51.0f;"],
] as const) {
    test(`${scene}: native steps and readouts repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start = performance.now(), binary = sampleBinary();
        const directory = mkdtempSync(join(tmpdir(), "box3d-robustness-"));
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const readouts = join(directory, "readouts.txt");
        const env: Record<string, string> = scene === "overflowColorPile" ? { ROBUSTNESS_READOUTS: readouts } : {};
        try {
            const hashes = run([binary, scene, "600"], env);
            const rows = scene === "overflowColorPile" ? readFileSync(readouts, "utf8") : "";
            equalSampleSteps(hashes, readFileSync(new URL(`${scene}.gold.txt`, base), "utf8"), scene);
            equalSampleSteps(run([binary, scene, "600"], env), hashes, `${scene} repeat`);
            expect(sampleGold(hashes)).toHaveLength(600);
            if (scene === "overflowColorPile") {
                expect(readFileSync(readouts, "utf8")).toBe(rows);
                expect(rows).toBe(readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8"));
                expect(rows.trimEnd().split("\n")).toHaveLength(600);
            }
            if (scene === "overlapRecovery") {
                const args = [binary, scene, "600", fileURLToPath(new URL(`${scene}.controls.txt`, base))];
                const controlled = run(args);
                equalSampleSteps(controlled, readFileSync(new URL(`${scene}.controls.gold.txt`, base), "utf8"), `${scene} controls`);
                equalSampleSteps(run(args), controlled, `${scene} controls repeat`);
                expect(controlled).not.toBe(hashes);
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const file of readdirSync(join(process.env.BOX3D!, "samples")))
                if (file.endsWith(".cpp") || file.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", file), join(samples, file));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(directory, scene === "overflowColorPile" ? "shared/overflow_color.c" : "samples/sample_robustness.cpp");
            const source = readFileSync(path, "utf8"), offset = name ? source.indexOf(`class ${name} :`) : 0;
            expect(offset).toBeGreaterThanOrEqual(0);
            expect(source.slice(offset)).toContain(before);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            const mutated = run([sampleBinary(samples), scene, "600"]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            console.log(`${scene} native parity, readouts, repeats, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

for (const [scene, name, before, after] of [
    ["sphereVsSphere", "SphereAndSphere", "2.0f };", "2.1f };"],
    ["capsuleVsSphere", "CapsuleAndSphere", "2.0f };", "2.1f };"],
    ["hullVsSphere", "HullAndSphere", "1.0f };", "1.1f };"],
    ["triangleVsSphere", "TriangleAndSphere", "0.25f };", "0.26f };"],
    ["capsuleVsCapsule", "CapsuleAndCapsule", "1.0f };", "1.1f };"],
    ["capsuleVsHull", "CapsuleAndHull", "0.15f };", "0.16f };"],
    ["triangleVsCapsule", "TriangleAndCapsule", "0.05f };", "0.06f };"],
    ["hullVsHull", "HullAndHull", "0.5f, 1.0f, 1.0f, transform", "0.6f, 1.0f, 1.0f, transform"],
    ["triangleVsHull", "TriangleAndHull", "0.304800004f;", "0.314800004f;"],
] as const) {
    test(`${scene}: native manifolds repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start=performance.now(), binary=sampleBinary();
        const directory=mkdtempSync(join(tmpdir(),"box3d-manifold-")),records=join(directory,"manifold.jsonl"),base=new URL("../../examples/physics/tests/",import.meta.url);
        const digests=()=>readFileSync(records,"utf8").trimEnd().split("\n").map(line=>drawingDigest(JSON.parse(line))).join("\n")+"\n";
        try {
            for(const controlled of [false,true]) {
                const stem=scene+(controlled?".controls":""),args=[binary,scene,"600",...(controlled?[fileURLToPath(new URL(`${scene}.controls.txt`,base))]:[])];
                const hashes=run(args,{MANIFOLD_RECORDS:records}),manifold=digests();
                equalSampleSteps(hashes,readFileSync(new URL(`${stem}.gold.txt`,base),"utf8"),stem);
                expect(manifold).toBe(readFileSync(new URL(`${stem}.manifold.gold.txt`,base),"utf8"));
                expect(manifold.trimEnd().split("\n")).toHaveLength(600);
                equalSampleSteps(run(args,{MANIFOLD_RECORDS:records}),hashes,`${stem} repeat`);
                expect(digests()).toBe(manifold);
            }
            const samples=join(directory,"samples");mkdirSync(samples);mkdirSync(join(directory,"shared"));
            for(const file of readdirSync(join(process.env.BOX3D!,"samples")))
                if(file.endsWith(".cpp")||file.endsWith(".h"))copyFileSync(join(process.env.BOX3D!,"samples",file),join(samples,file));
            copyFileSync(join(process.env.BOX3D!,"shared/benchmarks.c"),join(directory,"shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path=join(samples,"sample_manifold.cpp"),source=readFileSync(path,"utf8"),offset=source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0);expect(source.slice(offset)).toContain(before);
            writeFileSync(path,source.slice(0,offset)+source.slice(offset).replace(before,after));
            run([sampleBinary(samples),scene,"600"],{MANIFOLD_RECORDS:records});
            expect(digests()).not.toBe(readFileSync(new URL(`${scene}.manifold.gold.txt`,base),"utf8"));
            console.log(`${scene} native parity, repeats, controls and mutation: ${(performance.now()-start).toFixed(0)} ms`);
        } finally {rmSync(directory,{recursive:true,force:true});}
    });
}

for (const [scene, name, before, after] of [
    ["boxHull", "BoxHull", "1.0f, 0.5f, 0.25f", "1.1f, 0.5f, 0.25f"],
    ["geometryHull", "Hull", "0.01f * points[i]", "0.011f * points[i]"],
    ["hullReduction", "HullReduction", "m_count = 16;", "m_count = 15;"],
    ["hullTransform", "HullTransform", "0.0f, 9", "0.0f, 8"],
    ["capsuleMass", "CapsuleMass", "m_radius = 1.0f;", "m_radius = 1.1f;"],
] as const) {
    test(`${scene}: native geometry records repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start = performance.now(), binary = sampleBinary();
        const directory = mkdtempSync(join(tmpdir(), "box3d-geometry-"));
        const records = join(directory, "geometry.jsonl");
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const digests = () => readFileSync(records, "utf8").trimEnd().split("\n").map(line => drawingDigest(JSON.parse(line))).join("\n") + "\n";
        try {
            for (const controlled of [false, true]) {
                const sequence = fileURLToPath(new URL(`${scene}.controls.txt`, base));
                if (controlled && !existsSync(sequence)) continue;
                const stem = `${scene}${controlled ? ".controls" : ""}`;
                const args = [binary, scene, "600", ...(controlled ? [sequence] : [])];
                const hashes = run(args, { GEOMETRY_RECORDS: records }), geometry = digests();
                equalSampleSteps(hashes, readFileSync(new URL(`${stem}.gold.txt`, base), "utf8"), stem);
                expect(geometry).toBe(readFileSync(new URL(`${stem}.geometry.gold.txt`, base), "utf8"));
                expect(geometry.trimEnd().split("\n")).toHaveLength(600);
                equalSampleSteps(run(args, { GEOMETRY_RECORDS: records }), hashes, `${stem} repeat`);
                expect(digests()).toBe(geometry);
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const file of readdirSync(join(process.env.BOX3D!, "samples")))
                if (file.endsWith(".cpp") || file.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", file), join(samples, file));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(samples, "sample_geometry.cpp"), source = readFileSync(path, "utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0); expect(source.slice(offset)).toContain(before);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            run([sampleBinary(samples), scene, "600"], { GEOMETRY_RECORDS: records });
            expect(digests()).not.toBe(readFileSync(new URL(`${scene}.geometry.gold.txt`, base), "utf8"));
            console.log(`${scene} native parity, repeats, controls and mutation: ${(performance.now()-start).toFixed(0)} ms`);
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

for (const [scene, name, before, after] of [
    ["shapeCastDebug", "ShapeCastDebug", "72267.117187500", "70000.000000000"],
    ["distanceDebug", "DistanceDebug", "1.00989532471", "1.10989532471"],
    ["shapeDistance", "ShapeDistance", "{ 0.0f, 1.0f, 0.0f }", "{ 0.0f, 1.1f, 0.0f }"],
    ["timeOfImpact", "TimeOfImpact", "-4.06512070", "-4.16512070"],
    ["rayCurtain", "RayCurtain", "0.9f };", "0.95f };"],
    ["castWorld", "CastWorld", "RandomVec3Uniform( -20.0f, 20.0f )", "RandomVec3Uniform( -20.0f, 19.0f )"],
    ["meshScale", "MeshScale", "m_start = { -2.0f, 0.0f, 0.0f };", "m_start = { -2.1f, 0.0f, 0.0f };"],
    ["shapeCast", "ShapeCast", "0.9f };", "0.95f };"],
    ["overlapWorld", "OverlapWorld", "{ -6.0f, 3.0f + 2.0f * index, 0.0f }", "{ -6.1f, 3.0f + 2.0f * index, 0.0f }"],
    ["longRayCast", "LongRayCast", "b3Vec3_zero, 1.0f", "b3Vec3_zero, 1.1f"],
    ["initialOverlap", "InitialOverlap", "-2.1f, -0.8f, 0.95f", "-2.2f, -0.8f, 0.95f"],
    ["capsuleCastRay", "CapsuleCastRay", ".radius = 0.5f,", ".radius = 0.55f,"],
] as const) {
    test(`${scene}: native hashes and every query repeat, controls use DrawControls, and a source mutation fails parity`, () => {
        const start = performance.now(), binary = sampleBinary();
        const directory = mkdtempSync(join(tmpdir(), "box3d-collision-"));
        const records = join(directory, "queries.txt");
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const digests = () => readFileSync(records, "utf8").split("step\n").slice(0, -1).map(recordDigest).join("\n") + "\n";
        try {
            for (const controlled of [false, true]) {
                const sequence = fileURLToPath(new URL(`${scene}.controls.txt`, base));
                if (controlled && !existsSync(sequence)) continue;
                const stem = `${scene}${controlled ? ".controls" : ""}`;
                const args = [binary, scene, "600", ...(controlled ? [sequence] : [])];
                const hashes = run(args, { COLLISION_QUERIES: records }), queries = digests();
                equalSampleSteps(hashes, readFileSync(new URL(`${stem}.gold.txt`, base), "utf8"), stem);
                expect(queries).toBe(readFileSync(new URL(`${stem}.queries.gold.txt`, base), "utf8"));
                expect(queries.trimEnd().split("\n")).toHaveLength(600);
                equalSampleSteps(run(args, { COLLISION_QUERIES: records }), hashes, `${stem} repeat`);
                expect(digests()).toBe(queries);
            }
            const samples = join(directory, "samples"); mkdirSync(samples); mkdirSync(join(directory, "shared"));
            for (const file of readdirSync(join(process.env.BOX3D!, "samples")))
                if (file.endsWith(".cpp") || file.endsWith(".h")) copyFileSync(join(process.env.BOX3D!, "samples", file), join(samples, file));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(directory, "shared/benchmarks.c"));
            copyRobustnessSources(directory);
            const path = join(samples, "sample_collision.cpp"), source = readFileSync(path, "utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0); expect(source.slice(offset)).toContain(before);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            const controlled = scene === "castWorld", stem = `${scene}${controlled ? ".controls" : ""}`;
            const args = [sampleBinary(samples), scene, "600", ...(controlled ? [fileURLToPath(new URL(`${scene}.controls.txt`, base))] : [])];
            const hashes = run(args, { COLLISION_QUERIES: records });
            expect(hashes !== readFileSync(new URL(`${stem}.gold.txt`, base), "utf8") || digests() !== readFileSync(new URL(`${stem}.queries.gold.txt`, base), "utf8")).toBe(true);
            console.log(`${scene} native parity, repeats, controls and mutation: ${(performance.now()-start).toFixed(0)} ms`);
        } finally { rmSync(directory, { recursive: true, force: true }); }
    });
}

for (const [scene, name, before, after] of [
    ["ragdollBox", "RagdollOnBox", "m_jointHertz = 1.0f;", "m_jointHertz = 1.1f;"],
    ["ragdollMesh", "RagdollOnMesh", "m_jointHertz = 2.0f;", "m_jointHertz = 2.1f;"],
    ["ragdollPile", "RagdollPile", "float hertz = 0.5f;", "float hertz = 0.6f;"],
    ["ragdollIncline", "RagdollIncline", "float hertz = 2.0f;", "float hertz = 2.1f;"],
] as const) {
    test(`${scene}: native steps repeat, controls run DrawControls and a source mutation fails parity`, () => {
        const start = performance.now();
        const binary = sampleBinary();
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const hashes = run([binary, scene, "600"]);
        equalSampleSteps(hashes, readFileSync(new URL(`${scene}.gold.txt`,base),"utf8"),scene);
        equalSampleSteps(run([binary,scene,"600"]), hashes,`${scene} repeat`);
        if (scene === "ragdollBox" || scene === "ragdollMesh") {
            const args = [binary,scene,"600",fileURLToPath(new URL(`${scene}.controls.txt`,base))];
            const controlled = run(args);
            equalSampleSteps(controlled,readFileSync(new URL(`${scene}.controls.gold.txt`,base),"utf8"),`${scene} controls`);
            equalSampleSteps(run(args),controlled,`${scene} controls repeat`);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(),"box3d-ragdoll-mutation-"));
        try {
            mkdirSync(join(copy,"samples")); mkdirSync(join(copy,"shared"));
            copyFileSync(join(process.env.BOX3D!,"shared/benchmarks.c"),join(copy,"shared/benchmarks.c"));
            for (const file of ["sample_manifold.cpp", "sample_geometry.cpp", "sample_collision.cpp", "sample_ragdoll.cpp", "sample_mesh.cpp", "sample_compound.cpp", "sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!,"samples",file),join(copy,"samples",file));
            copyRobustnessSources(copy);
            const path = join(copy,"samples/sample_ragdoll.cpp"), source = readFileSync(path,"utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset)).toContain(before);
            writeFileSync(path,source.slice(0,offset)+source.slice(offset).replace(before,after));
            expect(sampleGold(run([sampleBinary(join(copy,"samples")),scene,"600"])).map(step=>step.hash)).not.toEqual(sampleGold(hashes).map(step=>step.hash));
            expect(sampleGold(hashes)).toHaveLength(600);
        } finally { rmSync(copy,{recursive:true,force:true}); }
        console.log(`${scene} native parity, repeats, controls and mutation: ${(performance.now()-start).toFixed(0)} ms`);
    });
}

test("gyroscopicPrecession readouts are the sample's own averaged rates", () => {
    const dir = mkdtempSync(join(tmpdir(), "precession-readouts-"));
    try {
        const path = join(dir, "rates.txt");
        run([sampleBinary(), "gyroscopicPrecession", "600"], { PRECESSION_READOUTS: path });
        expect(readFileSync(path, "utf8")).toBe(readFileSync(new URL("../../examples/physics/tests/gyroscopicPrecession.readouts.gold.txt", import.meta.url), "utf8"));
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test("grab sequence reproduces the example's native gold", () => {
    const hashes = run([sampleBinary(), "box_stack", "45", "grab"]);
    equalSampleSteps(run([sampleBinary(), "box_stack", "45", "grab"]), hashes, "grab repeat");
    equalSampleSteps(hashes, readFileSync(new URL("../../examples/physics/tests/grab.gold.txt", import.meta.url), "utf8"), "grab");
});

test("rigidBody: native hashes and character state repeat through held keys and reject a source mutation", () => {
    const start = performance.now();
    const binary = sampleBinary();
    const directory = mkdtempSync(join(tmpdir(), "box3d-rigid-body-"));
    const records = join(directory, "character.txt");
    const sequence = fileURLToPath(new URL("../../examples/physics/tests/rigidBody.controls.txt", import.meta.url));
    try {
        for (const controlled of [false, true]) {
            const stem = `rigidBody${controlled ? ".controls" : ""}`;
            const args = [binary, "rigidBody", "600", ...(controlled ? [sequence] : [])];
            const hashes = run(args, { RIGID_READOUTS: records });
            const states = readFileSync(records, "utf8");
            expect(run(args, { RIGID_READOUTS: records })).toBe(hashes);
            expect(readFileSync(records, "utf8")).toBe(states);
            expect(hashes).toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.gold.txt`, import.meta.url), "utf8"));
            expect(states).toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.character.gold.txt`, import.meta.url), "utf8"));
            expect(hashes.trimEnd().split("\n")).toHaveLength(600);
            expect(states.trimEnd().split("\n")).toHaveLength(600);
        }
        const copy = join(directory, "source"); mkdirSync(join(copy, "samples"), { recursive: true }); mkdirSync(join(copy, "shared"));
        copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
        for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"]) copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_continuous.cpp"), join(copy, "samples/sample_continuous.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
        copyRobustnessSources(copy);
        const path = join(copy, "samples/sample_character.cpp"); const source = readFileSync(path, "utf8");
        expect(source.includes("m_walkSpeed = 230.0f * SRC;")).toBe(true);
        writeFileSync(path, source.replace("m_walkSpeed = 230.0f * SRC;", "m_walkSpeed = 231.0f * SRC;"));
        const mutated = run([sampleBinary(join(copy, "samples")), "rigidBody", "600", sequence], { RIGID_READOUTS: records });
        expect(mutated).not.toBe(readFileSync(new URL("../../examples/physics/tests/rigidBody.controls.gold.txt", import.meta.url), "utf8"));
        expect(readFileSync(records, "utf8")).not.toBe(readFileSync(new URL("../../examples/physics/tests/rigidBody.controls.character.gold.txt", import.meta.url), "utf8"));
        console.log(`rigidBody native hashes, character state, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
    } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("geometricMover: native hashes and mover state repeat through keys and DrawControls, and a source mutation reds them", () => {
    const start = performance.now();
    const copy = mkdtempSync(join(tmpdir(), "box3d-geometric-mutation-"));
    const records = join(copy, "mover.txt");
    const sequence = fileURLToPath(new URL("../../examples/physics/tests/geometricMover.controls.txt", import.meta.url));
    try {
        const binary = sampleBinary();
        for (const controlled of [false, true]) {
            const stem = `geometricMover${controlled ? ".controls" : ""}`;
            const args = [binary, "geometricMover", "600", ...(controlled ? [sequence] : [])];
            const hashes = run(args, { MOVER_READOUTS: records });
            const state = readFileSync(records, "utf8");
            equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${stem}.gold.txt`, import.meta.url), "utf8"), stem);
            expect(state).toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.mover.gold.txt`, import.meta.url), "utf8"));
            equalSampleSteps(run(args, { MOVER_READOUTS: records }), hashes, `${stem} repeat`);
            expect(readFileSync(records, "utf8")).toBe(state);
        }
        mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
        copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
        for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
            copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_continuous.cpp"), join(copy, "samples/sample_continuous.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
        copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
        copyRobustnessSources(copy);
        const path = join(copy, "samples/mover.h");
        const source = readFileSync(path, "utf8");
        expect(source.includes("m_maxSpeed = 6.0f;")).toBe(true);
        writeFileSync(path, source.replace("m_maxSpeed = 6.0f;", "m_maxSpeed = 6.1f;"));
        const mutated = run([sampleBinary(join(copy, "samples")), "geometricMover", "600", sequence], { MOVER_READOUTS: records });
        expect(mutated).not.toBe(readFileSync(new URL("../../examples/physics/tests/geometricMover.controls.gold.txt", import.meta.url), "utf8"));
        expect(readFileSync(records, "utf8")).not.toBe(readFileSync(new URL("../../examples/physics/tests/geometricMover.controls.mover.gold.txt", import.meta.url), "utf8"));
        console.log(`geometricMover native hashes, mover state, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
    } finally { rmSync(copy, { recursive: true, force: true }); }
});

for (const scene of ["capsulePlane", "moverOverlap"]) {
    test(`${scene}: native hashes and drawing repeat, controls run the sample handlers, and a source mutation reds the drawing`, () => {
        const start = performance.now();
        const copy = mkdtempSync(join(tmpdir(), "box3d-character-mutation-"));
        const drawing = join(copy, "drawing.txt");
        const digests = () => readFileSync(drawing, "utf8").trimEnd().split("\n").map(line => drawingDigest(JSON.parse(line))).join("\n") + "\n";
        try {
            const binary = sampleBinary();
            for (const controlled of [false, true]) {
                const stem = `${scene}${controlled ? ".controls" : ""}`;
                const args = [binary, scene, "600", ...(controlled ? [fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url))] : [])];
                const hashes = run(args, { CHARACTER_DRAWING: drawing });
                const records = readFileSync(drawing, "utf8");
                equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${stem}.gold.txt`, import.meta.url), "utf8"), stem);
                expect(digests()).toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.drawing.gold.txt`, import.meta.url), "utf8"));
                equalSampleSteps(run(args, { CHARACTER_DRAWING: drawing }), hashes, `${stem} repeat`);
                expect(readFileSync(drawing, "utf8")).toBe(records);
            }
            mkdirSync(join(copy, "samples"));
            mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"]) copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_continuous.cpp"), join(copy, "samples/sample_continuous.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, "samples/sample_character.cpp");
            const source = readFileSync(path, "utf8");
            const before = scene === "capsulePlane" ? "0.25f };" : "0.35f };";
            const after = scene === "capsulePlane" ? "0.26f };" : "0.36f };";
            expect(source.includes(before)).toBe(true);
            writeFileSync(path, source.replace(before, after));
            run([sampleBinary(join(copy, "samples")), scene, "600"], { CHARACTER_DRAWING: drawing });
            expect(digests()).not.toBe(readFileSync(new URL(`../../examples/physics/tests/${scene}.drawing.gold.txt`, import.meta.url), "utf8"));
            console.log(`${scene} native parity, records, controls and mutation: ${(performance.now() - start).toFixed(0)} ms`);
        } finally { rmSync(copy, { recursive: true, force: true }); }
    });
}

const meshMutations: Record<string, [string,string,string]> = {
    voxel:["VoxelMesh","float scale = 0.01f;","float scale = 0.011f;"],
    hollowBox:["HollowBox","{ 10.0f, 10.0f, 10.0f }","{ 10.1f, 10.0f, 10.0f }"],
    gridMesh:["GridMesh","20, 20, 1.0f, 0, true","20, 20, 1.1f, 0, true"],
    bigBoxMesh:["BigBoxMesh","{ 50.0f, 1.0f, 50.0f }","{ 51.0f, 1.0f, 50.0f }"],
    boxMesh:["BoxMesh","{ 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }","{ 0.0f, 1.1f, 0.0f }, { 1.0f, 1.0f, 1.0f }"],
};
for (const [scene,[name,before,after]] of Object.entries(meshMutations)) {
    test(`${scene}: Mesh sample repeats its gold, exercises DrawControls and rejects a source mutation`,()=>{
        const start = performance.now();
        const copy = mkdtempSync(join(tmpdir(),"box3d-mesh-mutation-"));
        const records = join(copy,"mesh.txt");
        try {
            const binary = sampleBinary();
            const hashes = run([binary,scene,"600"],{MESH_GEOMETRY:records});
            const geometry = nativeMeshRecords(readFileSync(records,"utf8"));
            expect(geometry).toEqual(readFileSync(new URL(`../../examples/physics/tests/${scene}.mesh.gold.txt`,import.meta.url),"utf8").trimEnd().split("\n"));
            rmSync(records);
            equalSampleSteps(run([binary,scene,"600"],{MESH_GEOMETRY:records}),hashes,`${scene} repeat`);
            expect(nativeMeshRecords(readFileSync(records,"utf8"))).toEqual(geometry);
            equalSampleSteps(hashes,readFileSync(new URL(`../../examples/physics/tests/${scene}.gold.txt`,import.meta.url),"utf8"),scene);
            expect(sampleGold(hashes)).toHaveLength(600);
            if (!["voxel","hollowBox"].includes(scene)) {
                const args=[binary,scene,"600",fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`,import.meta.url))];
                const controlled=run(args);
                equalSampleSteps(controlled,readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`,import.meta.url),"utf8"),`${scene} controls`);
                equalSampleSteps(run(args),controlled,`${scene} controls repeat`);
                expect(controlled).not.toBe(hashes);
            }
            mkdirSync(join(copy,"samples"));mkdirSync(join(copy,"shared"));
            copyFileSync(join(process.env.BOX3D!,"shared/benchmarks.c"),join(copy,"shared/benchmarks.c"));
            for (const file of ["sample_mesh.cpp","sample_compound.cpp","sample_continuous.cpp","sample.cpp","sample_stacking.cpp","sample_joint.cpp","sample_events.cpp","sample_benchmark.cpp","sample_bodies.cpp","sample_shapes.cpp","sample_character.cpp","mover.h","mover.cpp"])
                copyFileSync(join(process.env.BOX3D!,"samples",file),join(copy,"samples",file));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path=join(copy,"samples/sample_mesh.cpp");const source=readFileSync(path,"utf8");
            const offset=source.indexOf(`class ${name} :`);expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset).includes(before)).toBe(true);
            writeFileSync(path,source.slice(0,offset)+source.slice(offset).replace(before,after));
            rmSync(records);
            run([sampleBinary(join(copy,"samples")),scene,"600"],{MESH_GEOMETRY:records});
            expect(nativeMeshRecords(readFileSync(records,"utf8"))).not.toEqual(geometry);
        } finally {rmSync(copy,{recursive:true,force:true});}
        console.log(`${scene} native parity, repeat, controls and mutation: ${(performance.now()-start).toFixed(0)} ms`);
    });
}

for (const [scene, name, before, after] of [
    ["meshReflection", "MeshReflection", "20, 20, 2.0f, 2, true", "20, 20, 2.1f, 2, true"],
    ["meshHeightField", "HeightField", "m_amplitude = 0.75f;", "m_amplitude = 0.8f;"],
]) {
    test(`${scene}: Mesh sample repeats hashes, geometry and drawings, drives controls and rejects a source mutation`, () => {
        const start = performance.now();
        const copy = mkdtempSync(join(tmpdir(), "box3d-mesh-frame-mutation-"));
        const frames = join(copy, "frames.txt"), drawing = join(copy, "drawing.txt");
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        try {
            const binary = sampleBinary();
            const observe = (binary: string, controlled: boolean) => {
                rmSync(frames, {force:true});
                const args = [binary, scene!, "600", ...(controlled ? [fileURLToPath(new URL(`${scene}.controls.txt`, base))] : [])];
                const hashes = run(args, {MESH_FRAMES:frames, CHARACTER_DRAWING:drawing});
                return {hashes, geometry:nativeMeshFrames(readFileSync(frames,"utf8")), drawings:readFileSync(drawing,"utf8").trim().split("\n").map(line=>drawingDigest(JSON.parse(line)))};
            };
            for (const controlled of [false,true]) {
                const stem = `${scene}${controlled?".controls":""}`;
                const a = observe(binary,controlled), b = observe(binary,controlled);
                expect(a).toEqual(b);
                equalSampleSteps(a.hashes,readFileSync(new URL(`${stem}.gold.txt`,base),"utf8"),stem);
                expect(a.geometry).toHaveLength(600);
                expect(a.geometry).toEqual(readFileSync(new URL(`${stem}.geometry.gold.txt`,base),"utf8").trim().split("\n"));
                if(scene==="meshHeightField") expect(a.drawings).toEqual(readFileSync(new URL(`${stem}.drawing.gold.txt`,base),"utf8").trim().split("\n"));
            }
            mkdirSync(join(copy,"samples"));mkdirSync(join(copy,"shared"));
            copyFileSync(join(process.env.BOX3D!,"shared/benchmarks.c"),join(copy,"shared/benchmarks.c"));
            for (const file of ["sample_mesh.cpp","sample_compound.cpp","sample_continuous.cpp","sample.cpp","sample_stacking.cpp","sample_joint.cpp","sample_events.cpp","sample_benchmark.cpp","sample_bodies.cpp","sample_shapes.cpp","sample_character.cpp","mover.h","mover.cpp"])
                copyFileSync(join(process.env.BOX3D!,"samples",file),join(copy,"samples",file));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path=join(copy,"samples/sample_mesh.cpp"),source=readFileSync(path,"utf8");
            const offset=source.indexOf(`class ${name} :`);expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset)).toContain(before!);
            writeFileSync(path,source.slice(0,offset)+source.slice(offset).replace(before!,after!));
            const changed=observe(sampleBinary(join(copy,"samples")),false);
            expect(changed.geometry).not.toEqual(readFileSync(new URL(`${scene}.geometry.gold.txt`,base),"utf8").trim().split("\n"));
        } finally { rmSync(copy,{recursive:true,force:true}); }
        console.log(`${scene} native hashes, geometry, drawings, controls, repeats and mutation: ${(performance.now()-start).toFixed(0)} ms`);
    });
}

for (const [scene, name, before, after] of [
    ["meshViewer", "MeshViewer", "m_meshIndex = 0;", "m_meshIndex = 1;"],
    ["meshCreation", "MeshCreationBenchmark", "int iterations = m_isDebug ? 1 : 10;", "int iterations = m_isDebug ? 1 : 9;"],
]) {
    test(`${scene}: native readouts and node bounds repeat, drive controls and reject a source mutation`, () => {
        const start = performance.now();
        const copy = mkdtempSync(join(tmpdir(), "box3d-mesh-inspector-"));
        const readouts = join(copy, "readouts.txt"), bounds = join(copy, "bounds.txt");
        const base = new URL("../../examples/physics/tests/", import.meta.url);
        const observe = (binary: string, controlled: boolean) => {
            const hashes = run([binary, scene!, "600", ...(controlled ? [fileURLToPath(new URL(`${scene}.controls.txt`, base))] : [])], {
                MESH_READOUTS: readouts, ...(scene === "meshViewer" ? { MESH_BOUNDS: bounds } : {}),
            });
            const digest = (path: string) => readFileSync(path, "utf8").trim().split("\n").map((line) => drawingDigest(JSON.parse(line)));
            return { hashes, readouts: digest(readouts), bounds: scene === "meshViewer" ? digest(bounds) : [] };
        };
        try {
            const binary = sampleBinary();
            for (const controlled of scene === "meshViewer" ? [false, true] : [false]) {
                const stem = `${scene}${controlled ? ".controls" : ""}`;
                const a = observe(binary, controlled), b = observe(binary, controlled);
                expect(a).toEqual(b);
                equalSampleSteps(a.hashes, readFileSync(new URL(`${stem}.gold.txt`, base), "utf8"), stem);
                expect(a.readouts).toHaveLength(600);
                expect(a.readouts).toEqual(readFileSync(new URL(`${stem}.readouts.gold.txt`, base), "utf8").trim().split("\n"));
                if (scene === "meshViewer") {
                    expect(a.bounds).toHaveLength(600);
                    expect(a.bounds).toEqual(readFileSync(new URL(`${stem}.bounds.gold.txt`, base), "utf8").trim().split("\n"));
                }
            }
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const file of ["sample_mesh.cpp", "sample_compound.cpp", "sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", file), join(copy, "samples", file));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, "samples/sample_mesh.cpp"), source = readFileSync(path, "utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset)).toContain(before!);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before!, after!));
            expect(observe(sampleBinary(join(copy, "samples")), false).readouts).not.toEqual(
                readFileSync(new URL(`${scene}.readouts.gold.txt`, base), "utf8").trim().split("\n"),
            );
        } finally { rmSync(copy, { recursive: true, force: true }); }
        console.log(`${scene} native hashes, readouts, node bounds, repeats and mutation: ${(performance.now() - start).toFixed(0)} ms`);
    });
}

const compoundMutations: Record<string, [string, string, string]> = {
    simpleCompound: ["SimpleCompound", "float a = 4.0f;", "float a = 4.1f;"],
    compoundSpheres: ["CompoundSpheres", "float h = 10.0f;", "float h = 10.1f;"],
    compoundHulls: ["CompoundHulls", "float h = 10.0f;", "float h = 10.1f;"],
    tileFloor: ["TileFloor", "float a = 4.0f;", "float a = 4.1f;"],
    meshTile: ["MeshTile", "float a = 4.0f;", "float a = 4.1f;"],
    village: ["Village", "float a = 4.0f;", "float a = 4.1f;"],
};
for (const [scene, [name, before, after]] of Object.entries(compoundMutations)) {
    test(`${scene}: Compound sample repeats its gold and rejects a source mutation`, () => {
        const start = performance.now();
        const binary = sampleBinary();
        const copy = mkdtempSync(join(tmpdir(), "box3d-compound-mutation-"));
        const records = join(copy, "compound.txt");
        try {
            const hashes = run([binary, scene, "600"], { COMPOUND_GEOMETRY: records });
            const geometry = nativeCompoundRecords(readFileSync(records, "utf8"));
            expect(geometry).toEqual(readFileSync(new URL(`../../examples/physics/tests/${scene}.compound.gold.txt`, import.meta.url), "utf8").trimEnd().split("\n"));
            rmSync(records);
            equalSampleSteps(run([binary, scene, "600"], { COMPOUND_GEOMETRY: records }), hashes, `${scene} repeat`);
            expect(nativeCompoundRecords(readFileSync(records, "utf8"))).toEqual(geometry);
            equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url), "utf8"), scene);
            expect(hashes.trimEnd().split("\n")).toHaveLength(600);
            if (scene === "village") {
                const drawing = join(copy, "drawing.txt"), mover = join(copy, "mover.txt");
                for (const mode of ["", ".controls", ".casts"]) {
                    const stem = `village${mode}`;
                    const args = [binary, scene, "600", ...(mode === ".controls" ? [fileURLToPath(new URL("../../examples/physics/tests/village.controls.txt", import.meta.url))] : [])];
                    const env = { CHARACTER_DRAWING: drawing, VILLAGE_READOUTS: mover, ...(mode === ".casts" ? { VILLAGE_CAST_ORIGIN: "1" } : {}) };
                    const result = run(args, env);
                    equalSampleSteps(result, readFileSync(new URL(`../../examples/physics/tests/${mode === ".controls" ? stem : "village"}.gold.txt`, import.meta.url), "utf8"), stem);
                    const drawings = readFileSync(drawing, "utf8");
                    const states = readFileSync(mover, "utf8");
                    expect(drawings.trimEnd().split("\n").map(line => {
                        const rows: number[][] = JSON.parse(line);
                        const first = rows.findIndex(r => r[0] === 1 && r[2] === 20 && r[5] === -20);
                        expect(first).toBeGreaterThanOrEqual(0);
                        return drawingDigest(rows.slice(first, -1));
                    }).join("\n") + "\n").toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.drawing.gold.txt`, import.meta.url), "utf8"));
                    if (mode !== ".casts") expect(states.trimEnd().split("\n").map(line => drawingDigest([JSON.parse(line)])).join("\n") + "\n").toBe(readFileSync(new URL(`../../examples/physics/tests/${stem}.mover.gold.txt`, import.meta.url), "utf8"));
                    expect(run(args, env)).toBe(result);
                    expect(readFileSync(drawing, "utf8")).toBe(drawings);
                    expect(readFileSync(mover, "utf8")).toBe(states);
                }
            }
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const file of ["sample_mesh.cpp", "sample_compound.cpp", "sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", file), join(copy, "samples", file));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, "samples/sample_compound.cpp");
            const source = readFileSync(path, "utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset).includes(before)).toBe(true);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            rmSync(records);
            const mutated = run([sampleBinary(join(copy, "samples")), scene, "600"], { COMPOUND_GEOMETRY: records });
            expect(nativeCompoundRecords(readFileSync(records, "utf8"))).not.toEqual(geometry);
            if (scene === "village") {
                const drawing = join(copy, "mutation-drawing.txt");
                run([sampleBinary(join(copy, "samples")), scene, "600"], { CHARACTER_DRAWING: drawing, VILLAGE_CAST_ORIGIN: "1" });
                const digests = readFileSync(drawing, "utf8").trimEnd().split("\n").map(line => {
                    const rows: number[][] = JSON.parse(line);
                    return drawingDigest(rows.slice(rows.findIndex(r => r[0] === 1 && r[2] === 20 && r[5] === -20), -1));
                }).join("\n") + "\n";
                expect(digests).not.toBe(readFileSync(new URL("../../examples/physics/tests/village.casts.drawing.gold.txt", import.meta.url), "utf8"));
            }
            if (scene === "simpleCompound" || scene === "tileFloor")
                expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
        } finally { rmSync(copy, { recursive: true, force: true }); }
        console.log(`${scene} native parity, repeat and mutation: ${(performance.now() - start).toFixed(0)} ms`);
    });
}

const continuousMutations: Record<string, [string, string, string]> = {
    bulletVersusStack: ["BulletVersusStack", "1.1f * row", "1.2f * row"],
    meshDrop: ["MeshDrop", "m_groundAmplitude = 0.5f;", "m_groundAmplitude = 0.6f;"],
    thinWall: ["ThinWall", "-180.0f", "-181.0f"],
    bounceHouse: ["BounceHouse", "120.0f", "121.0f"],
    spinningStick: ["SpinningStick", "-100.0f", "-101.0f"],
    needleMesh: ["NeedleMesh", "-10.0f", "-11.0f"],
    humpMesh: ["HumpMesh", "-50.0f", "-51.0f"],
    isFast: ["IsFast", "4.0f", "4.1f"],
    stall: ["Stall", "600.0f", "590.0f"],
};
for (const [scene, [name, before, after]] of Object.entries(continuousMutations)) {
    test(`${scene}: Continuous sample repeats its gold and rejects a source mutation`, () => {
        const start = performance.now();
        const binary = sampleBinary();
        const hashes = run([binary, scene, "600"]);
        equalSampleSteps(run([binary, scene, "600"]), hashes, `${scene} repeat`);
        equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url), "utf8"), scene);
        expect(hashes.trimEnd().split("\n")).toHaveLength(600);
        if (["stall", "bulletVersusStack", "meshDrop"].includes(scene)) {
            const sequence = fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url));
            const env: Record<string, string> = scene === "meshDrop" ? { SAMPLE_DISABLE_SLEEP: "1" } : {};
            const controlled = run([binary, scene, "600", sequence], env);
            equalSampleSteps(run([binary, scene, "600", sequence], env), controlled, `${scene} controls repeat`);
            equalSampleSteps(controlled, readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`, import.meta.url), "utf8"), `${scene} controls`);
            expect(controlled).not.toBe(hashes);
        }
        if (scene === "meshDrop") {
            const sequence = fileURLToPath(new URL("../../examples/physics/tests/meshDrop.regeneration.controls.txt", import.meta.url));
            const controlled = run([binary, scene, "600", sequence]);
            equalSampleSteps(run([binary, scene, "600", sequence]), controlled, "meshDrop regeneration repeat");
            equalSampleSteps(controlled, readFileSync(new URL("../../examples/physics/tests/meshDrop.regeneration.controls.gold.txt", import.meta.url), "utf8"), "meshDrop regeneration");
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-continuous-mutation-"));
        try {
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const file of ["sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", file), join(copy, "samples", file));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, "samples/sample_continuous.cpp");
            const source = readFileSync(path, "utf8");
            const offset = source.indexOf(`class ${name} :`);
            expect(offset).toBeGreaterThan(0);
            expect(source.slice(offset).includes(before)).toBe(true);
            writeFileSync(path, source.slice(0, offset) + source.slice(offset).replace(before, after));
            const mutated = run([sampleBinary(join(copy, "samples")), scene, "600"]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
        } finally { rmSync(copy, { recursive: true, force: true }); }
        console.log(`${scene} native parity, repeat and mutation: ${(performance.now() - start).toFixed(0)} ms`);
    });
}

const stackingMutations: Record<string, [string, string, string]> = {
    jenga: ["JengaStack", "float r = 0.1f;", "float r = 0.11f;"],
    cardHouse: ["CardHouse", "float cardHeight = 0.2f;", "float cardHeight = 0.21f;"],
    sphereStack: ["SphereStack", "float r = 0.5f;", "float r = 0.51f;"],
    capsuleStack: ["CapsuleStack", "float r = 0.5f;", "float r = 0.51f;"],
    singleBox: ["SingleBox", "b3MakeCubeHull( 0.5f )", "b3MakeCubeHull( 0.51f )"],
    cylinder: ["Cylinder", "b3CreateCylinder( 1.0f, 0.25f", "b3CreateCylinder( 1.0f, 0.26f"],
    cylinderStack: ["CylinderStack", "b3CreateCylinder( 1.0f, 0.5f", "b3CreateCylinder( 1.0f, 0.51f"],
    dominoes: ["Dominoes", "7.0f + 1.1f * ring", "7.1f + 1.1f * ring"],
    wedge: ["Wedge", "{ -0.5, 0.5f, 0.0f }", "{ -0.5, 0.6f, 0.0f }"],
    arch: ["Arch", "float scale = 0.25f;", "float scale = 0.26f;"],
    doubleDomino: ["DoubleDomino", "shapeDef.density = 4.0f;", "shapeDef.density = 4.1f;"],
    pyramid2D: ["Pyramid2D", "float a = 1.0f;", "float a = 1.01f;"],
    edgeCrossing: ["EdgeCrossing", "{ 0.2f, 0.02f, 0.04f }", "{ 0.21f, 0.02f, 0.04f }"],
};
const bodyMutations: Record<string, [string, string, string]> = {
    classRing: ["ClassRing", "constexpr float r = 1.0f;", "constexpr float r = 1.01f;"],
    cast: ["BodyCast", "b3CreateCylinder( 2.0f, 0.5f, 0.0f, 16 )", "b3CreateCylinder( 2.0f, 0.6f, 0.0f, 16 )"],
    bodyType: ["BodyType", "{ -2.0f, 3.0f, 0.0f }", "{ -2.1f, 3.0f, 0.0f }"],
    weeble: ["Weeble", "offset = { 0.0f, -1.5f, 0.0f }", "offset = { 0.0f, -1.4f, 0.0f }"],
    disable: ["DisableBody", "float linkRadius = 0.1f;", "float linkRadius = 0.11f;"],
    spinningBook: ["SpinningBooks", "b3MakeBoxHull( 0.35f, 0.08f, 0.5f )", "b3MakeBoxHull( 0.36f, 0.08f, 0.5f )"],
    gyroscopicPrecession: ["GyroscopicPrecession", "constexpr float r = 2.0f;", "constexpr float r = 2.1f;"],
    gyroscopicTorque: ["GyroscopicTorque", "0.6f, 0.15f, 0.0f, 32", "0.61f, 0.15f, 0.0f, 32"],
    kinematic: ["Kinematic", "m_amplitude = 2.0f;", "m_amplitude = 2.1f;"],
    lockMixing: ["LockMixing", "{ 0.0f, 2.0f, 0.0f }", "{ 0.0f, 2.1f, 0.0f }"],
    fixedRotation: ["FixedRotation", "capsule.radius = 0.2f;", "capsule.radius = 0.21f;"],
    offsetKinematic: ["OffsetKinematic", "{ 0.5f, 0.5f, 1.0f }", "{ 0.6f, 0.5f, 1.0f }"],
};
const jointMutations: Record<string, [string, string, string]> = {
    driving: ["Driving", "m_suspensionHertz = 4.0f;", "m_suspensionHertz = 4.1f;"],
    gearLift: ["GearLift", "m_motorTorque = 30000.0f;", "m_motorTorque = 30001.0f;"],
    door: ["Door", "bodyDef.gravityScale = 2.0f;", "bodyDef.gravityScale = 2.1f;"],
    distance: ["DistanceJoint", "float radius = 0.25f;", "float radius = 0.26f;"],
    spherical: ["SphericalJoint", "bodyDef.position = { 0.0f, 4.0f, 0.0f };", "bodyDef.position = { 0.0f, 4.1f, 0.0f };"],
    motionLocks: ["MotionLocks", "b3Pos position = { -12.5f, 10.0, 0.0f };", "b3Pos position = { -12.4f, 10.0, 0.0f };"],
    ballAndChain: ["BallAndChain", "float linkRadius = 0.125f;", "float linkRadius = 0.126f;"],
    foldOut: ["FoldOut", "float halfLength = 2.0f;", "float halfLength = 2.1f;"],
    topDownFriction: ["TopDownFriction", "int n = 10;", "int n = 9;"],
    weld: ["WeldJoint", "bodyDef.position = { 0.0f, 4.0f, 0.0f };", "bodyDef.position = { 0.0f, 4.1f, 0.0f };"],
};
const shapeMutations: Record<string, [string, string, string]> = {
    restitution: ["Restitution", "40.0f", "40.1f"],
    staticInvoke: ["StaticInvoke", "0.25f", "0.3f"],
    wind: ["Wind", "m_lift = 0.75f;", "m_lift = 0.8f;"],
    inclinedPlane: ["InclinedPlane", "bodyDef.position = { 0.0f, 7.5f, -5.0f };", "bodyDef.position = { 0.0f, 7.6f, -5.0f };"],
    rollingResistance: ["RollingResistance", "b3Sphere sphere = { b3Vec3_zero, 1.0f };", "b3Sphere sphere = { b3Vec3_zero, 1.1f };"],
    highResistance: ["HighResistance", "shapeDef.baseMaterial.rollingResistance = 0.2f * index;", "shapeDef.baseMaterial.rollingResistance = 0.21f * index;"],
    isotropicFriction: ["IsotropicFriction", "shapeDef.baseMaterial.friction = 0.6f;", "shapeDef.baseMaterial.friction = 0.61f;"],
    slideTwist: ["SlideTwist", "bodyDef.position = { 0.0f, 5.0f, 0.0f };", "bodyDef.position = { 0.0f, 5.1f, 0.0f };"],
    conveyorBelt: ["ConveyorBelt", "{ 2.0f, 0.0f, 0.0f };", "{ 2.1f, 0.0f, 0.0f };"],
    conveyorMesh: ["ConveyorMesh", "materials[i].friction = 0.8f;", "materials[i].friction = 0.81f;"],
    windDrop: ["WindDrop", "m_lift = 4.0f;", "m_lift = 4.1f;"],
    windFlap: ["WindFlap", "m_lift = 2.0f;", "m_lift = 2.1f;"],
};
const steps = 600;
for (const scene of ["box_stack", "bridge", "revolute", "prismatic", "wheel", "filter", "parallel", "motor", "joint", "sensor", "contact", "hit", "move", "persistentContact", "sensorHits", ...Object.keys(stackingMutations), ...Object.keys(jointMutations), ...Object.keys(bodyMutations), ...Object.keys(shapeMutations)]) {
    test(`${scene}: upstream sample records repeat and reject a source mutation`, () => {
        const binary = sampleBinary();
        const hashes = run([binary, scene, String(steps)]);
        equalSampleSteps(run([binary, scene, String(steps)]), hashes, `${scene} repeat`);
        equalSampleSteps(hashes, readFileSync(new URL(`../../examples/physics/tests/${scene === "box_stack" ? "boxStack" : scene}.gold.txt`, import.meta.url), "utf8"), scene);
        if (["restitution", "staticInvoke", "wind", "bodyType", "weeble", "disable", "revolute", "prismatic", "wheel", "parallel", "motor", "joint", "contact", "jenga", "weld", "topDownFriction", "distance", "spherical", "motionLocks", "door", "gearLift", "driving", "sensorHits"].includes(scene)) {
            const sequence = fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url));
            const controlledSteps = scene === "contact" ? 1204 : steps;
            const controlled = run([binary, scene, String(controlledSteps), sequence]);
            equalSampleSteps(run([binary, scene, String(controlledSteps), sequence]), controlled, `${scene} controls repeat`);
            equalSampleSteps(controlled, readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`, import.meta.url), "utf8"), `${scene} controls`);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-sample-mutation-"));
        try {
            if (scene === "sensorHits") {
                for (const controlled of [false, true]) {
                    const drawing = join(copy, "sensor.txt");
                    const args = [binary, scene, String(steps)];
                    if (controlled) args.push(fileURLToPath(new URL("../../examples/physics/tests/sensorHits.controls.txt", import.meta.url)));
                    run(args, { SENSOR_READOUTS: drawing });
                    expect(readFileSync(drawing, "utf8")).toBe(readFileSync(new URL(`../../examples/physics/tests/sensorHits${controlled ? ".controls" : ""}.drawing.gold.txt`, import.meta.url), "utf8"));
                }
            }
            if (scene === "persistentContact") {
                const drawing = join(copy, "drawing.txt");
                run([binary, scene, String(steps)], { CONTACT_DRAWING: drawing });
                expect(readFileSync(drawing, "utf8")).toBe(readFileSync(new URL("../../examples/physics/tests/persistentContact.drawing.gold.txt", import.meta.url), "utf8"));
            }
            mkdirSync(join(copy, "samples"));
            mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const name of ["sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"]) {
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            }
            copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, "samples", scene in shapeMutations ? "sample_shapes.cpp" : scene in bodyMutations ? "sample_bodies.cpp" : scene === "box_stack" || scene in stackingMutations ? "sample_stacking.cpp" : ["joint", "sensor", "contact", "hit", "move", "persistentContact", "sensorHits"].includes(scene) ? "sample_events.cpp" : "sample_joint.cpp");
            const source = readFileSync(path, "utf8");
            const classes: Record<string, string> = { box_stack: "BoxStack", bridge: "Bridge", revolute: "RevoluteJoint", prismatic: "PrismaticJoint", wheel: "WheelJoint", filter: "FilterJoint", parallel: "ParallelJoint", motor: "MotorJoint", joint: "JointEvent", sensor: "SensorVisit", contact: "ContactEvent", hit: "HitEvent", move: "MoveEvent", persistentContact: "PersistentContact", sensorHits: "SensorHits" };
            const start = source.indexOf(`class ${shapeMutations[scene]?.[0] ?? bodyMutations[scene]?.[0] ?? jointMutations[scene]?.[0] ?? stackingMutations[scene]?.[0] ?? classes[scene]} :`);
            const prefix = source.slice(0, start);
            const selected = source.slice(start);
            const [constant, replacement] = scene === "box_stack" ? ["float a = 0.5f;", "float a = 0.6f;"] : scene === "bridge" ? ["shapeDef.density = 20.0f;", "shapeDef.density = 21.0f;"] : ["bodyDef.position = { 0.0f, 4.0f, 0.0f };", "bodyDef.position = { 0.0f, 4.1f, 0.0f };"];
            const eventConstants: Record<string, [string, string]> = {
                motor: ["bodyDef.position = { -2.0, 2.0, 0.0 };", "bodyDef.position = { -2.1, 2.0, 0.0 };"],
                joint: ["b3Vec3 position = { -12.5f, 10.0, 0.0f };", "b3Vec3 position = { -12.4f, 10.0, 0.0f };"],
                sensor: ["bodyDef.position = { 0.0f, 12.5f, 0.0f };", "bodyDef.position = { 0.0f, 12.6f, 0.0f };"],
                contact: ["bodyDef.position = { 0.0f, 1.0f, 0.0f };", "bodyDef.position = { 0.0f, 1.1f, 0.0f };"],
                hit: ["float r = 0.75f;", "float r = 0.8f;"],
                move: ["b3Pos pivot = { 0.0f, 1.0f, 0.0f };", "b3Pos pivot = { 0.0f, 1.1f, 0.0f };"],
                persistentContact: ["shapeDef.density = 20.0f;", "shapeDef.density = 21.0f;"],
                sensorHits: ["RandomFloatRange( 200.0f, 300.0f )", "RandomFloatRange( 201.0f, 300.0f )"],
            };
            const before = shapeMutations[scene]?.[1] ?? bodyMutations[scene]?.[1] ?? jointMutations[scene]?.[1] ?? stackingMutations[scene]?.[1] ?? eventConstants[scene]?.[0] ?? (scene === "wheel" ? "bodyDef.position = { 0.0f, 2.0f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.0f, 4.0f, 0.0f };" : constant!);
            const after = shapeMutations[scene]?.[2] ?? bodyMutations[scene]?.[2] ?? jointMutations[scene]?.[2] ?? stackingMutations[scene]?.[2] ?? eventConstants[scene]?.[1] ?? (scene === "wheel" ? "bodyDef.position = { 0.0f, 2.1f, 0.0f };" : scene === "filter" ? "bodyDef.position = { 2.1f, 4.0f, 0.0f };" : replacement!);
            expect(selected.includes(before)).toBe(true);
            writeFileSync(path, prefix + selected.replace(before, after));
            const queryPath = join(copy, "queries.txt");
            if (scene === "cast") {
                run([binary, scene, String(steps)], { BODY_QUERIES: queryPath });
                const queries = readFileSync(queryPath, "utf8");
                expect(querySteps("cast", queries).map(recordDigest).join("\n") + "\n").toBe(readFileSync(new URL("../../examples/physics/tests/cast.queries.gold.txt", import.meta.url), "utf8"));
                run([binary, scene, String(steps)], { BODY_QUERIES: queryPath });
                expect(readFileSync(queryPath, "utf8")).toBe(queries);
            }
            const mutated = run([sampleBinary(join(copy, "samples")), scene, String(steps)], scene === "cast" ? { BODY_QUERIES: queryPath } : {});
            if (scene === "cast") expect(querySteps("cast", readFileSync(queryPath, "utf8")).map(recordDigest).join("\n") + "\n").not.toBe(readFileSync(new URL("../../examples/physics/tests/cast.queries.gold.txt", import.meta.url), "utf8"));
            if (scene === "cast") equalSampleSteps(mutated, hashes, "cast unstepped mutation");
            else expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
            expect(hashes.trim().split("\n")).toHaveLength(steps);
            console.log(`${scene}: ${hashes.split("\n")[0]} … ${hashes.trim().split("\n").at(-1)}`);
            console.log(`${scene} mutation: ${mutated.split("\n")[0]} … ${mutated.trim().split("\n").at(-1)}`);
        } finally {
            rmSync(copy, { recursive: true, force: true });
        }
    });
}

const benchmarkMutations: Record<string, [string, string]> = {
    largeWorld: ["#define STATIC_FLOOR_DROP_INTERVAL 5", "#define STATIC_FLOOR_DROP_INTERVAL 6"],
    benchmarkSensor: ["float gridSize = 3.0f;", "float gridSize = 3.1f;"],
    chains: ["float linkRadius = 0.125f;", "float linkRadius = 0.126f;"],
    explosion: ["constexpr int n = m_isDebug ? 3 : 16;", "constexpr int n = m_isDebug ? 3 : 15;"],
    destruction: ["m_explosionDef.impulsePerArea = m_small ? 200.0f : 1000.0f;", "m_explosionDef.impulsePerArea = m_small ? 200.0f : 1001.0f;"],
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
        const steps = scene === "benchmarkSensor" ? 3000 : 600;
        const hashes = run([sampleBinary(), scene, String(steps)]);
        equalSampleSteps(run([sampleBinary(), scene, String(steps)]), hashes, `${scene} repeat`);
        const goldText = readFileSync(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url), "utf8");
        if (scene === "benchmarkSensor") {
            const expected = goldText.trimEnd().split("\n");
            const actual = eventDigests(hashes);
            expect(actual.length).toBe(expected.length);
            for (let step = 0; step < expected.length; step++) expect(actual[step], digestMismatch(scene, step)).toBe(expected[step]);
        } else equalSampleSteps(hashes, goldText, scene);
        if (scene === "fallingTrees" || scene === "explosion") {
            const sequence = fileURLToPath(new URL(`../../examples/physics/tests/${scene}.controls.txt`, import.meta.url));
            const controlled = run([sampleBinary(), scene, String(steps), sequence]);
            equalSampleSteps(run([sampleBinary(), scene, String(steps), sequence]), controlled, `${scene} controls repeat`);
            equalSampleSteps(controlled, readFileSync(new URL(`../../examples/physics/tests/${scene}.controls.gold.txt`, import.meta.url), "utf8"), `${scene} controls`);
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-benchmark-mutation-"));
        try {
            mkdirSync(join(copy, "samples"));
            mkdirSync(join(copy, "shared"));
            for (const name of ["sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", name), join(copy, "samples", name));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_continuous.cpp"), join(copy, "samples/sample_continuous.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_compound.cpp"), join(copy, "samples/sample_compound.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_mesh.cpp"), join(copy, "samples/sample_mesh.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_ragdoll.cpp"), join(copy, "samples/sample_ragdoll.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_collision.cpp"), join(copy, "samples/sample_collision.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_geometry.cpp"), join(copy, "samples/sample_geometry.cpp"));
            copyFileSync(join(process.env.BOX3D!, "samples/sample_manifold.cpp"), join(copy, "samples/sample_manifold.cpp"));
            copyRobustnessSources(copy);
            const path = join(copy, ["benchmarkSensor", "chains", "fallingBoxes", "candyCups", "explosion", "destruction"].includes(scene) ? "samples/sample_benchmark.cpp" : "shared/benchmarks.c");
            const source = readFileSync(path, "utf8");
            const start = scene === "fallingBoxes" ? source.indexOf("class FallingBoxes :") : scene === "candyCups" ? source.indexOf("class CandyCups :") : 0;
            expect(source.slice(start).includes(before)).toBe(true);
            writeFileSync(path, source.slice(0, start) + source.slice(start).replace(before, after));
            const mutated = run([sampleBinary(join(copy, "samples")), scene, String(steps)]);
            expect(sampleGold(mutated).map(step => step.hash)).not.toEqual(sampleGold(hashes).map(step => step.hash));
        } finally { rmSync(copy, { recursive: true, force: true }); }
    });
}
