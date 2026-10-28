import { expect, setDefaultTimeout, test } from "bun:test";
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { run } from "./native";
import { sampleBinary } from "./sample";
import { sampleGold } from "../../examples/physics/tests/gold";
import { digestMismatch, drawingDigest, eventDigests, querySteps, recordDigest } from "../../examples/physics/tests/digest";

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

const continuousMutations: Record<string, [string, string, string]> = {
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
        if (scene === "stall") {
            const sequence = fileURLToPath(new URL("../../examples/physics/tests/stall.controls.txt", import.meta.url));
            const controlled = run([binary, scene, "600", sequence]);
            equalSampleSteps(run([binary, scene, "600", sequence]), controlled, "stall controls repeat");
            equalSampleSteps(controlled, readFileSync(new URL("../../examples/physics/tests/stall.controls.gold.txt", import.meta.url), "utf8"), "stall controls");
            expect(controlled).not.toBe(hashes);
        }
        const copy = mkdtempSync(join(tmpdir(), "box3d-continuous-mutation-"));
        try {
            mkdirSync(join(copy, "samples")); mkdirSync(join(copy, "shared"));
            copyFileSync(join(process.env.BOX3D!, "shared/benchmarks.c"), join(copy, "shared/benchmarks.c"));
            for (const file of ["sample_continuous.cpp", "sample.cpp", "sample_stacking.cpp", "sample_joint.cpp", "sample_events.cpp", "sample_benchmark.cpp", "sample_bodies.cpp", "sample_shapes.cpp", "sample_character.cpp", "mover.h", "mover.cpp"])
                copyFileSync(join(process.env.BOX3D!, "samples", file), join(copy, "samples", file));
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
