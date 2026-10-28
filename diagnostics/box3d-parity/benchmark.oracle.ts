import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import * as physics from "@dylanebert/shallot/standard/physics";
import { MeshSamplePlugin } from "../../examples/physics/src/mesh";
import { Body, BodyType, createApp, hashPhysics, Shape, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { BenchmarkSamplePlugin, benchmarkScenes } from "../../examples/physics/src/benchmarks";
import { sceneSwitcher } from "../../examples/physics/src/sample";
import { changeControl, controlValues, createScene, scenes } from "../../examples/physics/src/scenes";
import { sampleGold } from "../../examples/physics/tests/gold";
import { setLinearVelocity } from "@dylanebert/shallot/standard/physics";
import { digestMismatch, drawingDigest, eventRecord, recordDigest } from "../../examples/physics/tests/digest";
import { sensorEvents } from "./records";
import "./village.oracle";
import { VillagePlugin } from "../../examples/physics/src/village";
import { manualParityScenes } from "../../examples/physics/tests/manual-parity";
import { expectSwitchesEqualFresh } from "../../examples/physics/tests/switch";

// Run by path with sample.oracle.ts. Scene parity needs one solver thread; the engine's
// small determinism scenes hold scheduling and cross-host equality.
setDefaultTimeout(0);

// Large World's million-body switch is held by large-world-construction.oracle.ts.
for (const scene of [...manualParityScenes].filter(scene => scene !== "largeWorld")) {
    test(`${scene} switches equal to a fresh scene through 600 steps`, () =>
        expectSwitchesEqualFresh([scene], [BenchmarkSamplePlugin, VillagePlugin, MeshSamplePlugin]));
}

test("meshCreation triangle count and build work equal every recorded native step", async () => {
    const gold = sampleGold(await Bun.file(new URL("../../examples/physics/tests/meshCreation.gold.txt", import.meta.url)).text());
    const readouts = (await Bun.file(new URL("../../examples/physics/tests/meshCreation.readouts.gold.txt", import.meta.url)).text()).trim().split("\n");
    expect(gold).toHaveLength(600);
    expect(readouts).toHaveLength(gold.length);
    const start = performance.now();
    const app = await createApp({ defaults: false, plugins: [MeshSamplePlugin], setup(world) { createScene(world, "meshCreation"); } });
    const builds = spyOn(physics, "createPhysicsMesh");
    const releases = spyOn(physics, "destroyPhysicsMesh");
    try {
        expect(app.world.resource(physics.PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
        for (const [step, expected] of gold.entries()) {
            app.world.tick();
            expect(hashPhysics(app.world), `step ${step} hash`).toBe(expected.hash);
            const count = scenes.meshCreation.controls["triangle count"]!.read!(app.world)!;
            expect(drawingDigest([[count, builds.mock.calls.length]]), `readouts step ${step}`).toBe(readouts[step]!);
            expect(releases.mock.calls).toHaveLength(builds.mock.calls.length);
            builds.mockClear();
            releases.mockClear();
        }
    } finally {
        builds.mockRestore();
        releases.mockRestore();
        await app.dispose();
    }
    console.log(`meshCreation: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; 1 solver thread`);
});

test("explosion magnitude and button sequence equals every recorded native step", async () => {
    const gold = sampleGold(await Bun.file(new URL("../../examples/physics/tests/explosion.controls.gold.txt", import.meta.url)).text());
    const sequence = (await Bun.file(new URL("../../examples/physics/tests/explosion.controls.txt", import.meta.url)).text()).trim().split("\n").map(line => {
        const [step, name, value] = line.split("\t");
        return { step: Number(step), name: name!, value: Number(value) };
    });
    const start = performance.now();
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin] });
    try {
        const bodies = sceneSwitcher(app.world)("explosion");
        const values = controlValues("explosion");
        for (const [step, expected] of gold.entries()) {
            for (const change of sequence.filter(change => change.step === step)) changeControl(app.world, "explosion", bodies, values, change.name, change.value);
            app.world.tick();
            expect(hashPhysics(app.world), `controlled step ${step} hash`).toBe(expected.hash);
        }
    } finally { app.dispose(); }
    console.log(`explosion controls: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; 1 solver thread`);
});

test("fallingTrees grid-size sequence equals every recorded native step", async () => {
    const gold = sampleGold(await Bun.file(new URL("../../examples/physics/tests/fallingTrees.controls.gold.txt", import.meta.url)).text());
    const sequence = (await Bun.file(new URL("../../examples/physics/tests/fallingTrees.controls.txt", import.meta.url)).text()).trim().split("\n").map(line => {
        const [step, name] = line.split("\t");
        return { step: Number(step), size: Number.parseInt(name!, 10) };
    });
    expect(sequence.map(change => change.size).sort()).toEqual([100, 25, 50]);
    expect(gold).toHaveLength(600);
    const start = performance.now();
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin] });
    try {
        const bodies = sceneSwitcher(app.world)("fallingTrees");
        const values = controlValues("fallingTrees");
        for (const [step, expected] of gold.entries()) {
            for (const change of sequence.filter(change => change.step === step)) changeControl(app.world, "fallingTrees", bodies, values, "Grid size", change.size);
            app.world.tick();
            expect(hashPhysics(app.world), `controlled step ${step} hash`).toBe(expected.hash);
        }
    } finally { app.dispose(); }
    console.log(`fallingTrees controls: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; 1 solver thread`);
});

for (const scene of (Object.keys(benchmarkScenes) as (keyof typeof benchmarkScenes)[]).filter(scene => scene !== "heightField" && scene !== "hull")) {
    test(`${scene} equals every recorded native step and rejects a ${scene === "benchmarkSensor" ? "velocity" : "density"} change`, async () => {
        const text = await Bun.file(new URL(`../../examples/physics/tests/${scene}.gold.txt`, import.meta.url)).text();
        const gold = scene === "benchmarkSensor" ? text.trimEnd().split("\n").map(digest => ({ hash: BigInt(`0x${digest}`), events: [] })) : sampleGold(text);
        expect(gold).toHaveLength(scene === "benchmarkSensor" ? 3000 : 600);
        const start = performance.now();
        for (const mutated of [false, true]) {
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, BenchmarkSamplePlugin],
                setup(world) { createScene(world, scene); },
            });
            try {
                let diverged = false;
                let changed = false;
                for (const [step, expected] of gold.entries()) {
                    app.world.tick();
                    const hash = hashPhysics(app.world);
                    if (mutated && !changed && (scene !== "benchmarkSensor" || step >= 32)) {
                        const dynamic = [...app.world.query([Shape])].find(eid => app.world.storage(Body).type.get(app.world.storage(Shape).body.get(eid) || eid) === BodyType.Dynamic);
                        if (dynamic) {
                            if (scene === "benchmarkSensor") setLinearVelocity(app.world, dynamic, { x: 1, y: -5, z: 0 });
                            else app.world.storage(Shape).density.set(dynamic, 101);
                            changed = true;
                        }
                    }
                    if (!mutated) {
                        if (scene === "benchmarkSensor") {
                            expect(recordDigest(eventRecord(hash, sensorEvents(app.world))), digestMismatch(scene, step)).toBe(expected.hash.toString(16).padStart(16, "0"));
                        } else expect(hash, `step ${step} hash`).toBe(expected.hash);
                    }
                    else if ((scene === "benchmarkSensor" ? BigInt(`0x${recordDigest(eventRecord(hash, sensorEvents(app.world)))}`) : hash) !== expected.hash) { diverged = true; break; }
                }
                if (mutated) expect(diverged).toBe(true);
            } finally { app.dispose(); }
        }
        console.log(`${scene}: ${(performance.now() - start).toFixed(0)} ms; ${gold.length} recorded steps; ${text.length} gold bytes; 1 solver thread`);
    });
}
