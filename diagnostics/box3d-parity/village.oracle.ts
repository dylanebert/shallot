import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics } from "@dylanebert/shallot";
import { createScene } from "../../examples/physics/src/scenes";
import { VillagePlugin, VillageState } from "../../examples/physics/src/village";
import { drawingDigest } from "../../examples/physics/tests/digest";
import { sampleGold } from "../../examples/physics/tests/gold";
import { physicsState } from "../../src/standard/physics/runtime";
import type { CompoundData } from "../../src/standard/physics/shapes/compound";
import { compoundRecord } from "./compound-records";

setDefaultTimeout(0);
for (const mode of ["", ".controls", ".casts"]) {
    const controlled = mode === ".controls";
    test(`Village${mode} equals 600 native hashes, mover states and cast drawings`, async () => {
        const stem = `village${controlled ? ".controls" : ""}`;
        const gold = sampleGold(await Bun.file(new URL(`../../examples/physics/tests/${stem}.gold.txt`, import.meta.url)).text());
        const states = (await Bun.file(new URL(`../../examples/physics/tests/${stem}.mover.gold.txt`, import.meta.url)).text()).trimEnd().split("\n");
        const drawings = (await Bun.file(new URL(`../../examples/physics/tests/village${mode}.drawing.gold.txt`, import.meta.url)).text()).trimEnd().split("\n");
        const sequence = controlled ? (await Bun.file(new URL("../../examples/physics/tests/village.controls.txt", import.meta.url)).text()).trimEnd().split("\n").map(row => {
            const [step, name, value] = row.split("\t"); return { step: Number(step), name: name!, value: Number(value) };
        }) : [];
        const start = performance.now();
        const app = await createApp({ defaults: false, plugins: [VillagePlugin] });
        try {
            createScene(app.world, "village");
            const s = app.world.resource(VillageState);
            if (mode === ".casts") s.rayOrigin = { x: 0, y: 20, z: 0 };
            let hits = 0;
            expect(gold).toHaveLength(600); expect(states).toHaveLength(600); expect(drawings).toHaveLength(600);
            for (let step = 0; step < 600; step++) {
                for (const change of sequence) if (change.step === step) s.keys[change.name as keyof typeof s.keys] = change.value;
                app.world.tick();
                expect(hashPhysics(app.world), `${stem} step ${step} hash`).toBe(gold[step]!.hash);
                const m = s.mover;
                expect(drawingDigest([[m.position.x,m.position.y,m.position.z,m.velocity.x,m.velocity.y,m.velocity.z,m.pogoVelocity,+m.onGround,+m.sprint,m.planes.length,m.totalIterations]]), `${stem} step ${step} mover`).toBe(states[step]!);
                if (step === 0) {
                    const compounds = [...physicsState(app.world)!.geometryIdentityValues.values()].filter((v): v is CompoundData => "hulls" in v && "spheres" in v);
                    const expected = (await Bun.file(new URL("../../examples/physics/tests/village.compound.gold.txt", import.meta.url)).text()).trimEnd();
                    expect(compounds.map(compoundRecord)).toEqual([expected]);
                }
                expect(drawingDigest(s.drawing), `village${mode} step ${step} drawing`).toBe(drawings[step]!);
                if (s.drawing.some(row => row[0] === 4)) hits++;
            }
            if (mode === ".casts") expect(hits).toBe(600);
        } finally { app.dispose(); }
        console.log(`village${mode} 600-step hashes, mover and casts: ${(performance.now() - start).toFixed(0)} ms`);
    });
}
