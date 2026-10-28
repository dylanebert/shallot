import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { CharacterSample, CharacterSamplePlugin, characterDrag } from "../src/character";
import { characterDrawing } from "../src/character-presentation";
import { GeometricMoverPlugin, GeometricMoverState } from "../src/geometric-mover";
import { RigidBodyPlugin, RigidBodyState } from "../src/rigid-body";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { drawingDigest } from "./digest";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const controlled of [false, true])
    test(`rigidBody equals all 600 native hashes and character states${controlled ? " through held keys" : ""}`, async () => {
        const stem = `rigidBody${controlled ? ".controls" : ""}`;
        const gold = sampleGold(
            await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
        );
        const states = (
            await Bun.file(new URL(`./${stem}.character.gold.txt`, import.meta.url)).text()
        )
            .trimEnd()
            .split("\n")
            .map((line) => (JSON.parse(line) as number[]).map(Math.fround));
        const sequence = controlled
            ? (await Bun.file(new URL("./rigidBody.controls.txt", import.meta.url)).text())
                  .trimEnd()
                  .split("\n")
                  .map((row) => {
                      const [step, name, value] = row.split("\t");
                      return { step: Number(step), name: name!, value: Number(value) };
                  })
            : [];
        const app = await createApp({ defaults: false, plugins: [RigidBodyPlugin] });
        const start = performance.now();
        try {
            createScene(app.world, "rigidBody");
            const state = app.world.resource(RigidBodyState),
                s = state.character;
            expect(gold).toHaveLength(600);
            expect(states).toHaveLength(600);
            for (let i = 0; i < 600; i++) {
                for (const change of sequence)
                    if (change.step === i)
                        state.keys[change.name as keyof typeof state.keys] = change.value;
                app.world.tick();
                expect(hashPhysics(app.world), `${stem} step ${i} hash`).toBe(gold[i]!.hash);
                expect(
                    [
                        s.position.x,
                        s.position.y,
                        s.position.z,
                        s.velocity.x,
                        s.velocity.y,
                        s.velocity.z,
                        +s.onGround,
                        +s.sprint,
                        s.jumpCooldown,
                        s.groundNormal.x,
                        s.groundNormal.y,
                        s.groundNormal.z,
                        s.groundVelocity.x,
                        s.groundVelocity.y,
                        s.groundVelocity.z,
                        +s.didStep,
                        s.stepPosition.x,
                        s.stepPosition.y,
                        s.stepPosition.z,
                        s.wishVelocity.x,
                        s.wishVelocity.y,
                        s.wishVelocity.z,
                    ].map(Math.fround),
                    `${stem} step ${i} character`,
                ).toEqual(states[i]);
            }
        } finally {
            app.dispose();
        }
        console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
    });
for (const controlled of [false, true])
    test(`geometricMover equals all 600 native hashes and mover states${controlled ? " through keys and Clip Velocity" : ""}`, async () => {
        const stem = `geometricMover${controlled ? ".controls" : ""}`;
        const gold = sampleGold(
            await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
        );
        const states = (await Bun.file(new URL(`./${stem}.mover.gold.txt`, import.meta.url)).text())
            .trimEnd()
            .split("\n")
            .map((line) => (JSON.parse(line) as number[]).map(Math.fround));
        const sequence = controlled
            ? (await Bun.file(new URL("./geometricMover.controls.txt", import.meta.url)).text())
                  .trimEnd()
                  .split("\n")
                  .map((row) => {
                      const [step, name, value] = row.split("\t");
                      return { step: Number(step), name: name!, value: Number(value) };
                  })
            : [];
        const app = await createApp({ defaults: false, plugins: [GeometricMoverPlugin] });
        const start = performance.now();
        try {
            const bodies = createScene(app.world, "geometricMover");
            const values = controlValues("geometricMover");
            const s = app.world.resource(GeometricMoverState);
            expect(gold).toHaveLength(600);
            expect(states).toHaveLength(600);
            for (let i = 0; i < 600; i++) {
                for (const change of sequence)
                    if (change.step === i) {
                        if (Object.hasOwn(s.keys, change.name))
                            s.keys[change.name as keyof typeof s.keys] = change.value;
                        else
                            changeControl(
                                app.world,
                                "geometricMover",
                                bodies,
                                values,
                                change.name,
                                change.value,
                            );
                    }
                app.world.tick();
                expect(hashPhysics(app.world), `${stem} step ${i} hash`).toBe(gold[i]!.hash);
                const m = s.mover;
                expect(
                    [
                        m.position.x,
                        m.position.y,
                        m.position.z,
                        m.velocity.x,
                        m.velocity.y,
                        m.velocity.z,
                        m.pogoVelocity,
                        +m.onGround,
                        +m.sprint,
                        m.planes.length,
                        m.totalIterations,
                        s.treeTimer,
                        s.respawnTimer,
                        ...[s.boxImpact, s.treeImpact].flatMap((impact) => [
                            impact.point.x,
                            impact.point.y,
                            impact.point.z,
                            impact.normal.x,
                            impact.normal.y,
                            impact.normal.z,
                            impact.speed,
                            +impact.valid,
                        ]),
                        s.impactTransform.p.x,
                        s.impactTransform.p.y,
                        s.impactTransform.p.z,
                        s.impactTransform.q.v.x,
                        s.impactTransform.q.v.y,
                        s.impactTransform.q.v.z,
                        s.impactTransform.q.s,
                    ].map(Math.fround),
                    `${stem} step ${i} mover`,
                ).toEqual(states[i]);
            }
        } finally {
            app.dispose();
        }
        console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
    });
for (const scene of ["capsulePlane", "moverOverlap"] as const)
    for (const controlled of [false, true]) {
        test(`${scene} equals all 600 native hashes and drawing records${controlled ? " through dragging and controls" : ""}`, async () => {
            const start = performance.now();
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const gold = sampleGold(
                await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
            );
            const drawing = (
                await Bun.file(new URL(`./${stem}.drawing.gold.txt`, import.meta.url)).text()
            )
                .trimEnd()
                .split("\n");
            const sequence = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trimEnd()
                      .split("\n")
                      .map((line) => {
                          const [step, name, value] = line.split("\t");
                          return {
                              step: Number(step),
                              name: name!,
                              value: Math.fround(Number(value)),
                          };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [StandardPhysicsPlugin, CharacterSamplePlugin],
            });
            try {
                const bodies = createScene(app.world, scene);
                const values = controlValues(scene);
                const origin = { x: 0, y: 0, z: 0 },
                    translation = { x: 0, y: 0, z: -20 };
                let held = false;
                expect(gold).toHaveLength(600);
                expect(drawing).toHaveLength(600);
                for (let step = 0; step < 600; step++) {
                    for (const change of sequence)
                        if (change.step === step) {
                            if (change.name === "dragX") origin.x = change.value;
                            else if (change.name === "dragY") origin.y = change.value;
                            else if (change.name === "dragZ") origin.z = change.value;
                            else if (change.name === "drag") {
                                held = change.value !== 0;
                                characterDrag(app.world, origin, translation, held, held);
                            }
                        }
                    characterDrag(app.world, origin, translation, false, held);
                    for (const change of sequence)
                        if (change.step === step && change.name === "Solve")
                            changeControl(app.world, scene, bodies, values, "Solve", change.value);
                    app.world.tick();
                    expect(hashPhysics(app.world), `${stem} step ${step} hash`).toBe(
                        gold[step]!.hash,
                    );
                    const records = characterDrawing(app.world).records;
                    expect(drawingDigest(records), `${stem} step ${step} drawing`).toBe(
                        drawing[step]!,
                    );
                    expect(app.world.resource(CharacterSample).zeroNormalCount).toBe(0);
                }
            } finally {
                app.dispose();
            }
            console.log(`${stem}: ${(performance.now() - start).toFixed(0)} ms`);
        });
    }
