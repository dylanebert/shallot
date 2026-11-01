import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, type EntityRef, hashPhysics, Shape } from "@dylanebert/shallot";
import { PhysicsEvents, StepPhysicsSystem } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DynamicMoverPlugin, DynamicMoverState } from "../src/dynamic-mover-scene";
import { changeControl, controlValues, createScene } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const controlled of [false, true])
    test(`dynamicMover equals 600 native hashes${controlled ? " through recorded input and controls" : ""}`, async () => {
        const stem = `dynamicMover${controlled ? ".controls" : ""}`;
        const gold = sampleGold(
            await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
        );
        const states = (await Bun.file(new URL(`./${stem}.mover.gold.txt`, import.meta.url)).text())
            .trim()
            .split("\n")
            .map((row) => (JSON.parse(row) as number[]).map(Math.fround));
        if (controlled) {
            expect(states.some((row) => row[1] === 0 && row[4] === 1)).toBe(true);
            expect(states.some((row) => row[10]! < 17)).toBe(true);
            expect(states.at(-1)![11]).toBe(4);
        }
        const sequence = controlled
            ? (await Bun.file(new URL("./dynamicMover.controls.txt", import.meta.url)).text())
                  .trim()
                  .split("\n")
                  .map((row) => {
                      const [step, name, value] = row.split("\t");
                      return { step: Number(step), name: name!, value: Number(value) };
                  })
            : [];
        const app = await createApp({ defaults: false, plugins: [DynamicMoverPlugin] });
        let steppedHash = 0n;
        app.world.addSystem({
            name: "observe-dynamic-mover-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            before: DynamicMoverPlugin.systems!.filter(
                (system) => system.name === "dynamic-mover-after-step",
            ),
            update(world) {
                steppedHash = hashPhysics(world);
            },
        });
        const shapeIds = new Map<EntityRef, number>(),
            bodyIds = new Map<EntityRef, number>();
        for (const [kind, ids] of [
            [Shape, shapeIds],
            [Body, bodyIds],
        ] as const) {
            let next = 1;
            app.world.observeMembership(kind, (eid, present) => {
                if (present) ids.set(app.world.ref(eid), next++);
            });
        }
        try {
            const bodies = createScene(app.world, "dynamicMover"),
                values = controlValues("dynamicMover"),
                s = app.world.resource(DynamicMoverState);
            expect(gold).toHaveLength(600);
            for (let step = 0; step < 600; step++) {
                for (const change of sequence)
                    if (change.step === step) {
                        if (Object.hasOwn(s.keys, change.name))
                            s.keys[change.name as keyof typeof s.keys] = change.value;
                        else
                            changeControl(
                                app.world,
                                "dynamicMover",
                                bodies,
                                values,
                                change.name,
                                change.value,
                            );
                    }
                app.world.tick();
                expect(steppedHash, `${stem} step ${step}`).toBe(gold[step]!.hash);
                const m = s.mover;
                expect(
                    [
                        s.time,
                        +s.plateArmed,
                        +m.onGround,
                        +m.walkable,
                        +m.jumping,
                        m.jumpTicks,
                        m.velocity.x,
                        m.velocity.y,
                        m.velocity.z,
                        m.pogoLength,
                        s.gems.filter(Boolean).length,
                        s.ballIndex,
                    ].map(Math.fround),
                    `${stem} step ${step} mover`,
                ).toEqual(states[step]!);
                const events = app.world.resource(PhysicsEvents);
                const sensors = (records: typeof events.sensorBegin) =>
                    records.records
                        .slice(0, records.count)
                        .map((e) => [
                            shapeIds.get(e.sensorShape)!,
                            bodyIds.get(e.sensorBody)!,
                            shapeIds.get(e.visitorShape)!,
                            bodyIds.get(e.visitorBody)!,
                        ]);
                expect([
                    events.contactBegin.count,
                    events.contactEnd.count,
                    events.contactHit.count,
                    events.joints.count,
                ]).toEqual([0, 0, 0, 0]);
                expect<(number | string)[][][]>(
                    [sensors(events.sensorBegin), sensors(events.sensorEnd), [], [], [], []],
                    `${stem} step ${step} events`,
                ).toEqual(gold[step]!.events);
            }
        } finally {
            app.dispose();
        }
    });
