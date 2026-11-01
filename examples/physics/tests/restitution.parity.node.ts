import { expect, setDefaultTimeout, test } from "bun:test";
import { Body, createApp, hashPhysics } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { RestitutionPlugin, RestitutionState, restitutionScenes } from "../src/restitution";
import { type Controls, changeControl, controlValues, createScene, scenes } from "../src/scenes";
import { sampleGold } from "./gold";

setDefaultTimeout(TIMEOUT.node);
for (const scene of Object.keys(restitutionScenes) as (keyof typeof restitutionScenes)[])
    for (const controlled of scenes[scene].controls ? [false, true] : [false])
        test(`${scene} equals every native step${controlled ? " through all controls" : " and rejects a velocity change"}`, async () => {
            const stem = `${scene}${controlled ? ".controls" : ""}`;
            const gold = sampleGold(
                await Bun.file(new URL(`./${stem}.gold.txt`, import.meta.url)).text(),
            );
            expect(gold).toHaveLength(600);
            const readouts =
                scene === "bounceImpulse" || scene === "restitutionPropagation"
                    ? (
                          await Bun.file(
                              new URL(`./${stem}.readouts.gold.txt`, import.meta.url),
                          ).text()
                      )
                          .trimEnd()
                          .split("\n")
                          .map((line) => (JSON.parse(line) as number[]).map(Math.fround))
                    : null;
            const changes = controlled
                ? (await Bun.file(new URL(`./${scene}.controls.txt`, import.meta.url)).text())
                      .trimEnd()
                      .split("\n")
                      .map((line) => {
                          const [step, label, value] = line.split("\t");
                          return { step: Number(step), label: label!, value: Number(value) };
                      })
                : [];
            const app = await createApp({
                defaults: false,
                plugins: [RestitutionPlugin],
                setup(world) {
                    createScene(world, scene);
                },
            });
            try {
                const values = controlValues(scene);
                const controls: Controls = scenes[scene].controls ?? {};
                const used = new Set<string>();
                for (let step = 0; step < 600; step++) {
                    for (const change of changes)
                        if (change.step === step) {
                            const key = Object.entries(controls).find(
                                ([, c]) => c.label === change.label,
                            )![0];
                            changeControl(app.world, scene, [], values, key, change.value);
                            used.add(key);
                        }
                    app.world.tick();
                    expect(hashPhysics(app.world), `step ${step}`).toBe(gold[step]!.hash);
                    if (readouts) {
                        const s = app.world.resource(RestitutionState);
                        const actual =
                            scene === "bounceImpulse"
                                ? [
                                      s.mass,
                                      s.approachSpeed,
                                      s.totalImpulse,
                                      s.expectedImpulse,
                                      s.contactSteps,
                                      Number(s.hit),
                                      Number(s.bouncing),
                                  ]
                                : [...s.apex, ...s.rebounding.map(Number), ...s.done.map(Number)];
                        expect(actual, `step ${step} readouts`).toEqual(readouts[step]!);
                    }
                }
                if (controlled)
                    expect([...used].sort()).toEqual(
                        Object.entries(controls)
                            .filter(([, c]) => !c.read)
                            .map(([key]) => key)
                            .sort(),
                    );
            } finally {
                app.dispose();
            }
            if (!controlled) {
                const wrong = await createApp({
                    defaults: false,
                    plugins: [RestitutionPlugin],
                    setup(world) {
                        const bodies = createScene(world, scene);
                        world.storage(Body).linearVelocity.set(bodies[0]!, 0, 123, 0);
                    },
                });
                try {
                    wrong.world.tick();
                    expect(hashPhysics(wrong.world)).not.toBe(gold[0]!.hash);
                } finally {
                    wrong.dispose();
                }
            }
        });
