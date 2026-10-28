import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { Body, BodyType, Shape } from "../../core/physics";
import { createApp } from "../../engine";
import { PhysicsWorldDefinition, StandardPhysicsPlugin, StepPhysicsSystem, stepPhysics } from ".";

setDefaultTimeout(TIMEOUT.node);

test("manual solve phases accumulate under their caller, not the disabled automatic step", async () => {
    let calls = 1;
    const app = await createApp({
        defaults: false,
        plugins: [
            StandardPhysicsPlugin,
            {
                name: "Game",
                systems: [
                    {
                        name: "physics",
                        group: "fixed",
                        after: [StepPhysicsSystem],
                        update(world) {
                            for (let i = 0; i < calls; i++) stepPhysics(world, 1 / 120, 4);
                        },
                    },
                ],
            },
        ],
    });
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        const world = app.world;
        world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape);
        const totals = new Map<string, number>();
        const samples = new Map<string, number[]>();
        world.recordSink = (name, measurement) => {
            totals.set(name, (totals.get(name) ?? 0) + measurement[1]);
            const values = samples.get(name) ?? [];
            values.push(measurement[1]);
            samples.set(name, values);
        };
        for (calls = 1; calls <= 2; calls++) {
            totals.clear();
            samples.clear();
            world.tick();
            for (const phase of ["collide", "solve"]) {
                const name = `Game/physics/${phase}`;
                expect(samples.get(name)).toHaveLength(calls);
                expect(totals.get(name)).toBeGreaterThan(0);
                expect(totals.get(name)).toBe(samples.get(name)!.reduce((sum, ms) => sum + ms, 0));
            }
            expect(totals.get("StandardPhysics/step")).toBeGreaterThan(0);
            expect(
                [...totals.keys()].filter((name) => name.startsWith("StandardPhysics/step/")),
            ).toEqual([]);
        }
    } finally {
        now.mockRestore();
        app.dispose();
    }
});
