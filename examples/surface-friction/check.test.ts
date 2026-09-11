import { afterAll, afterEach, expect } from "bun:test";
import { State, Time } from "../../src/engine";
import { clear, register } from "../../src/engine/ecs/traits";
import { check } from "../../src/harness/check";
import { runBrowserCheck } from "../../src/harness/driver";
import { Color, Part } from "../../src/standard/part";
import {
    Body,
    bodyTraits,
    Joint,
    jointTraits,
    Physics,
    PhysicsPlugin,
    Spring,
    springTraits,
} from "../../src/standard/physics";
import { shutdown } from "../../src/standard/physics/api";
import { attach } from "../../src/standard/physics/headless.fixture";
import { Slab } from "../../src/standard/slab";
import { build } from "./src/ramp";

async function headlessRecipeState(): Promise<State> {
    clear();
    const state = new State();
    register("body", Body, bodyTraits);
    register("spring", Spring, springTraits);
    register("joint", Joint, jointTraits);
    register("part", Part);
    register("color", Color);
    Slab.collect();
    PhysicsPlugin.initialize?.(state);
    await PhysicsPlugin.warm?.(state);
    attach(state, PhysicsPlugin);
    return state;
}

let live: State | null = null;

afterEach(() => {
    if (live) PhysicsPlugin.dispose?.(live);
    live = null;
});
afterAll(shutdown);

check(
    "a low-friction box slides off the ramp while a high-friction box holds",
    {
        claim: "Body.friction makes a low-friction box leave the ramp while a high-friction box holds",
    },
    async () => {
        const state = await headlessRecipeState();
        live = state;
        build(state);

        const boxes = [...state.query([Body])].filter((eid) => Body.mass.get(eid) > 0);
        expect(boxes).toHaveLength(5);
        const low = boxes[0];
        const high = boxes[boxes.length - 1];

        for (let i = 0; i < 180; i++) state.step(Time.FIXED_DT);

        const lowPosition = Physics.body(low)?.getPosition();
        const highPosition = Physics.body(high)?.getPosition();
        expect(lowPosition).toBeDefined();
        expect(highPosition).toBeDefined();
        // The low-friction box has fallen past the ramp's top surface; the high-friction box remains near it.
        expect(lowPosition!.y).toBeLessThan(8);
        expect(highPosition!.y).toBeGreaterThan(9);
    },
);

check(
    "Chromium observes the friction ladder through the in-page harness",
    {
        claim: "Chromium observes Body.friction making a low-friction box leave the ramp while a high-friction box holds",
        size: "integration",
        requires: ["chromium"],
    },
    async () => {
        const verdict = await runBrowserCheck(import.meta.dir);
        if (!verdict.ok) {
            throw Object.assign(
                new Error("the browser harness did not observe the friction claim"),
                { runtime: verdict.runtime, hardware: verdict.hardware },
            );
        }
        return verdict;
    },
);
