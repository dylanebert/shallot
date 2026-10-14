import { expect, spyOn, test } from "bun:test";
import { BodyType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { kernel } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/build-hull";
import { createHullShape } from "../shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../solver/distanceJoint";
import { defaultJointDef, destroyJointInternal } from "../solver/joint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { worldProfile } from "./world";

function splitSubject(world: ReturnType<typeof createTestWorld>): () => void {
    const a = createTestBody(world, { type: BodyType.Dynamic, enableSleep: false });
    const b = createTestBody(world, {
        type: BodyType.Dynamic,
        enableSleep: false,
        position: { x: 1, y: 0, z: 0 },
    });
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    createDistanceJoint(world, defaultDistanceJointDef(base));
    const removed = createDistanceJoint(world, defaultDistanceJointDef(base)).joint;
    destroyJointInternal(world, removed, true);
    const k = kernel(world.ecsState);
    const island = bodyField(world, a, BodyField.islandId);
    return () => {
        k.bodySetActiveWorld(world.worldId);
        k.islandSetField(island, 3, 1);
        k.islandSetSplitCandidate(island);
        stepTestWorld(world, Math.fround(1 / 60), 4);
        if (k.islandField(island, 3) !== 0 || k.islandSplitCandidate() !== -1)
            throw new Error("split allocation subject did not complete its split task");
    };
}

test("the kernel times every solve substage through its platform clock", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 } });
    const body = createTestBody(world, { type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
    createHullShape(world, body, testShapeDef(), makeBoxHull(0.5, 0.5, 0.5));
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        stepTestWorld(world, 1 / 60, 4);
        const profile = worldProfile(world);
        for (const name of [
            "prepareConstraints",
            "warmStart",
            "solveImpulses",
            "relaxImpulses",
            "applyRestitution",
            "storeImpulses",
            "transforms",
        ] as const)
            expect(profile[name]).toBeGreaterThan(0);
        expect(profile.step).toBeGreaterThan(profile.solve);
        expect(profile.solve).toBeGreaterThan(profile.constraints);
        expect(profile.constraints).toBeGreaterThan(profile.prepareConstraints);
    } finally {
        now.mockRestore();
        destroyTestWorld(world);
    }
});

test("split timers measure only a candidate task and disabled sleep stays untimed", () => {
    const disabled = createTestWorld({ enableSleep: false, gravity: { x: 0, y: 0, z: 0 } });
    const enabled = createTestWorld({ enableSleep: true, gravity: { x: 0, y: 0, z: 0 } });
    createTestBody(disabled, { type: BodyType.Dynamic, enableSleep: false });
    createTestBody(enabled, { type: BodyType.Dynamic, enableSleep: false });
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        stepTestWorld(disabled, Math.fround(1 / 60), 4);
        stepTestWorld(enabled, Math.fround(1 / 60), 4);
        const a = worldProfile(disabled);
        const b = worldProfile(enabled);
        expect(a.splitIslands).toBe(0);
        expect(b.splitIslands).toBe(0);
        expect(a.sleepIslands).toBe(0);
        expect(b.sleepIslands).toBe(1);
        expect(a.transforms).toBe(b.transforms);
        splitSubject(disabled)();
        expect(worldProfile(disabled).splitIslands).toBe(1);
        expect(worldProfile(disabled).sleepIslands).toBe(0);
    } finally {
        now.mockRestore();
        destroyTestWorld(disabled);
        destroyTestWorld(enabled);
    }
});

test("sleep gathers the deferred split candidate before the next step's split task", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: true });
    const a = createTestBody(world, { type: BodyType.Dynamic });
    const b = createTestBody(world, { type: BodyType.Dynamic, position: { x: 1, y: 0, z: 0 } });
    const base = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b };
    createDistanceJoint(world, defaultDistanceJointDef(base));
    const removed = createDistanceJoint(world, defaultDistanceJointDef(base)).joint;
    destroyJointInternal(world, removed, true);
    const k = kernel(world.ecsState);
    let clock = 0;
    const now = spyOn(performance, "now").mockImplementation(() => ++clock);
    try {
        for (let i = 0; i < 40 && k.islandSplitCandidate() === -1; ++i)
            stepTestWorld(world, Math.fround(1 / 60), 4);
        expect(k.islandSplitCandidate()).toBeGreaterThanOrEqual(0);
        expect(worldProfile(world).splitIslands).toBe(0);
        expect(worldProfile(world).sleepIslands).toBe(1);
        stepTestWorld(world, Math.fround(1 / 60), 4);
        expect(k.islandSplitCandidate()).toBe(-1);
        expect(worldProfile(world).splitIslands).toBe(1);
        expect(worldProfile(world).sleepIslands).toBe(1);
    } finally {
        now.mockRestore();
        destroyTestWorld(world);
    }
});

test("a bare falling box records step and solve timings without setup", () => {
    const world = createTestWorld({ gravity: { x: 0, y: -10, z: 0 } });
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 5, z: 0 },
        });
        createHullShape(world, body, testShapeDef(), makeBoxHull(0.5, 0.5, 0.5));
        stepTestWorld(world, 1 / 60, 4);
        const profile = worldProfile(world);
        expect(profile.step).toBeGreaterThan(0);
        expect(profile.solve).toBeGreaterThan(0);
        expect(profile.step).toBeGreaterThanOrEqual(profile.solve);
        profile.step = -1;
        expect(worldProfile(world).step).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});
