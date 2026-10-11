import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { World } from "../../engine";
import { restore, snapshot } from "./api/snapshot";
import { contactIds } from "./collision/contact";
import { readContactManifolds } from "./collision/manifoldstore";
import { BodyType, defaultQueryFilter, defaultShapeDef } from "./common/types";
import { queryColumns } from "./kernel/querycolumns";
import { makeBoxHull } from "./shapes/hull";
import { createHullShape, createSphereShape } from "./shapes/shape";
import { defaultJointDef } from "./solver/joint";
import { createRevoluteJoint, defaultRevoluteJointDef } from "./solver/revoluteJoint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { bodySetAwake } from "./world/body";
import { hashWorldState } from "./world/hash";
import type { WorldState } from "./world/world";
import { worldCounters } from "./world/world";

setDefaultTimeout(CEILING.node);

function makeWorld() {
    const owner = new World();
    return { owner, state: createTestWorld({}, owner) };
}

function ray(world: WorldState, x = 0): void {
    const query = queryColumns(world);
    const k = query.prepare({ x, y: 10, z: 0 }, defaultQueryFilter());
    query.translation({ x: 0, y: -20, z: 0 });
    k.worldQuery(world.worldId, 3, 0);
}

function dispose(owner: World, world: WorldState): void {
    destroyTestWorld(world);
    owner.dispose();
}

test("snapshot after a stacked hull scene and ray never clones detached query views", () => {
    const { owner, state } = makeWorld();
    try {
        const ground = createTestBody(state, { type: BodyType.Static });
        createHullShape(state, ground, defaultShapeDef(), makeBoxHull(10, 1, 10));
        for (let i = 0; i < 6; i++) {
            const body = createTestBody(state, {
                type: BodyType.Dynamic,
                position: { x: 0, y: 2 + i * 2, z: 0 },
            });
            createSphereShape(state, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 1,
            });
        }
        for (let tick = 0; tick < 90; tick++) stepTestWorld(state, 1 / 60);
        ray(state);
        expect(() => snapshot(state)).not.toThrow();
    } finally {
        dispose(owner, state);
    }
});

test("a restored empty queried World uploads geometry added after restore", () => {
    const { owner, state } = makeWorld();
    try {
        ray(state);
        const query = queryColumns(state);
        expect(query.resultU[0]).toBe(0xffffffff);
        restore(state, snapshot(state));
        const body = createTestBody(state, {
            type: BodyType.Static,
            position: { x: 10, y: 0, z: 0 },
        });
        createHullShape(state, body, defaultShapeDef(), makeBoxHull(1, 1, 1));
        ray(state, 10);
        expect(query.resultU[0]).not.toBe(0xffffffff);
    } finally {
        dispose(owner, state);
    }
});

test("manifold allocation and sleeping-contact normals replay through a fresh World restore", () => {
    const sourceWorld = makeWorld();
    const targetWorld = makeWorld();
    const { owner: sourceOwner, state: source } = sourceWorld;
    const { owner: targetOwner, state: target } = targetWorld;
    function continueTick(world: WorldState, tick: number): bigint {
        if (tick === 0) {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 5, y: 2, z: 0 },
            });
            createHullShape(world, body, testShapeDef(), makeBoxHull(1, 1, 1));
        }
        if (tick === 30) bodySetAwake(world, 1, true);
        stepTestWorld(world, 1 / 60);
        return hashWorldState(world);
    }
    try {
        const ground = createTestBody(source, { type: BodyType.Static });
        createHullShape(source, ground, defaultShapeDef(), makeBoxHull(10, 1, 10));
        const sphere = createTestBody(source, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
        });
        createSphereShape(source, sphere, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 1,
        });
        for (let tick = 0; tick < 120; tick++) stepTestWorld(source, 1 / 60);
        bodySetAwake(source, sphere, false);
        const saved = snapshot(source);
        const contact = contactIds(source).find(
            (id) => readContactManifolds(source, id).length > 0,
        )!;
        const normal = { ...readContactManifolds(source, contact)[0].normal };
        const expected = Array.from({ length: 120 }, (_, tick) => continueTick(source, tick));
        restore(target, saved);
        const restoredNormal = { ...readContactManifolds(target, contact)[0].normal };
        for (let tick = 0; tick < 120; tick++)
            expect({ tick, hash: continueTick(target, tick) }).toEqual({
                tick,
                hash: expected[tick],
            });
        expect(restoredNormal).toEqual(normal);
    } finally {
        dispose(targetOwner, target);
        dispose(sourceOwner, source);
    }
});

test("every reachable WorldState in a stepped contact, sleeping-island and joint scene is the live root", () => {
    const { owner, state } = makeWorld();
    try {
        const ground = createTestBody(state, { type: BodyType.Static });
        createHullShape(state, ground, defaultShapeDef(), makeBoxHull(10, 1, 10));
        const sphere = createTestBody(state, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 2, z: 0 },
        });
        createSphereShape(state, sphere, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 1,
        });
        const arm = createTestBody(state, {
            type: BodyType.Dynamic,
            position: { x: 5, y: 3, z: 0 },
        });
        createHullShape(state, arm, testShapeDef(), makeBoxHull(0.5, 0.5, 0.5));
        createRevoluteJoint(
            state,
            defaultRevoluteJointDef({ ...defaultJointDef(), bodyIdA: ground, bodyIdB: arm }),
        );
        for (let tick = 0; tick < 120; tick++) stepTestWorld(state, 1 / 60);
        bodySetAwake(state, sphere, false);
        expect(worldCounters(state).contactCount).toBeGreaterThan(0);
        ray(state);
        const saved = snapshot(state);
        restore(state, saved);
        const seen = new Set<object>();
        function visit(value: unknown): void {
            if (value === null || typeof value !== "object" || seen.has(value)) return;
            seen.add(value);
            if ("bodyStore" in value && "broadPhase" in value && "worldId" in value)
                expect(value).toBe(state);
            if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) return;
            if (value instanceof Map)
                for (const [key, item] of value) {
                    visit(key);
                    visit(item);
                }
            else if (value instanceof Set) for (const item of value) visit(item);
            else
                for (const key of Reflect.ownKeys(value)) {
                    const descriptor = Object.getOwnPropertyDescriptor(value, key);
                    if (descriptor && "value" in descriptor) visit(descriptor.value);
                }
        }
        visit(state);
        expect(seen.has(state.queryColumns!)).toBe(true);
    } finally {
        dispose(owner, state);
    }
});
