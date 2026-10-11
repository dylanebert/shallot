import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { BodyType } from "../common/types";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import { kernel, rethrowQueryError, setQueryCallback } from "../kernel/kernel";
import { queryColumns } from "../kernel/querycolumns";
import { makeBoxHull } from "../shapes/hull";
import { createHullShape, createSphereShape, getSensorData, shapeIsValid } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { hashWorldState } from "../world/hash";
import {
    type WorldState,
    worldCounters,
    worldSetCustomFilterCallback,
    worldSetPreSolveCallback,
} from "../world/world";
import { restore, snapshot } from "./snapshot";

function contactWorld(pre = false): WorldState {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const shape = testShapeDef({
        enableCustomFiltering: !pre,
        enablePreSolveEvents: pre,
        enableContactEvents: true,
    });
    const ground = createTestBody(world);
    createHullShape(world, ground, shape, makeBoxHull(2, 0.5, 2));
    const ball = createTestBody(world, {
        type: BodyType.Dynamic,
        position: { x: 0, y: 0.99, z: 0 },
    });
    createSphereShape(world, ball, shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    return world;
}

function contactIdentities(world: WorldState): string[] {
    return Array.from({ length: eventCount(world, EventKind.ContactBegin) }, (_, i) => {
        const contact = eventId(world, EventKind.ContactBegin, i, 4);
        const shapeA = eventId(world, EventKind.ContactBegin, i, 0);
        const shapeB = eventId(world, EventKind.ContactBegin, i, 2);
        return `${contact.index1}:${shapeA.index1}:${shapeB.index1}`;
    }).sort();
}

function emptyEventCounts(world: WorldState): number[] {
    if (world.locked) return [0, 0, 0, 0, 0, 0, 0];
    return [
        eventCount(world, EventKind.ContactBegin),
        eventCount(world, EventKind.ContactEnd),
        eventCount(world, EventKind.ContactHit),
        eventCount(world, EventKind.SensorBegin),
        eventCount(world, EventKind.SensorEnd),
        eventCount(world, EventKind.BodyMove),
        eventCount(world, EventKind.Joint),
    ];
}

function moveSummary(world: WorldState) {
    const k = kernel(world.ecsState);
    return {
        stepIndex: world.stepIndex,
        syncCount: world.bodyStore.syncCount,
        moves: eventCount(world, EventKind.BodyMove),
        transform: {
            p: [
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 1),
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 2),
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 3),
            ],
            q: [
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 4),
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 5),
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 6),
                k.eventFloat(world.worldId, EventKind.BodyMove, 0, 7),
            ],
        },
    };
}

function siblingOverlap(world: WorldState): void {
    const q = queryColumns(world);
    const k = q.prepare({ x: 0, y: 0, z: 0 });
    q.bounds({
        lowerBound: { x: -1, y: -1, z: -1 },
        upperBound: { x: 1, y: 1, z: 1 },
    });
    const previous = setQueryCallback(world.ecsState, () => 1);
    try {
        k.worldQuery(world.worldId, 0, 1);
        rethrowQueryError(world.ecsState);
    } finally {
        setQueryCallback(world.ecsState, previous);
    }
}

test("custom filter rejects new pairs and clearing it restores default filtering", () => {
    const world = contactWorld();
    try {
        let calls = 0;
        worldSetCustomFilterCallback(world, (a, b) => {
            expect([a.index1, b.index1]).toEqual([1, 2]);
            expect(() => shapeIsValid(world, a)).toThrow("shared kernel instance");
            calls++;
            return false;
        });
        stepTestWorld(world, 1 / 60);
        expect(calls).toBe(1);
        expect(worldCounters(world).contactCount).toBe(0);
        worldSetCustomFilterCallback(world, null);
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 0.99, z: 0 },
        });
        createSphereShape(world, body, testShapeDef({ enableCustomFiltering: true }), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(world, 1 / 60);
        expect(calls).toBe(1);
        expect(worldCounters(world).contactCount).toBeGreaterThan(0);
    } finally {
        destroyTestWorld(world);
    }
});

describe("first-shape filtering under contact capacity pressure", () => {
    let control: WorldState | undefined;
    let filtered: WorldState | undefined;

    beforeAll(() => {
        control = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        filtered = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
        for (const world of [control, filtered]) {
            for (let i = 0; i < 100; ++i) {
                const body = createTestBody(world, { type: BodyType.Dynamic });
                createSphereShape(
                    world,
                    body,
                    testShapeDef({
                        enableCustomFiltering: i === 0,
                        enableContactEvents: true,
                    }),
                    { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                );
            }
        }
        worldSetCustomFilterCallback(filtered, () => true);
    });

    afterAll(() => {
        if (control) destroyTestWorld(control);
        if (filtered) destroyTestWorld(filtered);
    });

    test("preserves native survivor identities at capacity", () => {
        if (!control || !filtered) throw new Error("capacity fixtures did not initialize");
        stepTestWorld(control, 1 / 60);
        stepTestWorld(filtered, 1 / 60);
        expect(worldCounters(control).contactCount).toBe(1600);
        expect(worldCounters(filtered).contactCount).toBe(1600);
        expect(contactIdentities(filtered)).toEqual(contactIdentities(control));
        expect(hashWorldState(filtered)).toBe(hashWorldState(control));
    });
});

test("pre-solve refuses stepping an unlocked sibling sharing the kernel instance", () => {
    const world = contactWorld(true);
    const sibling = createTestWorld();
    try {
        worldSetPreSolveCallback(world, () => {
            stepTestWorld(sibling, 1 / 60);
            return false;
        });
        expect(() => stepTestWorld(world, 1 / 60)).toThrow(
            "cannot re-enter the shared kernel instance from a step callback",
        );
        expect(sibling.stepIndex).toBe(0);
        expect(sibling.locked).toBe(false);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

test("pre-solve sees the world-space point and A-to-B normal after custom material mixing", () => {
    const world = contactWorld(true);
    try {
        const order: string[] = [];
        world.frictionCallback = () => {
            order.push("material");
            return 0.5;
        };
        worldSetPreSolveCallback(world, (a, b, point, normal) => {
            expect(a.index1).toBe(1);
            expect(b.index1).toBe(2);
            expect(point.x).toBe(0);
            expect(point.y).toBeCloseTo(0.495);
            expect(normal).toEqual({ x: 0, y: 1, z: 0 });
            order.push("pre");
            return false;
        });
        stepTestWorld(world, 1 / 60);
        expect(order).toEqual(["material", "pre"]);
        expect(eventCount(world, EventKind.ContactBegin)).toBe(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("a throwing pre-solve completes step bookkeeping and views before rethrowing", () => {
    const world = contactWorld(true);
    const control = contactWorld(true);
    const error = new Error("bookkeeping probe");
    try {
        kernel(world.ecsState).bodySetEntity(world.worldId, 1, 42);
        kernel(control.ecsState).bodySetEntity(control.worldId, 1, 42);
        worldSetPreSolveCallback(control, () => false);
        worldSetPreSolveCallback(world, () => {
            throw error;
        });
        stepTestWorld(control, 1 / 60);
        const expected = moveSummary(control);
        expect(() => stepTestWorld(world, 1 / 60)).toThrow(error);
        expect(moveSummary(world)).toEqual(expected);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(control);
    }
});

test("destroying a world releases its callback references without changing the caller's functions", () => {
    const world = contactWorld(true);
    const callback = () => true;
    worldSetCustomFilterCallback(world, callback);
    worldSetPreSolveCallback(world, callback);
    destroyTestWorld(world);
    expect(world.customFilterCallback).toBeNull();
    expect(world.preSolveCallback).toBeNull();
    expect(callback()).toBe(true);
});

test("a collision callback observes locked-world guards and cannot capture or reenter a partial step", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const sensorBody = createTestBody(world);
    const sensor = createSphereShape(
        world,
        sensorBody,
        testShapeDef({
            isSensor: true,
            enableSensorEvents: true,
            enableCustomFiltering: true,
        }),
        { center: { x: 0, y: 0, z: 0 }, radius: 2 },
    )!;
    const visitorBody = createTestBody(world, { type: BodyType.Dynamic });
    createSphereShape(world, visitorBody, testShapeDef({ enableSensorEvents: true }), {
        center: { x: 0, y: 0, z: 0 },
        radius: 0.5,
    });
    try {
        stepTestWorld(world, 1 / 60);
        expect(getSensorData(world, sensor)).toHaveLength(1);
        const saved = snapshot(world);
        let calls = 0;
        worldSetCustomFilterCallback(world, () => {
            calls++;
            expect(getSensorData(world, sensor)).toHaveLength(0);
            expect(emptyEventCounts(world)).toEqual([0, 0, 0, 0, 0, 0, 0]);
            expect(() => snapshot(world)).toThrow("while it is stepping");
            expect(() => restore(world, saved)).toThrow("while it is stepping");
            expect(() => stepTestWorld(world, 1 / 60)).toThrow("shared kernel instance");
            destroyTestWorld(world);
            expect(world.inUse).toBe(true);
            return true;
        });
        stepTestWorld(world, 1 / 60);
        expect(calls).toBe(1);
        expect(world.stepIndex).toBe(2);
        expect(getSensorData(world, sensor)).toHaveLength(1);
    } finally {
        destroyTestWorld(world);
    }
});

test("collision callbacks refuse interleaving a sibling query on the same kernel", () => {
    const world = contactWorld();
    const sibling = createTestWorld();
    const saved = snapshot(sibling);
    try {
        worldSetCustomFilterCallback(world, () => {
            expect(() => snapshot(sibling)).toThrow(
                "cannot re-enter the shared kernel instance from a step callback",
            );
            expect(() => restore(sibling, saved)).toThrow(
                "cannot re-enter the shared kernel instance from a step callback",
            );
            siblingOverlap(sibling);
            return true;
        });
        expect(() => stepTestWorld(world, 1 / 60)).toThrow(
            "cannot re-enter the shared kernel instance from a step callback",
        );
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});

for (const pre of [false, true]) {
    test(`${pre ? "pre-solve" : "custom filter"} restores caller callback identity and its kernel gate`, () => {
        const world = contactWorld(pre);
        const target = createTestWorld();
        let calls = 0;
        const callback = () => {
            calls++;
            return false;
        };
        try {
            if (pre) worldSetPreSolveCallback(world, callback);
            else worldSetCustomFilterCallback(world, callback);
            const saved = snapshot(world);
            if (pre) worldSetPreSolveCallback(world, null);
            else worldSetCustomFilterCallback(world, null);
            for (const destination of [world, target]) {
                restore(destination, saved);
                expect(pre ? destination.preSolveCallback : destination.customFilterCallback).toBe(
                    callback,
                );
                const before = calls;
                stepTestWorld(destination, 1 / 60);
                expect(calls - before).toBe(1);
            }
        } finally {
            destroyTestWorld(world);
            destroyTestWorld(target);
        }
    });
    test(`${pre ? "pre-solve" : "custom filter"} exceptions return through the kernel before being rethrown`, () => {
        const world = contactWorld(pre);
        const error = new Error("callback failure");
        try {
            if (pre)
                worldSetPreSolveCallback(world, () => {
                    throw error;
                });
            else
                worldSetCustomFilterCallback(world, () => {
                    throw error;
                });
            expect(() => stepTestWorld(world, 1 / 60)).toThrow(error);
            expect(world.locked).toBe(false);
            if (pre) worldSetPreSolveCallback(world, null);
            else worldSetCustomFilterCallback(world, null);
            expect(() => stepTestWorld(world, 1 / 60)).not.toThrow();
        } finally {
            destroyTestWorld(world);
        }
    });
}
