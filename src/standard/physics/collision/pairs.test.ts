import { afterAll, beforeAll, expect, test } from "bun:test";
import { BodyType, init, makeBoxHull, shutdown } from "../api/index";
import { defaultFilter, type Filter } from "../common/types";
import { EventKind, eventCount } from "../kernel/eventbuffers";
import { createHullShape, createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";

beforeAll(async () => {
    await init(undefined);
});
afterAll(() => shutdown(undefined));

function touches(a: Filter, b: Filter): boolean {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        for (const filter of [a, b]) {
            const body = createTestBody(world, { type: BodyType.Dynamic });
            createSphereShape(world, body, testShapeDef({ filter, enableContactEvents: true }), {
                center: { x: 0, y: 0, z: 0 },
                radius: 1,
            });
        }
        stepTestWorld(world, 1 / 60);
        return eventCount(world, EventKind.ContactBegin) !== 0;
    } finally {
        destroyTestWorld(world);
    }
}
const filter = (categoryBits: bigint, maskBits: bigint, groupIndex = 0): Filter => ({
    categoryBits,
    maskBits,
    groupIndex,
});

test("pair filtering requires both masks, preserves both u64 halves, and gives shared nonzero groups precedence", () => {
    expect(touches(defaultFilter(), defaultFilter())).toBe(true);
    expect(touches(filter(1n, 2n), filter(2n, 2n))).toBe(false);
    expect(touches(filter(1n, 1n), filter(2n, 2n))).toBe(false);
    expect(touches(filter(1n, 2n), filter(2n, 1n))).toBe(true);
    const high = 1n << 40n;
    expect(touches(filter(high, high), filter(high, high))).toBe(true);
    expect(touches(filter(high, high), filter(1n << 8n, 1n << 8n))).toBe(false);
    const top = (1n << 63n) | (1n << 31n);
    expect(touches(filter(top, top), filter(top, top))).toBe(true);
    expect(touches(filter(1n, 1n, 7), filter(2n, 2n, 7))).toBe(true);
    expect(touches(filter(255n, 255n, -3), filter(255n, 255n, -3))).toBe(false);
    expect(touches(filter(1n, 1n, 1), filter(2n, 2n, 2))).toBe(false);
});

test("two overlapping moving boxes begin touching once, and the next step begins or ends no touch", () => {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    const drift = { x: 0, y: 0.5, z: 0 };
    try {
        const a = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: -0.25, y: 0, z: 0 },
            linearVelocity: drift,
        });
        createHullShape(
            world,
            a,
            testShapeDef({ enableContactEvents: true }),
            makeBoxHull(0.5, 0.5, 0.5),
        );
        const b = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0.25, y: 0, z: 0 },
            linearVelocity: drift,
        });
        createHullShape(
            world,
            b,
            testShapeDef({ enableContactEvents: true }),
            makeBoxHull(0.5, 0.5, 0.5),
        );

        stepTestWorld(world, 1 / 60);
        expect(eventCount(world, EventKind.ContactBegin)).toBe(1);
        stepTestWorld(world, 1 / 60);
        expect(eventCount(world, EventKind.ContactBegin)).toBe(0);
        expect(eventCount(world, EventKind.ContactEnd)).toBe(0);
    } finally {
        destroyTestWorld(world);
    }
});
