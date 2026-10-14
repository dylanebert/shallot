import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import { contactIds } from "./collision/contact";
import { readContactManifolds } from "./collision/manifoldstore";
import { BodyType } from "./common/types";
import { kernel, rethrowQueryError, setQueryCallback } from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import { readShapeAabb } from "./kernel/shapecolumns";
import { makeBoxHull } from "./shapes/hull";
import { computeShapeMass, createHullShape, createSphereShape } from "./shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { bodyGetLinearVelocity, bodyGetTransform } from "./world/body";
import type { WorldState } from "./world/world";

setDefaultTimeout(TIMEOUT.node);

const identity = () => ({ p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } });

function contact(world: WorldState): number {
    const found = contactIds(world)[0];
    if (found === undefined) throw new Error("view test did not create a contact");
    return found;
}

function reads(world: WorldState, body: number, shape: number, contactId: number): unknown {
    return {
        position: { ...bodyGetTransform(world, body, identity()).p },
        velocity: { ...bodyGetLinearVelocity(world, body, { x: 0, y: 0, z: 0 }) },
        bounds: readShapeAabb(world, shape, {
            lowerBound: { x: 0, y: 0, z: 0 },
            upperBound: { x: 0, y: 0, z: 0 },
        }),
        mass: computeShapeMass(world, shape),
        manifolds: readContactManifolds(world, contactId),
    };
}

test("body, shape and contact reads survive a sibling's memory growth and destruction without a step or query", () => {
    const owner = new World();
    const a = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    let b: WorldState | undefined;
    try {
        const mover = createTestBody(a, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = createSphereShape(a, mover, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        })!;
        const ground = createTestBody(a, { type: BodyType.Static, position: { x: 5, y: 0, z: 0 } });
        createHullShape(a, ground, testShapeDef(), makeBoxHull(2, 0.5, 2));
        const touching = createTestBody(a, {
            type: BodyType.Dynamic,
            position: { x: 5, y: 1, z: 0 },
        });
        createSphereShape(a, touching, testShapeDef({ enableContactEvents: true }), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(a, 1 / 60);
        const contactId = contact(a);
        const before = reads(a, mover, shape, contactId);
        const bytes = kernel(owner).memory.buffer.byteLength;
        b = createTestWorld({}, owner);
        for (let i = 0; i < 2000; i++) {
            const body = createTestBody(b, {
                type: BodyType.Dynamic,
                position: { x: i * 2, y: 10, z: 0 },
            });
            createSphereShape(b, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
        }
        expect(kernel(owner).memory.buffer.byteLength).toBeGreaterThan(bytes);
        expect(reads(a, mover, shape, contactId)).toEqual(before);
        destroyTestWorld(b);
        b = undefined;
        expect(reads(a, mover, shape, contactId)).toEqual(before);
    } finally {
        if (b) destroyTestWorld(b);
        destroyTestWorld(a);
        owner.dispose();
    }
});

test("body, shape and contact reads survive memory growth that moves no column", () => {
    const owner = new World();
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 } }, owner);
    try {
        const mover = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 1, y: 2, z: 3 },
            linearVelocity: { x: 7, y: 0, z: 0 },
        });
        const shape = createSphereShape(world, mover, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        })!;
        const ground = createTestBody(world, {
            type: BodyType.Static,
            position: { x: 5, y: 0, z: 0 },
        });
        createHullShape(world, ground, testShapeDef(), makeBoxHull(2, 0.5, 2));
        const touching = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 5, y: 1, z: 0 },
        });
        createSphereShape(world, touching, testShapeDef({ enableContactEvents: true }), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(world, 1 / 60);
        const contactId = contact(world);
        const before = reads(world, mover, shape, contactId);
        const memory = kernel(owner).memory;
        const buffer = memory.buffer;
        expect(buffer).not.toBeInstanceOf(SharedArrayBuffer);
        memory.grow(1);
        expect(buffer.byteLength).toBe(0);
        expect(reads(world, mover, shape, contactId)).toEqual(before);
    } finally {
        destroyTestWorld(world);
        owner.dispose();
    }
});

test("a query callback reads a sibling's resident views without changing the traversal World", () => {
    const owner = new World();
    const a = createTestWorld({}, owner);
    const b = createTestWorld({}, owner);
    try {
        for (const x of [2, 4, 6]) {
            const body = createTestBody(a, { type: BodyType.Static, position: { x, y: 0, z: 0 } });
            createSphereShape(a, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
        }
        const body = createTestBody(b, {
            type: BodyType.Dynamic,
            position: { x: 100, y: 0, z: 0 },
        });
        createSphereShape(b, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        const expected = { ...bodyGetTransform(b, body, identity()).p };
        stepTestWorld(a, 1 / 60);
        const query = queryColumns(a);
        const k = query.prepare({ x: 0, y: 0, z: 0 });
        query.translation({ x: 10, y: 0, z: 0 });
        const hits: number[] = [];
        const previous = setQueryCallback(owner, (_kind, _id, data) => {
            expect(bodyGetTransform(b, body, identity()).p).toEqual(expected);
            hits.push(data);
            return 1;
        });
        try {
            k.worldQuery(a.worldId, 2, 1);
            rethrowQueryError(owner);
        } finally {
            setQueryCallback(owner, previous);
        }
        expect(hits).toHaveLength(3);
    } finally {
        destroyTestWorld(b);
        destroyTestWorld(a);
        owner.dispose();
    }
});
