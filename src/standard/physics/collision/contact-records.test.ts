import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import { DIR_BLOCK, M_POINTS, POINT_STRIDE } from "../kernel/contact-layout";
import { EventKind, eventCount, eventId } from "../kernel/eventbuffers";
import { kernel } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/hull";
import { createMesh } from "../shapes/mesh";
import { createHullShape, createMeshShape, createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodySetTransform } from "../world/body";
import { hashWorldState } from "../world/hash";
import { ContactField, ContactFlags, contactCapacity, contactField, contactIds } from "./contact";
import { DIR_STRIDE, MANIFOLD_STRIDE } from "./manifoldstore";
import { updateBroadPhasePairs } from "./pairs";

function eventContacts(world: ReturnType<typeof createTestWorld>, kind: number): number[] {
    return Array.from(
        { length: eventCount(world, kind) },
        (_, i) => eventId(world, kind, i, 4).index1 - 1,
    );
}

test("a steady mesh cluster count reuses and clears its resident block without allocating or freeing", () => {
    const world = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        world.contactRecycleDistance = 0;
        const mesh = createMesh({
            vertices: [
                { x: -4, y: 0, z: -4 },
                { x: 4, y: 0, z: -4 },
                { x: 4, y: 0, z: 4 },
                { x: -4, y: 0, z: 4 },
            ],
            indices: [0, 2, 1, 0, 3, 2],
        })!;
        const ground = createTestBody(world, { type: BodyType.Static });
        createMeshShape(world, ground, testShapeDef(), mesh, { x: 1, y: 1, z: 1 });
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.49, z: 0 },
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        stepTestWorld(world, 1 / 60, 1);
        const ids = contactIds(world);
        expect(ids.length).toBe(1);
        const id = ids[0]!;
        const count = contactField(world, id, ContactField.manifoldCount);
        expect(count).toBe(1);
        const address = world.manifoldStore.dirU[id * DIR_STRIDE + DIR_BLOCK];
        const k = kernel(world.ecsState);
        const operations = k.manifoldAllocatorOperations(world.worldId);
        expect(operations).toBe(1n);
        for (let i = 1; i <= 5; ++i) {
            new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE)[
                M_POINTS + 4 * POINT_STRIDE - 1
            ] = 0x12345678;
            bodySetTransform(
                world,
                body,
                { x: i * 0.1, y: 0.49, z: 0 },
                { v: { x: 0, y: 0, z: 0 }, s: 1 },
            );
            stepTestWorld(world, 1 / 60, 1);
            expect(contactIds(world)).toEqual(ids);
            expect(contactField(world, id, ContactField.manifoldCount)).toBe(count);
            expect(world.manifoldStore.dirU[id * DIR_STRIDE + DIR_BLOCK]).toBe(address);
            expect(
                new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE)[
                    M_POINTS + 4 * POINT_STRIDE - 1
                ],
            ).toBe(0);
            expect(k.manifoldAllocatorOperations(world.worldId)).toBe(operations);
        }
    } finally {
        destroyTestWorld(world);
    }
});

test("manifold-count blocks stay at their addresses through chunk growth, recycle zeroed, and snapshot only contents with contact-id reuse order", () => {
    const world = createTestWorld();
    try {
        const k = kernel(world.ecsState);
        k.bodySetActiveWorld(world.worldId);
        const first = k.allocContact();
        const address = k.allocateManifolds(first, 1);
        new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE).fill(0x12345678);
        const ids = [first];
        for (let i = 0; i < 700; ++i) {
            const id = k.allocContact();
            ids.push(id);
            k.allocateManifolds(id, (i % 2) + 1);
        }
        expect(world.manifoldStore.dirU[first * DIR_STRIDE + DIR_BLOCK]).toBe(address);
        expect(Array.from(new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE))).toEqual(
            new Array(MANIFOLD_STRIDE).fill(0x12345678),
        );
        const oldGeneration = contactField(world, first, ContactField.generation);
        k.freeManifolds(first);
        k.freeContact(first);
        expect(k.allocContact()).toBe(first);
        expect(contactField(world, first, ContactField.generation)).toBe(oldGeneration + 1);
        expect(k.allocateManifolds(first, 1)).toBe(address);
        expect(Array.from(new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE))).toEqual(
            new Array(MANIFOLD_STRIDE).fill(0),
        );
        for (const id of [ids[2]!, ids[7]!]) {
            k.freeManifolds(id);
            k.freeContact(id);
        }
        const saved = snapshot(world);
        expect(k.allocContact()).toBe(ids[7]);
        expect(k.allocContact()).toBe(ids[2]);
        restore(world, saved);
        expect(contactCapacity(world)).toBe(ids.length);
        expect(snapshot(world).bytes).toEqual(saved.bytes);
        expect(k.allocContact()).toBe(ids[7]);
        expect(k.allocContact()).toBe(ids[2]);
        expect(contactField(world, first, ContactField.manifoldCount)).toBe(1);
        const restoredAddress = world.manifoldStore.dirU[first * DIR_STRIDE + DIR_BLOCK];
        expect(
            Array.from(new Uint32Array(k.memory.buffer, restoredAddress, MANIFOLD_STRIDE)),
        ).toEqual(new Array(MANIFOLD_STRIDE).fill(0));
    } finally {
        destroyTestWorld(world);
    }
});

test("the touch bitset crosses word boundaries in contact-id order and restores records, generation and links after destruction and id reuse", () => {
    const world = createTestWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        const ground = createTestBody(world, { type: BodyType.Static });
        createHullShape(
            world,
            ground,
            testShapeDef({ enableContactEvents: true }),
            makeBoxHull(64, 0.5, 2),
        );
        const bodies = Array.from({ length: 40 }, (_, i) => {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: i * 3 - 60, y: 0.9, z: 0 },
            });
            createSphereShape(world, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
            return body;
        });
        updateBroadPhasePairs(world);
        stepTestWorld(world, 1 / 60, 1);
        const ids = contactIds(world);
        expect(ids.length).toBe(40);
        expect(eventContacts(world, EventKind.ContactBegin)).toEqual(ids);
        const generations = ids.map((id) => contactField(world, id, ContactField.generation));
        for (const id of ids)
            expect(
                contactField(world, id, ContactField.flags) & ContactFlags.contactTouchingFlag,
            ).not.toBe(0);
        const saved = snapshot(world);
        const savedHash = hashWorldState(world);
        const replay = () => {
            for (let i = 0; i < bodies.length; ++i)
                bodySetTransform(
                    world,
                    bodies[i]!,
                    { x: i * 3 - 60, y: 50, z: 0 },
                    { v: { x: 0, y: 0, z: 0 }, s: 1 },
                );
            stepTestWorld(world, 1 / 60, 1);
            expect(eventContacts(world, EventKind.ContactEnd)).toEqual(ids);
            expect(contactIds(world)).toEqual([]);
            for (const id of ids) expect(contactField(world, id, ContactField.contactId)).toBe(-1);
            for (let i = 0; i < bodies.length; ++i)
                bodySetTransform(
                    world,
                    bodies[i]!,
                    { x: i * 3 - 60, y: 0.9, z: 0 },
                    { v: { x: 0, y: 0, z: 0 }, s: 1 },
                );
            stepTestWorld(world, 1 / 60, 1);
            expect(contactIds(world)).toEqual(ids);
            expect(eventContacts(world, EventKind.ContactBegin)).toEqual(ids);
            for (const id of ids)
                expect(contactField(world, id, ContactField.generation)).toBe(generations[id]! + 1);
            return hashWorldState(world);
        };
        const expected = replay();
        restore(world, saved);
        expect(hashWorldState(world)).toBe(savedHash);
        expect(ids.map((id) => contactField(world, id, ContactField.generation))).toEqual(
            generations,
        );
        expect(replay()).toBe(expected);
    } finally {
        destroyTestWorld(world);
    }
});
