import { expect, test } from "bun:test";
import { BodyType, hash, makeBoxHull, PhysicsWorld } from "../api";
import { kernel } from "../kernel/kernel";
import { ContactField, ContactFlags, contactCapacity, contactField, contactIds } from "./contact";
import { DIR_STRIDE, MANIFOLD_STRIDE } from "./manifoldstore";

test("manifold-count blocks stay at their addresses through chunk growth, recycle zeroed, and snapshot only contents with contact-id reuse order", () => {
    const world = new PhysicsWorld();
    try {
        const state = world.state,
            k = kernel(state.ecsState);
        k.bodySetActiveWorld(state.worldId);
        const first = k.allocContact();
        const address = k.allocateManifolds(first, 1);
        new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE).fill(0x12345678);
        const ids = [first];
        for (let i = 0; i < 700; ++i) {
            const id = k.allocContact();
            ids.push(id);
            k.allocateManifolds(id, (i % 2) + 1);
        }
        expect(state.manifoldStore.dirU[first * DIR_STRIDE + 8]).toBe(address);
        expect(Array.from(new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE))).toEqual(
            new Array(MANIFOLD_STRIDE).fill(0x12345678),
        );
        const oldGeneration = contactField(state, first, ContactField.generation);
        k.freeManifolds(first);
        k.freeContact(first);
        expect(k.allocContact()).toBe(first);
        expect(contactField(state, first, ContactField.generation)).toBe(oldGeneration + 1);
        expect(k.allocateManifolds(first, 1)).toBe(address);
        expect(Array.from(new Uint32Array(k.memory.buffer, address, MANIFOLD_STRIDE))).toEqual(
            new Array(MANIFOLD_STRIDE).fill(0),
        );
        for (const id of [ids[2], ids[7]]) {
            k.freeManifolds(id);
            k.freeContact(id);
        }
        const saved = world.snapshot();
        expect(k.allocContact()).toBe(ids[7]);
        expect(k.allocContact()).toBe(ids[2]);
        world.restore(saved);
        expect(contactCapacity(state)).toBe(ids.length);
        expect(world.snapshot().bytes).toEqual(saved.bytes);
        expect(k.allocContact()).toBe(ids[7]);
        expect(k.allocContact()).toBe(ids[2]);
        expect(contactField(state, first, ContactField.manifoldCount)).toBe(1);
        const restoredAddress = state.manifoldStore.dirU[first * DIR_STRIDE + 8];
        expect(
            Array.from(new Uint32Array(k.memory.buffer, restoredAddress, MANIFOLD_STRIDE)),
        ).toEqual(new Array(MANIFOLD_STRIDE).fill(0));
    } finally {
        world.destroy();
    }
});

test("the touch bitset crosses word boundaries in contact-id order and restores records, generation and links after destruction and id reuse", () => {
    const world = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    try {
        world
            .createBody({ type: BodyType.Static })
            .createHull({ enableContactEvents: true }, makeBoxHull(64, 0.5, 2));
        const bodies = Array.from({ length: 40 }, (_, i) => {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: i * 3 - 60, y: 0.9, z: 0 },
            });
            body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            return body;
        });
        world.step(1 / 60, 1);
        const ids = contactIds(world.state);
        expect(ids.length).toBe(40);
        expect(
            world.getContactEvents().beginEvents.map((event) => event.contact.id.index1 - 1),
        ).toEqual(ids);
        const generations = ids.map((id) => contactField(world.state, id, ContactField.generation));
        for (const id of ids)
            expect(
                contactField(world.state, id, ContactField.flags) &
                    ContactFlags.contactTouchingFlag,
            ).not.toBe(0);
        const saved = world.snapshot();
        const savedHash = hash(world);
        const replay = () => {
            for (let i = 0; i < bodies.length; ++i)
                bodies[i].setTransform(
                    { x: i * 3 - 60, y: 50, z: 0 },
                    { v: { x: 0, y: 0, z: 0 }, s: 1 },
                );
            world.step(1 / 60, 1);
            expect(
                world.getContactEvents().endEvents.map((event) => event.contact.id.index1 - 1),
            ).toEqual(ids);
            expect(contactIds(world.state)).toEqual([]);
            for (const id of ids)
                expect(contactField(world.state, id, ContactField.contactId)).toBe(-1);
            for (let i = 0; i < bodies.length; ++i)
                bodies[i].setTransform(
                    { x: i * 3 - 60, y: 0.9, z: 0 },
                    { v: { x: 0, y: 0, z: 0 }, s: 1 },
                );
            world.step(1 / 60, 1);
            expect(contactIds(world.state)).toEqual(ids);
            expect(
                world.getContactEvents().beginEvents.map((event) => event.contact.id.index1 - 1),
            ).toEqual(ids);
            for (const id of ids)
                expect(contactField(world.state, id, ContactField.generation)).toBe(
                    generations[id] + 1,
                );
            return hash(world);
        };
        const expected = replay();
        world.restore(saved);
        expect(hash(world)).toBe(savedHash);
        expect(ids.map((id) => contactField(world.state, id, ContactField.generation))).toEqual(
            generations,
        );
        expect(replay()).toBe(expected);
    } finally {
        world.destroy();
    }
});
