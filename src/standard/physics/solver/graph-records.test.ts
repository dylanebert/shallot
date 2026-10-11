import { expect, test } from "bun:test";
import { restore, snapshot } from "../api/snapshot";
import { ContactField, ContactFlags, contactField, setContactField } from "../collision/contact";
import { DYNAMIC_COLOR_COUNT, OVERFLOW_INDEX } from "../common/constants";
import { BodyType } from "../common/types";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { kernel } from "../kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld } from "../testing";
import type { WorldState } from "../world/world";

function graphKernel(world: WorldState) {
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k;
}
function graphContacts(world: WorldState, color: number, scalar = false): Uint32Array {
    const k = graphKernel(world);
    const count = k.graphContactCount(color, +scalar) * (scalar ? 3 : 1);
    return new Uint32Array(k.memory.buffer, k.graphContactPtr(color, +scalar), count);
}

test("kernel graph contact swap-removal fixes both lists, clears body bits and restores packed specs and overflow without touching a sibling", () => {
    const world = createTestWorld();
    const sibling = createTestWorld();
    try {
        const k = graphKernel(world);
        const anchor = createTestBody(world);
        const bodies = Array.from({ length: 130 }, () =>
            createTestBody(world, { type: BodyType.Dynamic }),
        );
        function contact(body: number, mesh: boolean): number {
            const id = k.allocContact();
            setContactField(world, id, ContactField.bodyIdA, bodies[body]!);
            setContactField(world, id, ContactField.bodyIdA + 3, anchor);
            k.allocateManifolds(id, 3);
            setContactField(world, id, ContactField.flags, mesh ? ContactFlags.simMeshContact : 0);
            k.graphAddContact(
                id,
                bodyField(world, bodies[body]!, BodyField.localIndex),
                bodyField(world, anchor, BodyField.localIndex),
            );
            return id;
        }
        const convex = [contact(0, false), contact(1, false), contact(129, false)];
        const scalar = [contact(2, true), contact(3, true)];
        const specs = (color: number) => {
            const words = graphContacts(world, color, true);
            const packed: number[] = [];
            for (let i = 0; i < words.length; i += 3)
                packed.push(words[i]!, words[i + 1]! | (words[i + 2]! << 16));
            return packed;
        };
        const color = OVERFLOW_INDEX - 1;
        expect(Array.from(graphContacts(world, color))).toEqual(convex);
        expect(specs(color)).toEqual([scalar[0], 3 << 16, scalar[1], 3 << 16]);
        for (const [id, mesh] of [
            [convex[0]!, false],
            [scalar[0]!, true],
        ] as const) {
            const a = contactField(world, id, ContactField.bodyIdA);
            k.graphRemoveContact(a, anchor, color, 0, +mesh);
            expect(!!k.graphBodyBit(color, a)).toBe(false);
        }
        expect(Array.from(graphContacts(world, color))).toEqual([convex[2], convex[1]]);
        expect(contactField(world, convex[2]!, ContactField.localIndex)).toBe(0);
        expect(specs(color)).toEqual([scalar[1], 3 << 16]);
        expect(contactField(world, scalar[1]!, ContactField.localIndex)).toBe(0);
        const a = bodies[4]!;
        for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i)
            k.graphAssignColor(a, 1000 + i, BodyType.Dynamic, BodyType.Dynamic);
        for (let i = DYNAMIC_COLOR_COUNT; i < OVERFLOW_INDEX; ++i)
            k.graphAssignColor(a, 0, BodyType.Dynamic, BodyType.Static);
        const overflow = contact(4, false);
        expect(contactField(world, overflow, ContactField.colorIndex)).toBe(OVERFLOW_INDEX);
        expect(specs(OVERFLOW_INDEX)).toEqual([overflow, 3 << 16]);
        expect(!!k.graphBodyBit(OVERFLOW_INDEX, a)).toBe(false);
        const saved = snapshot(world);
        k.graphRemoveContact(a, anchor, OVERFLOW_INDEX, 0, 0);
        createTestBody(sibling, { type: BodyType.Dynamic });
        const siblingKernel = graphKernel(sibling);
        expect(siblingKernel.graphAssignColor(8, 9, BodyType.Dynamic, BodyType.Dynamic)).toBe(0);
        restore(world, saved);
        expect(specs(OVERFLOW_INDEX)).toEqual([overflow, 3 << 16]);
        expect(Array.from(graphContacts(world, color))).toEqual([convex[2], convex[1]]);
        k.bodySetActiveWorld(world.worldId);
        expect(!!k.graphBodyBit(color, bodies[129]!)).toBe(true);
        siblingKernel.bodySetActiveWorld(sibling.worldId);
        expect(!!siblingKernel.graphBodyBit(0, 8)).toBe(true);
    } finally {
        destroyTestWorld(world);
        destroyTestWorld(sibling);
    }
});
