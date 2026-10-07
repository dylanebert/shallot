import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { ContactField, ContactFlags, contactField, setContactField } from "../collision/contact";
import { DYNAMIC_COLOR_COUNT, OVERFLOW_INDEX } from "../common/constants";
import { kernel } from "../kernel/kernel";
import { addContactToGraph, graphContacts, removeContactFromGraph } from "./graph";

test("kernel graph contact swap-removal fixes both lists, clears body bits and restores packed specs and overflow without touching a sibling", () => {
    const world = new PhysicsWorld();
    const sibling = new PhysicsWorld();
    try {
        const state = world.state;
        const k = kernel(state.ecsState);
        const anchor = world.createBody();
        const bodies = Array.from({ length: 130 }, () =>
            world.createBody({ type: BodyType.Dynamic }),
        );
        function contact(body: number, mesh: boolean): number {
            k.bodySetActiveWorld(state.worldId);
            const id = k.allocContact();
            setContactField(state, id, ContactField.bodyIdA, bodies[body].id.index1 - 1);
            setContactField(state, id, ContactField.bodyIdA + 3, anchor.id.index1 - 1);
            k.allocateManifolds(id, 3);
            setContactField(state, id, ContactField.flags, mesh ? ContactFlags.simMeshContact : 0);
            addContactToGraph(state, id);
            return id;
        }
        const convex = [contact(0, false), contact(1, false), contact(129, false)];
        const scalar = [contact(2, true), contact(3, true)];
        // Compare the same logical spec fields in the frozen assertion's packed spelling.
        const specs = (color: number) => {
            const words = graphContacts(world.state, color, true);
            const packed: number[] = [];
            for (let i = 0; i < words.length; i += 3)
                packed.push(words[i], words[i + 1] | (words[i + 2] << 16));
            return packed;
        };
        const color = OVERFLOW_INDEX - 1;
        expect(Array.from(graphContacts(state, color))).toEqual(convex);
        expect(specs(color)).toEqual([scalar[0], 3 << 16, scalar[1], 3 << 16]);
        for (const [id, mesh] of [
            [convex[0], false],
            [scalar[0], true],
        ] as const) {
            const a = contactField(state, id, ContactField.bodyIdA);
            removeContactFromGraph(state, a, anchor.id.index1 - 1, color, 0, mesh);
            expect(!!k.graphBodyBit(color, a)).toBe(false);
        }
        expect(Array.from(graphContacts(state, color))).toEqual([convex[2], convex[1]]);
        expect(contactField(state, convex[2], ContactField.localIndex)).toBe(0);
        expect(specs(color)).toEqual([scalar[1], 3 << 16]);
        expect(contactField(state, scalar[1], ContactField.localIndex)).toBe(0);
        const a = bodies[4].id.index1 - 1;
        for (let i = 0; i < DYNAMIC_COLOR_COUNT; ++i)
            k.graphAssignColor(a, 1000 + i, BodyType.Dynamic, BodyType.Dynamic);
        // Reserve the remaining high colors for this body, forcing its static contact to overflow.
        for (let i = DYNAMIC_COLOR_COUNT; i < OVERFLOW_INDEX; ++i)
            k.graphAssignColor(a, 0, BodyType.Dynamic, BodyType.Static);
        const overflow = contact(4, false);
        expect(contactField(state, overflow, ContactField.colorIndex)).toBe(OVERFLOW_INDEX);
        expect(specs(OVERFLOW_INDEX)).toEqual([overflow, 3 << 16]);
        expect(!!k.graphBodyBit(OVERFLOW_INDEX, a)).toBe(false);
        const snapshot = world.snapshot();
        removeContactFromGraph(state, a, anchor.id.index1 - 1, OVERFLOW_INDEX, 0, false);
        sibling.createBody({ type: BodyType.Dynamic });
        const siblingKernel = kernel(sibling.state.ecsState);
        siblingKernel.bodySetActiveWorld(sibling.state.worldId);
        expect(siblingKernel.graphAssignColor(8, 9, BodyType.Dynamic, BodyType.Dynamic)).toBe(0);
        world.restore(snapshot);
        expect(specs(OVERFLOW_INDEX)).toEqual([overflow, 3 << 16]);
        expect(Array.from(graphContacts(world.state, color))).toEqual([convex[2], convex[1]]);
        k.bodySetActiveWorld(world.state.worldId);
        expect(!!k.graphBodyBit(color, bodies[129].id.index1 - 1)).toBe(true);
        siblingKernel.bodySetActiveWorld(sibling.state.worldId);
        expect(!!siblingKernel.graphBodyBit(0, 8)).toBe(true);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});
