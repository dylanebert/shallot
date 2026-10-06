import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld } from "../api";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    islandArrayCount,
    islandArrayGet,
    islandField,
    islandKernel,
} from "../kernel/islandcolumns";
import { splitIsland } from "./island";

test("island fix borrowing follows vector relocation and memory growth", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 32 }, () =>
            world.createBody({ type: BodyType.Dynamic }),
        );
        const records = bodies.map((b) => b.id.index1 - 1);
        for (let i = 1; i < bodies.length; ++i) {
            if (i === 16) islandKernel(world.state).memory.grow(1);
            world.createDistanceJoint(bodies[0], bodies[i], { length: 1 });
            for (let j = 0; j <= i; ++j) {
                const body = records[j];
                expect(bodyField(world.state, body, BodyField.islandId)).toBe(
                    bodyField(world.state, records[0], BodyField.islandId),
                );
                expect(
                    islandArrayGet(
                        world.state,
                        bodyField(world.state, body, BodyField.islandId),
                        0,
                        bodyField(world.state, body, BodyField.islandIndex),
                    ),
                ).toBe(bodyField(world.state, body, BodyField.id));
            }
            const joint = world.state.joints[i - 1];
            expect(islandArrayGet(world.state, joint.islandId, 2, joint.islandIndex)).toBe(
                joint.jointId,
            );
        }
    } finally {
        world.destroy();
    }
});

test("island split preserves link membership, fixes body and joint slots, and restores the id pool", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const bodies = Array.from({ length: 4 }, (_, x) =>
            world.createBody({ type: BodyType.Dynamic, position: { x, y: 0, z: 0 } }),
        );
        const joints = bodies
            .slice(1)
            .map((b, i) => world.createDistanceJoint(bodies[i], b, { length: 1 }));
        const records = bodies.map((b) => b.id.index1 - 1);
        const base = bodyField(world.state, records[0], BodyField.islandId);
        expect(islandArrayCount(world.state, base, 0)).toBe(4);
        joints[1].destroy();
        const saved = world.snapshot();
        const verify = () => {
            splitIsland(world.state, base);
            expect(bodyField(world.state, records[0], BodyField.islandId)).toBe(
                bodyField(world.state, records[1], BodyField.islandId),
            );
            expect(bodyField(world.state, records[2], BodyField.islandId)).toBe(
                bodyField(world.state, records[3], BodyField.islandId),
            );
            expect(bodyField(world.state, records[0], BodyField.islandId)).not.toBe(
                bodyField(world.state, records[2], BodyField.islandId),
            );
            for (const body of records) {
                expect(
                    islandArrayGet(
                        world.state,
                        bodyField(world.state, body, BodyField.islandId),
                        0,
                        bodyField(world.state, body, BodyField.islandIndex),
                    ),
                ).toBe(bodyField(world.state, body, BodyField.id));
                expect(
                    islandField(world.state, bodyField(world.state, body, BodyField.islandId), 3),
                ).toBe(0);
            }
            for (const joint of world.state.joints) {
                if (joint.islandId === -1) continue;
                expect(islandArrayGet(world.state, joint.islandId, 2, joint.islandIndex)).toBe(
                    joint.jointId,
                );
                expect(islandArrayCount(world.state, joint.islandId, 2)).toBe(1);
            }
            return records.map((b) => bodyField(world.state, b, BodyField.islandId));
        };
        const ids = verify();
        world.restore(saved);
        for (let i = 0; i < records.length; ++i) records[i] = bodies[i].id.index1 - 1;
        expect(verify()).toEqual(ids);
    } finally {
        world.destroy();
    }
});
