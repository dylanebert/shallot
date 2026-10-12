import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { restore, snapshot } from "../api/snapshot";
import { BodyType, defaultShapeDef } from "../common/types";
import { kernel } from "../kernel/kernel";
import { createMesh } from "../shapes/mesh";
import { createMeshShape, createSphereShape } from "../shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodySetAwake, destroyBody } from "../world/body";
import { hashWorldState } from "../world/hash";
import { ContactField, contactField, contactIds } from "./contact";

setDefaultTimeout(CEILING.node);

test("mesh contact cache replays through sleep, wake, contact destruction and recycled ids", () => {
    const world = createTestWorld();
    const k = kernel(world.ecsState);
    const id = world.worldId;
    try {
        const ground = createTestBody(world);
        const mesh = createMesh({
            vertices: [
                { x: -4, y: 0, z: -4 },
                { x: 4, y: 0, z: -4 },
                { x: 4, y: 0, z: 4 },
                { x: -4, y: 0, z: 4 },
            ],
            indices: [0, 2, 1, 0, 3, 2],
        })!;
        createMeshShape(world, ground, defaultShapeDef(), mesh, { x: 1, y: 1, z: 1 });
        const spawn = () => {
            const body = createTestBody(world, {
                type: BodyType.Dynamic,
                position: { x: 0, y: 0.5, z: 0 },
            });
            createSphereShape(world, body, testShapeDef(), {
                center: { x: 0, y: 0, z: 0 },
                radius: 0.5,
            });
            return body;
        };
        const body = spawn();
        for (let i = 0; i < 30; ++i) stepTestWorld(world, 1 / 60);
        expect(
            contactIds(world).some(
                (contact) => contactField(world, contact, ContactField.manifoldCount) > 0,
            ),
        ).toBe(true);
        expect(k.meshCacheCapacity(id)).toBeGreaterThan(0);
        const saved = snapshot(world);
        const run = () => {
            const hashes: bigint[] = [];
            bodySetAwake(world, body, false);
            for (let i = 0; i < 3; ++i) {
                stepTestWorld(world, 1 / 60);
                hashes.push(hashWorldState(world));
            }
            bodySetAwake(world, body, true);
            for (let i = 0; i < 3; ++i) {
                stepTestWorld(world, 1 / 60);
                hashes.push(hashWorldState(world));
            }
            destroyBody(world, body);
            spawn();
            for (let i = 0; i < 10; ++i) {
                stepTestWorld(world, 1 / 60);
                hashes.push(hashWorldState(world));
            }
            return hashes;
        };
        const expected = run();
        restore(world, saved);
        expect(run()).toEqual(expected);
    } finally {
        destroyTestWorld(world);
    }
    expect(k.meshCacheCapacity(id)).toBe(0);
});
