import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { restore, snapshot } from "./api/snapshot";
import { BodyType, defaultQueryFilter } from "./common/types";
import { queryColumns } from "./kernel/querycolumns";
import { createSphereShape } from "./shapes/shape";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { hashWorldState } from "./world/hash";
import type { WorldState } from "./world/world";

setDefaultTimeout(CEILING.node);

const origin = { x: 0, y: 10, z: 0 };
const translation = { x: 0, y: -20, z: 0 };

function castRayClosest(world: WorldState): void {
    const query = queryColumns(world);
    const k = query.prepare(origin, defaultQueryFilter());
    query.translation(translation);
    k.worldQuery(world.worldId, 3, 0);
}

function scene(count: number): WorldState {
    const world = createTestWorld({ enableSleep: false });
    for (let i = 0; i < count; i++) {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: i % 10, y: Math.floor(i / 10), z: 0 },
        });
        createSphereShape(world, body, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.6,
        });
    }
    stepTestWorld(world, 1 / 60);
    return world;
}

test("a World's snapshot bytes do not include a live sibling's 100 bodies", () => {
    const target = scene(4);
    let sibling: WorldState | undefined;
    try {
        castRayClosest(target);
        const before = snapshot(target).bytes;
        sibling = scene(100);
        castRayClosest(target);
        expect(snapshot(target).bytes).toEqual(before);
    } finally {
        if (sibling) destroyTestWorld(sibling);
        destroyTestWorld(target);
    }
});

test("restoring a target between sibling ticks preserves every solo tick hash", () => {
    const solo = scene(100);
    const expected: bigint[] = [];
    try {
        for (let tick = 0; tick < 120; tick++) {
            stepTestWorld(solo, 1 / 60);
            expected.push(hashWorldState(solo));
        }
    } finally {
        destroyTestWorld(solo);
    }
    const target = scene(4);
    const sibling = scene(100);
    try {
        const saved = snapshot(target);
        for (let tick = 0; tick < 120; tick++) {
            stepTestWorld(target, 1 / 60);
            restore(target, saved);
            stepTestWorld(sibling, 1 / 60);
            expect(hashWorldState(sibling)).toBe(expected[tick]);
        }
    } finally {
        destroyTestWorld(sibling);
        destroyTestWorld(target);
    }
});
