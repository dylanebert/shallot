import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../engine";
import { restore, snapshot } from "./api/snapshot";
import { moveCount } from "./collision/broadphase";
import { BodyType, defaultQueryFilter } from "./common/types";
import { EventKind, eventCount } from "./kernel/eventbuffers";
import { queryColumns } from "./kernel/querycolumns";
import { createSphereShape } from "./shapes/shape";
import { createFilterJoint, defaultJointDef } from "./solver/joint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "./testing";
import { bodySetTransform } from "./world/body";
import { hashWorldState } from "./world/hash";
import type { WorldState } from "./world/world";

setDefaultTimeout(TIMEOUT.node);

function scene(filtered: boolean, owner?: World): WorldState {
    const world = createTestWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false }, owner);
    const mover = createTestBody(world, {
        type: BodyType.Kinematic,
        position: { x: 4, y: 0, z: 0 },
    });
    createSphereShape(world, mover, testShapeDef(), {
        center: { x: 0, y: 0, z: 0 },
        radius: 0.5,
    });
    const probe = createTestBody(world, {
        type: BodyType.Dynamic,
        position: { x: 5.25, y: 0, z: 0 },
    });
    createSphereShape(world, probe, testShapeDef({ enableContactEvents: true }), {
        center: { x: 0, y: 0, z: 0 },
        radius: 0.5,
    });
    const bodies = filtered
        ? [0, 0.25].map((x) => {
              const body = createTestBody(world, {
                  type: BodyType.Dynamic,
                  position: { x, y: 0, z: 0 },
              });
              createSphereShape(world, body, testShapeDef(), {
                  center: { x: 0, y: 0, z: 0 },
                  radius: 0.5,
              });
              return body;
          })
        : [];
    if (filtered)
        createFilterJoint(world, {
            ...defaultJointDef(),
            bodyIdA: bodies[0]!,
            bodyIdB: bodies[1]!,
        });
    stepTestWorld(world, 1 / 60);
    for (let i = 0; i < bodies.length; i++) {
        const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };
        bodySetTransform(world, bodies[i]!, { x: i * 0.25 + 2, y: 0, z: 0 }, rotation);
        bodySetTransform(world, bodies[i]!, { x: i * 0.25, y: 0, z: 0 }, rotation);
    }
    bodySetTransform(world, mover, { x: 5, y: 0, z: 0 }, { v: { x: 0, y: 0, z: 0 }, s: 1 });
    return world;
}

function castRayClosest(
    world: WorldState,
    x = 5,
): {
    hit: boolean;
    fraction: number;
    point: { x: number; y: number; z: number };
    normal: { x: number; y: number; z: number };
} {
    const origin = { x, y: 2, z: 0 };
    const query = queryColumns(world);
    const k = query.prepare(origin, defaultQueryFilter());
    query.translation({ x: 0, y: -4, z: 0 });
    k.worldQuery(world.worldId, 3, 0);
    const f = query.resultF;
    const hit = query.resultU[0] !== 0xffffffff;
    return {
        hit,
        fraction: hit ? f[5]! : 0,
        point: hit
            ? { x: origin.x + f[6]!, y: origin.y + f[7]!, z: origin.z + f[8]! }
            : { x: 0, y: 0, z: 0 },
        normal: hit ? { x: f[9]!, y: f[10]!, z: f[11]! } : { x: 0, y: 0, z: 0 },
    };
}

for (const filtered of [false, true]) {
    test(`empty World construction preserves the owner's pending proxies and moves${filtered ? " and non-colliding joint pair" : ""}`, () => {
        const solo = scene(filtered);
        let expected: bigint;
        try {
            stepTestWorld(solo, 1 / 60);
            expect(eventCount(solo, EventKind.ContactBegin)).toBe(1);
            expected = hashWorldState(solo);
        } finally {
            destroyTestWorld(solo);
        }
        const owner = new World();
        const a = scene(filtered, owner);
        let b: WorldState | undefined;
        try {
            expect(castRayClosest(a).hit).toBe(true);
            b = createTestWorld({}, owner);
            expect(castRayClosest(a).hit).toBe(true);
            stepTestWorld(a, 1 / 60);
            expect(eventCount(a, EventKind.ContactBegin)).toBe(1);
            expect(hashWorldState(a)).toBe(expected);
        } finally {
            if (b) destroyTestWorld(b);
            destroyTestWorld(a);
            owner.dispose();
        }
    });
}

for (const claimed of [false, true]) {
    for (const previouslyOwned of [false, true]) {
        test(`snapshot reinstates resident initialization, proxies, moves, ray and tick hashes: ${claimed ? "claimed" : "unclaimed"}, ${previouslyOwned ? "previous owner" : "fresh kernel"}`, () => {
            const owner = new World();
            if (previouslyOwned) {
                const prior = createTestWorld({}, owner);
                const body = createTestBody(prior, { type: BodyType.Dynamic });
                createSphereShape(prior, body, testShapeDef(), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 0.5,
                });
                stepTestWorld(prior, 1 / 60);
                destroyTestWorld(prior);
            }
            const source = createTestWorld(
                { gravity: { x: 0, y: 0, z: 0 }, enableSleep: false },
                owner,
            );
            if (claimed) {
                const bodies = [0, 0.25, 4, 4.25].map((x) => {
                    const body = createTestBody(source, {
                        type: BodyType.Dynamic,
                        position: { x, y: 0, z: 0 },
                    });
                    createSphereShape(source, body, testShapeDef({ enableContactEvents: true }), {
                        center: { x: 0, y: 0, z: 0 },
                        radius: 0.5,
                    });
                    return body;
                });
                createFilterJoint(source, {
                    ...defaultJointDef(),
                    bodyIdA: bodies[2]!,
                    bodyIdB: bodies[3]!,
                });
            }
            const saved = snapshot(source);
            const proxies = source.broadPhase.trees.map((tree) => tree.proxyCount);
            const moves = moveCount(source.broadPhase);
            expect(proxies).toEqual([0, 0, claimed ? 4 : 0]);
            expect(moves).toBe(claimed ? 4 : 0);
            const expectedRay = castRayClosest(source, 0);
            expect(expectedRay.hit).toBe(claimed);
            const expected: bigint[] = [];
            try {
                for (let tick = 0; tick < 12; tick++) {
                    stepTestWorld(source, 1 / 60);
                    if (tick === 0)
                        expect(eventCount(source, EventKind.ContactBegin)).toBe(claimed ? 1 : 0);
                    expected.push(hashWorldState(source));
                }
            } finally {
                destroyTestWorld(source);
            }
            const target = createTestWorld({}, owner);
            try {
                restore(target, saved);
                expect(target.broadPhase.trees.map((tree) => tree.proxyCount)).toEqual(proxies);
                expect(moveCount(target.broadPhase)).toBe(moves);
                expect(castRayClosest(target, 0)).toEqual(expectedRay);
                for (let tick = 0; tick < expected.length; tick++) {
                    stepTestWorld(target, 1 / 60);
                    expect(hashWorldState(target), `tick ${tick}`).toBe(expected[tick]);
                    if (tick === 0)
                        expect(eventCount(target, EventKind.ContactBegin)).toBe(claimed ? 1 : 0);
                }
            } finally {
                destroyTestWorld(target);
                owner.dispose();
            }
        });
    }
}
