import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { hash } from "./api";
import { PhysicsWorld } from "./api/world";
import { BodyType } from "./common/types";
import { makeBoxHull } from "./shapes/hull";

setDefaultTimeout(CEILING.node);

const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };

function scene(offset = 0, extra = 0) {
    const world = new PhysicsWorld({ enableSleep: false, enableContinuous: false });
    const floor = world.createBody({
        type: BodyType.Kinematic,
        position: { x: offset, y: -0.5, z: 0 },
    });
    floor.createHull({}, makeBoxHull(8, 0.5, 8));
    const balls = [0, 3, 3.25].map((x) => {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: offset + x, y: 1, z: 0 },
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        return body;
    });
    // Identical overlapping bodies must not repel each other across an ownership transfer.
    world.createFilterJoint(balls[1], balls[2]);
    for (let i = 0; i < extra; i++) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: offset + 20 + i * 3, y: 2, z: 0 },
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world.createFilterJoint(balls[1], body);
    }
    return { world, floor, offset };
}

function advance(subject: ReturnType<typeof scene>, tick: number) {
    subject.floor.setTransform({ x: subject.offset + (tick & 1), y: -0.5, z: 0 }, rotation);
    subject.world.step(1 / 60, 4);
    return {
        hash: hash(subject.world),
        contacts: subject.world.getContactEvents().beginEvents.length,
        moves: subject.world.getBodyEvents().count,
    };
}

test("constructing an empty PhysicsWorld cannot clear another World's pending proxies and moves", () => {
    const a = scene();
    let b: PhysicsWorld | undefined;
    try {
        const before = hash(a.world);
        expect(a.world.castRayClosest({ x: 0, y: 4, z: 0 }, { x: 0, y: -8, z: 0 }).hit).toBe(true);
        b = new PhysicsWorld();
        expect(a.world.castRayClosest({ x: 0, y: 4, z: 0 }, { x: 0, y: -8, z: 0 }).hit).toBe(true);
        expect(hash(a.world)).toBe(before);
    } finally {
        b?.destroy();
        a.world.destroy();
    }
});

test("interleaved PhysicsWorlds preserve the solo bit-exact trajectory with proxy moves, contacts and non-colliding joint pairs", () => {
    const solo = scene();
    const expected = [];
    try {
        for (let tick = 0; tick < 48; tick++) expected.push(advance(solo, tick));
    } finally {
        solo.world.destroy();
    }
    const a = scene();
    let b: ReturnType<typeof scene> | undefined;
    try {
        // Creation itself, before B ever steps, must not mutate A's pending resident state.
        b = scene(100, 40);
        for (let tick = 0; tick < expected.length; tick++) {
            advance(b, tick);
            expect(advance(a, tick), `tick ${tick}`).toEqual(expected[tick]);
        }
    } finally {
        b?.world.destroy();
        a.world.destroy();
    }
});
