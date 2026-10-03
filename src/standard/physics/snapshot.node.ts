import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import { hash } from "./api";
import { PhysicsWorld } from "./api/world";
import { BodyType } from "./common/types";

setDefaultTimeout(CEILING.node);

function scene(count: number) {
    const world = new PhysicsWorld({ enableSleep: false });
    for (let i = 0; i < count; i++) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: i % 10, y: Math.floor(i / 10), z: 0 },
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.6 });
    }
    world.step(1 / 60);
    return world;
}

test("a World's snapshot bytes do not include a live sibling's 100 bodies", () => {
    const target = scene(4);
    let sibling: PhysicsWorld | undefined;
    try {
        target.castRayClosest({ x: 0, y: 10, z: 0 }, { x: 0, y: -20, z: 0 });
        const before = target.snapshot().bytes;
        sibling = scene(100);
        target.castRayClosest({ x: 0, y: 10, z: 0 }, { x: 0, y: -20, z: 0 });
        expect(target.snapshot().bytes).toEqual(before);
    } finally {
        sibling?.destroy();
        target.destroy();
    }
});

test("restoring a target between sibling ticks preserves every solo tick hash", () => {
    const solo = scene(100);
    const expected: bigint[] = [];
    try {
        for (let tick = 0; tick < 120; tick++) {
            solo.step(1 / 60);
            expected.push(hash(solo));
        }
    } finally {
        solo.destroy();
    }
    const target = scene(4);
    const sibling = scene(100);
    try {
        const saved = target.snapshot();
        for (let tick = 0; tick < 120; tick++) {
            target.step(1 / 60);
            target.restore(saved);
            sibling.step(1 / 60);
            expect(hash(sibling)).toBe(expected[tick]);
        }
    } finally {
        sibling.destroy();
        target.destroy();
    }
});
