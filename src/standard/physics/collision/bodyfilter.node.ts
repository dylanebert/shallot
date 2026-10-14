import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { World } from "../../../engine";
import { restore, snapshot } from "../api/snapshot";
import { BodyType } from "../common/types";
import { init, kernel, shutdown } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import {
    createFilterJoint,
    defaultJointDef,
    destroyJointInternal,
    setJointCollideConnected,
} from "../solver/joint";
import {
    createTestBody,
    createTestWorld,
    destroyTestWorld,
    stepTestWorld,
    testShapeDef,
} from "../testing";
import { bodyGetTransform, destroyBody } from "../world/body";
import type { WorldState } from "../world/world";
import { worldSetGravity } from "../world/world";

setDefaultTimeout(TIMEOUT.node);

const zero = { x: 0, y: 0, z: 0 };
const pose = () => ({ p: { ...zero }, q: { v: { ...zero }, s: 1 } });

function filterJoint(
    world: WorldState,
    bodyIdA: number,
    bodyIdB: number,
    collideConnected = false,
): number {
    return createFilterJoint(world, {
        ...defaultJointDef(),
        bodyIdA,
        bodyIdB,
        collideConnected,
    }).joint;
}

test("fast bodies and bullets pass static shapes jointed without collideConnected, but stop when enabled", () => {
    for (const isBullet of [false, true]) {
        for (const collideConnected of [false, true]) {
            const world = createTestWorld();
            try {
                worldSetGravity(world, zero);
                const target = createTestBody(world);
                createSphereShape(world, target, testShapeDef(), { center: zero, radius: 1 });
                const fast = createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: { x: -4, y: 0, z: 0 },
                    linearVelocity: { x: 480, y: 0, z: 0 },
                    isBullet,
                });
                createSphereShape(world, fast, testShapeDef(), { center: zero, radius: 0.25 });
                const free = createTestBody(world, {
                    type: BodyType.Dynamic,
                    position: { x: -4, y: 10, z: 0 },
                    linearVelocity: { x: 480, y: 0, z: 0 },
                    isBullet,
                });
                createSphereShape(world, free, testShapeDef(), { center: zero, radius: 0.25 });
                filterJoint(world, fast, target, collideConnected);
                stepTestWorld(world, 1 / 60, 1);
                expect(bodyGetTransform(world, free, pose()).p.x).toBeGreaterThan(1);
                if (collideConnected)
                    expect(bodyGetTransform(world, fast, pose()).p.x).toBeLessThan(-1);
                else
                    expect(bodyGetTransform(world, fast, pose()).p.x).toBe(
                        bodyGetTransform(world, free, pose()).p.x,
                    );
            } finally {
                destroyTestWorld(world);
            }
        }
    }
});

test("direct joint filters survive parallel joints, toggles, destroy, growth and snapshot restore without filtering transitive neighbors", async () => {
    const owner = new World();
    await init(owner, { threads: 0 });
    const world = createTestWorld({}, owner);
    try {
        const a = createTestBody(world, { type: BodyType.Dynamic });
        const b = createTestBody(world);
        const c = createTestBody(world);
        const allows = (x: number, y: number) => {
            const k = kernel(owner);
            k.bodySetActiveWorld(world.worldId);
            return k.bodyShouldBodiesCollide(x, y) !== 0;
        };
        expect(allows(a, b)).toBe(true);
        const first = filterJoint(world, a, b);
        const second = filterJoint(world, b, a);
        filterJoint(world, b, c);
        filterJoint(world, a, c, true);
        expect(allows(a, b)).toBe(false);
        expect(allows(b, a)).toBe(false);
        expect(allows(a, c)).toBe(true);
        setJointCollideConnected(world, first, true);
        setJointCollideConnected(world, first, true);
        expect(allows(a, b)).toBe(false);
        destroyJointInternal(world, second, true);
        expect(allows(a, b)).toBe(true);
        setJointCollideConnected(world, first, false);
        expect(allows(a, b)).toBe(false);
        const saved = snapshot(world);
        destroyJointInternal(world, first, true);
        expect(allows(a, b)).toBe(true);
        const others = Array.from({ length: 40 }, () => createTestBody(world));
        for (const body of others.reverse()) filterJoint(world, a, body);
        for (const body of others) expect(allows(a, body)).toBe(false);
        expect(allows(a, b)).toBe(true);
        restore(world, saved);
        expect(allows(a, b)).toBe(false);
        expect(allows(a, c)).toBe(true);
        setJointCollideConnected(world, first, true);
        expect(allows(a, b)).toBe(true);
        setJointCollideConnected(world, first, false);
        destroyBody(world, b);
        const replacement = createTestBody(world);
        expect(allows(a, replacement)).toBe(true);
        expect(allows(c, replacement)).toBe(false);
    } finally {
        destroyTestWorld(world);
        await shutdown(owner);
        owner.dispose();
    }
});
