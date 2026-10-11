import { expect, test } from "bun:test";
import { World } from "../../../engine";
import {
    type BodyDef,
    BodyType,
    defaultBodyDef,
    defaultShapeDef,
    defaultWorldDef,
    type ShapeDef,
} from "../common/types";
import { kernel } from "../kernel/kernel";
import { createSphereShape } from "../shapes/shape";
import { createFilterJoint, defaultJointDef } from "../solver/joint";
import { bodyGetLinearVelocity, bodySetLinearVelocity, createBody } from "../world/body";
import { createWorld, destroyWorld, getWorld } from "../world/world";
import { restore, snapshot } from "./snapshot";

function makeWorld(owner?: World) {
    const id = createWorld(owner, defaultWorldDef());
    return getWorld(id)!;
}

function makeBody(world: ReturnType<typeof makeWorld>, def: Partial<BodyDef> = {}) {
    return createBody(world, { ...defaultBodyDef(), ...def });
}

function makeSphere(
    world: ReturnType<typeof makeWorld>,
    body: number,
    def: Partial<ShapeDef> = {},
) {
    return createSphereShape(
        world,
        body,
        { ...defaultShapeDef(), ...def },
        { center: { x: 0, y: 0, z: 0 }, radius: 1 },
    )!;
}

function makeFilterJoint(
    world: ReturnType<typeof makeWorld>,
    a: number,
    b: number,
    userData: unknown,
) {
    const def = { ...defaultJointDef(), bodyIdA: a, bodyIdB: b, userData };
    return createFilterJoint(world, def).joint;
}

test("restore preserves opaque body, shape and joint userData identity and captured associations", () => {
    class Wiring {
        #value = 7;
        value() {
            return this.#value;
        }
    }
    const world = makeWorld();
    try {
        const value = new Wiring();
        const a = makeBody(world, { userData: value });
        const b = makeBody(world, { type: BodyType.Dynamic });
        const shape = makeSphere(world, a, { userData: value });
        const joint = makeFilterJoint(world, a, b, value);
        const saved = snapshot(world);
        world.bodyUserData[a] = {};
        world.shapeUserData[shape] = {};
        world.jointUserData[joint] = {};
        restore(world, saved);
        for (const userData of [
            world.bodyUserData[a],
            world.shapeUserData[shape],
            world.jointUserData[joint],
        ]) {
            expect(userData).toBe(value);
            expect((userData as Wiring).value()).toBe(7);
        }
        world.bodyUserData[a] = null;
        restore(world, saved);
        expect(world.bodyUserData[a]).toBe(value);
    } finally {
        destroyWorld(world);
    }
});

test("restore refuses a destroyed target even when a sibling is the only live World", () => {
    const target = makeWorld();
    const sibling = makeWorld();
    let targetDestroyed = false;
    try {
        makeBody(target, { type: BodyType.Kinematic, linearVelocity: { x: 1, y: 0, z: 0 } });
        const siblingBody = makeBody(sibling, {
            type: BodyType.Kinematic,
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        const saved = snapshot(target);
        bodySetLinearVelocity(sibling, siblingBody, { x: 8, y: 0, z: 0 });
        destroyWorld(target);
        targetDestroyed = true;

        let restoreError: unknown;
        try {
            restore(target, saved);
        } catch (error) {
            restoreError = error;
        }
        const velocity = { x: 0, y: 0, z: 0 };
        expect(bodyGetLinearVelocity(sibling, siblingBody, velocity)).toBe(velocity);
        expect(velocity).toEqual({ x: 8, y: 0, z: 0 });
        expect(String(restoreError)).toContain("target World is not live");
    } finally {
        destroyWorld(sibling);
        if (!targetDestroyed) destroyWorld(target);
    }
});

test("restoring a snapshot rebinds tree views before another proxy mutation", () => {
    const world = makeWorld(new World());
    try {
        makeSphere(world, makeBody(world, { type: BodyType.Dynamic }));
        const saved = snapshot(world);
        restore(world, saved);
        const broad = world.broadPhase;
        expect(broad.store.trees).toBe(broad.trees);
        expect(broad.trees[BodyType.Dynamic].ni.buffer).toBe(kernel(world.ecsState).memory.buffer);
        const second = makeBody(world, { type: BodyType.Dynamic, position: { x: 4, y: 0, z: 0 } });
        makeSphere(world, second);
        expect(broad.trees[BodyType.Dynamic].proxyCount).toBe(2);
    } finally {
        destroyWorld(world);
    }
});

test("restoring one World leaves a sibling World on the same kernel unchanged", () => {
    const target = makeWorld();
    const sibling = makeWorld();
    try {
        makeBody(target, { type: BodyType.Kinematic, linearVelocity: { x: 1, y: 0, z: 0 } });
        const siblingBody = makeBody(sibling, {
            type: BodyType.Kinematic,
            linearVelocity: { x: 1, y: 0, z: 0 },
        });
        const saved = snapshot(target);
        bodySetLinearVelocity(sibling, siblingBody, { x: 8, y: 0, z: 0 });

        expect(() => restore(target, saved)).not.toThrow();
        const velocity = { x: 0, y: 0, z: 0 };
        bodyGetLinearVelocity(sibling, siblingBody, velocity);
        expect(velocity).toEqual({ x: 8, y: 0, z: 0 });
    } finally {
        destroyWorld(sibling);
        destroyWorld(target);
    }
});
