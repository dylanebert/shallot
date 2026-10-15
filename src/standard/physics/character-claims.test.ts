import { expect, test } from "bun:test";
import { Body, BodyType, PhysicsPlugin, Shape, ShapeKind } from "../../core/physics";
import { GlobalTransform, Transform, TransformPlugin } from "../../core/transform";
import { Time, World } from "../../engine";
import { characterScratch } from "./character";
import {
    Character,
    CharacterPlugin,
    GroundState,
    StandardPhysicsPlugin,
    setLinearVelocity,
} from "./index";
import { shapeBodyId } from "./kernel/filtercolumns";
import { physicsBodyIndex, physicsState } from "./runtime";

async function scene() {
    const world = new World();
    for (const plugin of [TransformPlugin, PhysicsPlugin, CharacterPlugin])
        for (const registration of plugin.components!) world.registry.register(registration);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!])
        world.addSystem(system);
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, 0.9, 0] });
    world.add(eid, Body, { type: BodyType.Kinematic });
    world.add(eid, Shape, { kind: ShapeKind.Capsule });
    world.storage(Shape).capsuleCenter1.set(eid, 0, -0.5, 0, 0);
    world.storage(Shape).capsuleCenter2.set(eid, 0, 0.5, 0, 0);
    world.storage(Shape).radius.set(eid, 0.3);
    world.add(eid, Character);
    return {
        world,
        eid,
        dispose: () => {
            StandardPhysicsPlugin.dispose!(world);
            world.dispose();
        },
    };
}

test("the kinematic capsule mirrors the resolved endpoint without a second move or the rigid solver's speed cap", async () => {
    const { world, eid, dispose } = await scene();
    try {
        world.storage(Transform).translation.set(eid, 0, 100, 0);
        const c = world.storage(Character);
        c.velocity.set(eid, 1200, 0, 0, 0);
        world.step(Time.FIXED_DT);
        const endpoint = Math.fround(1200 * Math.fround(Time.FIXED_DT));
        expect(world.storage(Transform).translation.x.get(eid)).toBe(endpoint);
        expect(world.storage(Body).type.get(eid)).toBe(BodyType.Kinematic);
        const position = world.storage(GlobalTransform).translation;
        expect([position.x.get(eid), position.y.get(eid), position.z.get(eid)]).toEqual([
            endpoint,
            100,
            0,
        ]);
        expect(c.velocity.x.get(eid)).toBe(1200);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(endpoint);
        for (let tick = 0; tick < 60; ++tick) {
            c.velocity.set(eid, 3, 0, 0, 0);
            world.step(Time.FIXED_DT);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(
                world.storage(Transform).translation.x.get(eid),
            );
        }
    } finally {
        dispose();
    }
});

test("ascending platform carry stays grounded, but a jump relative to that platform leaves ground", async () => {
    const { world, eid, dispose } = await scene();
    try {
        const platform = floor(world, BodyType.Kinematic);
        StandardPhysicsPlugin.systems![0].update!(world);
        setLinearVelocity(world, platform, { x: 0, y: 2, z: 0 });
        const c = world.storage(Character);
        c.velocity.set(eid, 0, 2, 0, 0);
        world.step(Time.FIXED_DT);
        expect(c.groundState.get(eid)).toBe(GroundState.OnGround);
        expect(c.groundVelocity.y.get(eid)).toBe(2);
        c.velocity.set(eid, 0, 7, 0, 0);
        world.step(Time.FIXED_DT);
        expect(c.groundState.get(eid)).toBe(GroundState.InAir);
        expect(c.groundVelocity.y.get(eid)).toBe(0);
        expect(c.pogoVelocity.get(eid)).toBe(0);
    } finally {
        dispose();
    }
});

const floor = (world: World, type: BodyType = BodyType.Static) => {
    const eid = world.create();
    world.add(eid, Transform, { translation: [0, -0.5, 0] });
    world.add(eid, Body, { type });
    world.add(eid, Shape, { scale: [40, 1, 40, 0] });
    return eid;
};

test("a stepped character excludes every shape on its own body from planes and its pogo ray", async () => {
    const { world, eid, dispose } = await scene();
    try {
        floor(world);
        StandardPhysicsPlugin.systems![0].update!(world);
        // The ray starts inside the capsule, so a second own shape makes exclusion observable on the ray too.
        for (const [x, y] of [
            [0, -0.7],
            [0.8, 0],
        ]) {
            const shape = world.create();
            world.add(shape, Shape, { body: eid, kind: ShapeKind.Sphere });
            world.storage(Shape).sphereCenter.set(shape, x, y, 0, 0);
            world.storage(Shape).radius.set(shape, 0.1);
        }
        world.storage(Character).velocity.set(eid, 36, -12, 0, 0);
        world.step(Time.FIXED_DT);
        const s = world.resource(characterScratch);
        expect(s.groundShape).toBeGreaterThanOrEqual(0);
        const state = physicsState(world)!;
        const own = physicsBodyIndex(world, eid);
        expect(shapeBodyId(state, s.groundShape)).not.toBe(own);
        expect(s.count).toBeGreaterThan(0);
        for (let i = 0; i < s.count; ++i) expect(shapeBodyId(state, s.shapes[i])).not.toBe(own);
        expect(world.storage(Character).groundNormal.y.get(eid)).toBe(1);
        expect(world.storage(Transform).translation.x.get(eid)).toBeCloseTo(0.6, 6);
    } finally {
        dispose();
    }
});

test("ground above maxSlope reports steep ground and its normal", async () => {
    const { world, eid, dispose } = await scene();
    try {
        const slope = world.create();
        world.add(slope, Transform, {
            rotation: [0, 0, Math.sin(0.25), Math.cos(0.25)],
        });
        world.add(slope, Body);
        world.add(slope, Shape, { scale: [40, 0.2, 40, 0] });
        world.storage(Transform).translation.set(eid, 0, 1.8, 0);
        const c = world.storage(Character);
        c.maxSlope.set(eid, 0.1);
        c.velocity.set(eid, 3, -3, 1, 0);
        world.step(Time.FIXED_DT);
        expect(c.groundState.get(eid)).toBe(GroundState.OnSteepGround);
        expect(c.groundNormal.y.get(eid)).toBeGreaterThan(0);
        expect(c.groundNormal.y.get(eid)).toBeLessThan(Math.cos(c.maxSlope.get(eid)));
        c.maxSlope.set(eid, Math.PI / 4);
        c.velocity.set(eid, 0, -3, 0, 0);
        world.step(Time.FIXED_DT);
        expect(c.groundState.get(eid)).toBe(GroundState.OnGround);
    } finally {
        dispose();
    }
});

test("a sideways-moving kinematic platform reports velocity but does not carry the character", async () => {
    const { world, eid, dispose } = await scene();
    try {
        const platform = floor(world, BodyType.Kinematic);
        StandardPhysicsPlugin.systems![0].update!(world);
        setLinearVelocity(world, platform, { x: 2, y: 0, z: 0 });
        const c = world.storage(Character);
        for (let tick = 0; tick < 12; ++tick) {
            c.velocity.set(eid, 0, -1, 0, 0);
            world.step(Time.FIXED_DT);
            expect(c.groundState.get(eid)).toBe(GroundState.OnGround);
            expect([
                c.groundVelocity.x.get(eid),
                c.groundVelocity.y.get(eid),
                c.groundVelocity.z.get(eid),
            ]).toEqual([2, 0, 0]);
            expect(world.storage(Transform).translation.x.get(eid)).toBe(0);
            expect(c.velocity.x.get(eid)).toBe(0);
        }
        expect(world.storage(GlobalTransform).translation.x.get(platform)).toBeGreaterThan(0);
    } finally {
        dispose();
    }
});

async function walkInto(stepHeight: number, steepWalls: number): Promise<number> {
    const { world, eid, dispose } = await scene();
    try {
        floor(world);
        const step = world.create();
        world.add(step, Transform, { translation: [0, stepHeight / 2, -3] });
        world.add(step, Body, { type: BodyType.Static });
        world.add(step, Shape, { scale: [8, stepHeight, 2, 0] });
        const c = world.storage(Character);
        c.pogoRest.set(eid, 1 + 0.05 / 0.3);
        c.steepWalls.set(eid, steepWalls);
        let furthest = 0;
        for (let tick = 0; tick < 120; ++tick) {
            const grounded = c.groundState.get(eid) === GroundState.OnGround;
            c.velocity.set(
                eid,
                0,
                grounded ? 0 : c.velocity.y.get(eid) - 30 * Time.FIXED_DT,
                -4,
                0,
            );
            world.step(Time.FIXED_DT);
            furthest = Math.min(furthest, world.storage(GlobalTransform).translation.z.get(eid));
        }
        return furthest;
    } finally {
        dispose();
    }
}

test("without steep walls a capsule rides its rounding up a step edge", async () => {
    expect(await walkInto(0.15, 0)).toBeLessThan(-4);
});

test("steep walls stop a capsule riding its rounding up that step edge", async () => {
    expect(await walkInto(0.15, 1)).toBeGreaterThan(-2);
});
