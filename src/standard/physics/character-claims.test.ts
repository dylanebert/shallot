import { expect, test } from "bun:test";
import { Body, PhysicsPlugin, ShapeKind } from "../../core/physics";
import { GlobalTransform, Time, World } from "../../engine";
import { BodyType, createHull } from "./api";
import { characterScratch } from "./character";
import {
    Character,
    CharacterPlugin,
    GroundState,
    physicsWorld,
    StandardPhysicsPlugin,
} from "./index";
import lane from "./oracle/box3d/47d7f7cc7e091142c08d11dc7d2e493c5d34f536/v7/cases.json";

async function scene() {
    const world = new World();
    for (const plugin of [PhysicsPlugin, CharacterPlugin])
        for (const registration of plugin.components!) world.registry.register(registration);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of [...StandardPhysicsPlugin.systems!, ...CharacterPlugin.systems!])
        world.addSystem(system);
    const eid = world.create();
    world.add(eid, Body, {
        type: BodyType.Kinematic,
        shape: ShapeKind.Capsule,
        position: [0, 0.9, 0, 0],
        halfExtents: [0, 0.5, 0, 0.3],
    });
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
        world.storage(Body).position.set(eid, 0, 100, 0, 0);
        const c = world.storage(Character);
        c.velocity.set(eid, 1200, 0, 0, 0);
        world.step(Time.FIXED_DT);
        expect(world.storage(Body).position.x.get(eid)).toBe(20);
        const handle = physicsWorld(world)!.getBody(eid)!;
        expect(handle.getType()).toBe(BodyType.Kinematic);
        expect(handle.getPosition()).toEqual({ x: 20, y: 100, z: 0 });
        expect(c.velocity.x.get(eid)).toBe(1200);
        expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(20);
        for (let tick = 0; tick < 120; ++tick) {
            c.velocity.set(eid, 3, 0, 0, 0);
            world.step(Time.FIXED_DT);
            expect(world.storage(GlobalTransform).translation.x.get(eid)).toBe(
                handle.getPosition().x,
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
        physicsWorld(world)!.getBody(platform)!.setLinearVelocity({ x: 0, y: 2, z: 0 });
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
    world.add(eid, Body, { type, position: [0, -0.5, 0, 0], halfExtents: [20, 0.5, 20, 0] });
    return eid;
};

test("a stepped character excludes every shape on its own body from planes and its pogo ray", async () => {
    const { world, eid, dispose } = await scene();
    try {
        floor(world);
        StandardPhysicsPlugin.systems![0].update!(world);
        const physics = physicsWorld(world)!;
        const own = physics.getBody(eid)!;
        // The ray starts inside the capsule, so a second own shape makes exclusion observable on the ray too.
        own.createSphere({}, { center: { x: 0, y: -0.7, z: 0 }, radius: 0.1 });
        own.createSphere({}, { center: { x: 0.8, y: 0, z: 0 }, radius: 0.1 });
        world.storage(Character).velocity.set(eid, 36, -12, 0, 0);
        world.step(Time.FIXED_DT);
        const s = world.resource(characterScratch);
        expect(s.groundShape).toBeGreaterThanOrEqual(0);
        expect(physics.state.shapes[s.groundShape].bodyId).not.toBe(own.id.index1 - 1);
        expect(s.count).toBeGreaterThan(0);
        for (let i = 0; i < s.count; ++i)
            expect(physics.state.shapes[s.shapes[i]].bodyId).not.toBe(own.id.index1 - 1);
        expect(world.storage(Character).groundNormal.y.get(eid)).toBe(1);
        expect(world.storage(Body).position.x.get(eid)).toBeCloseTo(0.6, 6);
    } finally {
        dispose();
    }
});

test("ground above maxSlope reports steep ground and its normal", async () => {
    const { world, eid, dispose } = await scene();
    try {
        const item = lane.cases.find((item) => item.id === "m1.slope-hull.v1")!;
        const resource = item.input.commands.find(
            (command) => command.op === "resource.hull",
        )! as unknown as { points: string[][] };
        const word = new Uint32Array(1),
            scalar = new Float32Array(word.buffer);
        const number = (bits: string) => {
            word[0] = Number.parseInt(bits.slice(2), 16);
            return scalar[0];
        };
        const points = resource.points.map((p) => ({
            x: number(p[0]),
            y: number(p[1]),
            z: number(p[2]),
        }));
        const hull = createHull(points, points.length)!;
        const physics = physicsWorld(world)!;
        physics.createBody().createHull({}, hull);
        world.storage(Body).position.set(eid, 0, 1.8, 0, 0);
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
        const physics = physicsWorld(world)!;
        physics.getBody(platform)!.setLinearVelocity({ x: 2, y: 0, z: 0 });
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
            expect(world.storage(Body).position.x.get(eid)).toBe(0);
            expect(c.velocity.x.get(eid)).toBe(0);
        }
        expect(physics.getBody(platform)!.getPosition().x).toBeGreaterThan(0);
    } finally {
        dispose();
    }
});
