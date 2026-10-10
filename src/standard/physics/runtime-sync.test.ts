import { expect, spyOn, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { Body, BodyType, Hulls, ShapeKind } from "@dylanebert/shallot/physics";
import {
    physicsWorld,
    readBody,
    StandardPhysicsPlugin,
    setKinematic,
} from "@dylanebert/shallot/standard/physics";
import { Transform } from "@dylanebert/shallot/transform";

async function createPhysicsWorld(): Promise<World> {
    const world = new World();
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    const recovery = StandardPhysicsPlugin.recovery!;
    if (recovery === "stateless") throw new Error("StandardPhysicsPlugin must recover");
    world.registerRecovery(StandardPhysicsPlugin.name, recovery(world));
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    return world;
}

async function withPhysics(run: (world: World) => void | Promise<void>): Promise<void> {
    const world = await createPhysicsWorld();
    try {
        await run(world);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}

test("a corrected dynamic body's published velocity agrees with its solver body after one tick", async () => {
    await withPhysics((world) => {
        const floor = world.create();
        world.add(floor, Body, { position: [0, -0.5, 0, 0], halfExtents: [10, 0.5, 10, 0] });
        const eid = world.create();
        world.add(eid, Body, { type: BodyType.Dynamic, position: [0, 0.5, 0, 0] });
        world.tick();
        physicsWorld(world)!.getBody(eid)!.setLinearVelocity({ x: 4, y: 0, z: 0 });
        const saved = world.snapshot();
        world.tick();
        world.restore(saved);

        const body = physicsWorld(world)!.getBody(eid)!;
        const position = body.getPosition();
        setKinematic(world, eid, [position.x, position.y, position.z], [0, 0, 0, 1], true);
        world.tick();

        const published = readBody(world, eid)!.linearVelocity;
        const solved = physicsWorld(world)!.getBody(eid)!.getLinearVelocity();
        expect(published[0]).toBeCloseTo(solved.x, 5);
        expect(published[1]).toBeCloseTo(solved.y, 5);
        expect(published[2]).toBeCloseTo(solved.z, 5);
    });
});

test("physics sync warns when Transform is added to a bound Body", async () => {
    await withPhysics((world) => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            const eid = world.create();
            world.add(eid, Body);
            world.tick();
            expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);

            world.add(eid, Transform);
            world.tick();

            expect(warn).toHaveBeenCalledWith(
                expect.stringContaining(`entity ${eid} carries both Body and Transform`),
            );
        } finally {
            warn.mockRestore();
        }
    });
});

test("physics sync walks Bodies added before a restored snapshot was bound", async () => {
    await withPhysics((world) => {
        const first = world.create();
        world.add(first, Body);
        world.tick();

        const second = world.create();
        world.add(second, Body);
        const saved = world.snapshot();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        world.restore(saved);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.tick();

        expect(world.has(second, Body)).toBe(true);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(2);
        expect(physicsWorld(world)!.getBody(second)?.isValid()).toBe(true);
    });
});

test("physics sync retries failed hull bodies when the registry grows", async () => {
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body, {
            shape: ShapeKind.Hull,
            halfExtents: [1, 1, 1, 1],
        });
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);

        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        expect(hulls.register({ ...cube, name: "runtime-sync-recovery-hull" })).toBe(1);
        world.tick();

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
    });
});

test("physics sync visits changed Bodies beyond the first eid word", async () => {
    await withPhysics((world) => {
        world.tick();
        for (let i = 0; i < 40; i++) world.create();
        const eid = world.create();
        expect(eid).toBeGreaterThanOrEqual(32);
        world.add(eid, Body);

        world.tick();

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
    });
});

test("physics sync forgets a Body destroyed before the next tick", async () => {
    await withPhysics((world) => {
        const eid = world.create();
        world.add(eid, Body);
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);

        world.destroy(eid);
        world.tick();

        expect(world.exists(eid)).toBe(false);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);
        expect(physicsWorld(world)!.getBody(eid)).toBeNull();
    });
});

test("physics sync retries a failed hull body whose hull a restored snapshot registered before its sync", async () => {
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const eid = world.create();
        world.add(eid, Body, {
            shape: ShapeKind.Hull,
            halfExtents: [1, 1, 1, 1],
        });
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);

        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        hulls.register({ ...cube, name: "runtime-sync-restored-hull" });
        const saved = world.snapshot();
        world.tick();
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        world.restore(saved);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);

        world.tick();

        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getBody(eid)?.isValid()).toBe(true);
    });
});
