import { expect, spyOn, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { Body, BodyType, Hulls, RevoluteJoint, ShapeKind } from "@dylanebert/shallot/physics";
import {
    physicsWorld,
    readBody,
    StandardPhysicsPlugin,
    setKinematic,
} from "@dylanebert/shallot/standard/physics";
import { Transform } from "@dylanebert/shallot/transform";
import { makeJointId } from "./api/config";
import { RevoluteJoint as SolverRevoluteJoint } from "./api/joint";
import { RJ_MOTOR_SPEED } from "./kernel/columns";
import { readJointFloat } from "./kernel/jointcolumns";
import { JointField, jointField } from "./kernel/jointrecords";
import { jointIds } from "./solver/joint.fixture";

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

function addRevoluteMotor(world: World): number {
    const anchor = world.create();
    const body = world.create();
    world.add(anchor, Body, { type: BodyType.Static });
    world.add(body, Body);
    const joint = world.create();
    world.add(joint, RevoluteJoint, {
        a: anchor,
        b: body,
        enableMotor: 1,
        localRotationA: [0, 0, 0, 1],
        localRotationB: [0, 0, 0, 1],
    });
    return joint;
}

test("a RevoluteJoint motor-speed edit reuses its solver joint", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();

        expect(
            world.storage(RevoluteJoint).localRotationA.read(joint, new Float32Array(4)),
        ).toEqual(new Float32Array([0, 0, 0, 1]));
        const solver = physicsWorld(world)!;
        expect(solver.getCounters().jointCount).toBe(1);
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        );
        if (solverId === undefined) throw new Error("authored RevoluteJoint was not created");
        const generation = jointField(solver.state, solverId!, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");
        world.storage(RevoluteJoint).motorSpeed.set(joint, 4);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointIds(solver.state)).toContain(solverId);
        expect(jointField(solver.state, solverId!, JointField.generation)).toBe(generation);
        expect(readJointFloat(solver.state, solverId!, RJ_MOTOR_SPEED)).toBe(4);
    });
});

test("a joint field write survives a frame with no fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const generation = jointField(solver.state, solverId, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");

        world.storage(RevoluteJoint).motorSpeed.set(joint, 6);
        const saved = world.snapshot();
        world.step(0);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(0);

        world.storage(RevoluteJoint).motorSpeed.set(joint, 8);
        world.tick();
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        world.restore(saved);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(jointField(solver.state, solverId, JointField.generation)).toBe(generation);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(6);
    });
});

test("a CPU-only draw edit survives frame mark clearing without a fixed tick", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.storage(RevoluteJoint).motorSpeed.set(joint, 11);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(11);

        world.addSystem(
            {
                name: "write-joint-speed-in-draw",
                group: "draw",
                update() {
                    world.storage(RevoluteJoint).motorSpeed.set(joint, 12);
                },
            },
            "JointDrawWrite",
        );
        world.step(0);
        world.tick();

        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(12);
    });
});

test("a CPU-only draw edit followed by a throw reaches the next joint sync", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        const motorSpeed = world.storage(RevoluteJoint).motorSpeed;
        motorSpeed.set(joint, 12);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(12);
        world.clearChanges();

        const brokenDraw = {
            name: "write-joint-speed-then-throw",
            group: "draw" as const,
            update() {
                motorSpeed.set(joint, 13);
                throw new Error("draw failed");
            },
        };
        world.addSystem(brokenDraw, "JointDrawFailure");
        expect(() => world.step(0)).toThrow(
            'System "JointDrawFailure/write-joint-speed-then-throw" threw: draw failed',
        );
        world.removeSystem(brokenDraw);

        world.tick();

        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(13);
    });
});

test("a dirty field is applied when its ECS value matches an earlier value", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        const authored = world.storage(RevoluteJoint).motorSpeed;
        authored.set(joint, 6);
        world.tick();
        const solver = physicsWorld(world)!;
        const solverId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const handle = new SolverRevoluteJoint(solver.state, makeJointId(solver.state, solverId));
        handle.setMotorSpeed(8);
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(8);

        const create = spyOn(solver, "createRevoluteJoint");
        authored.set(joint, 6);
        world.tick();

        expect(create).not.toHaveBeenCalled();
        expect(readJointFloat(solver.state, solverId, RJ_MOTOR_SPEED)).toBe(6);
    });
});

test("joint snapshots restore the matching live constraints when worlds have different joints", async () => {
    await withPhysics((world) => {
        const first = addRevoluteMotor(world);
        world.tick();
        const firstSnapshot = world.snapshot();

        world.remove(first, RevoluteJoint);
        const second = addRevoluteMotor(world);
        world.tick();
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([second]);
        const secondSnapshot = world.snapshot();

        world.restore(firstSnapshot);
        world.tick();
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([first]);

        world.restore(secondSnapshot);
        world.tick();
        expect(physicsWorld(world)!.getCounters().jointCount).toBe(1);
        expect(
            jointIds(physicsWorld(world)!.state).map(
                (id) => physicsWorld(world)!.state.jointUserData[id],
            ),
        ).toEqual([second]);
    });
});

test("joint endpoint changes recreate the solver joint", async () => {
    await withPhysics((world) => {
        const joint = addRevoluteMotor(world);
        world.tick();
        const solver = physicsWorld(world)!;
        const originalId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        const generation = jointField(solver.state, originalId, JointField.generation);
        const create = spyOn(solver, "createRevoluteJoint");
        const replacement = world.create();
        world.add(replacement, Body, { type: BodyType.Static });
        world.tick();
        world.storage(RevoluteJoint).a.set(joint, replacement);
        world.tick();

        expect(create).toHaveBeenCalledTimes(1);
        const newId = jointIds(solver.state).find(
            (id) => solver.state.jointUserData[id] === joint,
        )!;
        expect(jointField(solver.state, newId, JointField.generation)).not.toBe(generation);
        expect(jointIds(solver.state).map((id) => solver.state.jointUserData[id])).toEqual([joint]);
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
