import { expect, test } from "bun:test";
import { kernel } from "../kernel/kernel";
import { BodyType, hash, makeBoxHull, PhysicsWorld, restore, snapshot } from "./index";

function contactWorld(pre = false): PhysicsWorld {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    const shape = {
        enableCustomFiltering: !pre,
        enablePreSolveEvents: pre,
        enableContactEvents: true,
    };
    world.createBody().createHull(shape, makeBoxHull(2, 0.5, 2));
    world
        .createBody({ type: BodyType.Dynamic, position: { x: 0, y: 0.99, z: 0 } })
        .createSphere(shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    return world;
}

test("custom filter rejects new pairs and clearing it restores default filtering", () => {
    const world = contactWorld();
    try {
        let calls = 0;
        world.setCustomFilterCallback((a, b) => {
            expect([a.id.index1, b.id.index1]).toEqual([1, 2]);
            expect(() => a.isValid()).toThrow("shared kernel instance");
            calls++;
            return false;
        });
        world.step(1 / 60);
        expect(calls).toBe(1);
        expect(world.getCounters().contactCount).toBe(0);
        world.setCustomFilterCallback(null);
        world
            .createBody({ type: BodyType.Dynamic, position: { x: 1, y: 0.99, z: 0 } })
            .createSphere(
                { enableCustomFiltering: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        world.step(1 / 60);
        expect(calls).toBe(1);
        expect(world.getCounters().contactCount).toBeGreaterThan(0);
    } finally {
        world.destroy();
    }
});

test("an always-true first-shape filter preserves native survivor identities under capacity pressure", () => {
    const control = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    const filtered = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        for (const world of [control, filtered]) {
            for (let i = 0; i < 100; ++i)
                world
                    .createBody({ type: BodyType.Dynamic })
                    .createSphere(
                        { enableCustomFiltering: i === 0, enableContactEvents: true },
                        { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                    );
        }
        filtered.setCustomFilterCallback(() => true);
        control.step(1 / 60);
        filtered.step(1 / 60);
        expect(control.getCounters().contactCount).toBe(1600);
        expect(filtered.getCounters().contactCount).toBe(1600);
        const identities = (world: PhysicsWorld) =>
            world
                .getContactEvents()
                .beginEvents.map(
                    (e) => `${e.contact.id.index1}:${e.shapeA.id.index1}:${e.shapeB.id.index1}`,
                )
                .sort();
        expect(identities(filtered)).toEqual(identities(control));
        expect(hash(filtered)).toBe(hash(control));
    } finally {
        control.destroy();
        filtered.destroy();
    }
});

test("pre-solve refuses stepping an unlocked sibling sharing the kernel instance", () => {
    const world = contactWorld(true);
    const sibling = new PhysicsWorld();
    try {
        world.setPreSolveCallback(() => {
            sibling.step(1 / 60);
            return false;
        });
        expect(() => world.step(1 / 60)).toThrow(
            "cannot re-enter the shared kernel instance from a step callback",
        );
        expect(sibling.state.stepIndex).toBe(0);
        expect(sibling.state.locked).toBe(false);
    } finally {
        world.destroy();
        sibling.destroy();
    }
});

test("pre-solve sees the world-space point and A-to-B normal after custom material mixing", () => {
    const world = contactWorld(true);
    try {
        const order: string[] = [];
        world.state.frictionCallback = () => {
            order.push("material");
            return 0.5;
        };
        world.setPreSolveCallback((a, b, point, normal) => {
            expect(a.id.index1).toBe(1);
            expect(b.id.index1).toBe(2);
            expect(point.x).toBe(0);
            expect(point.y).toBeCloseTo(0.495);
            expect(normal).toEqual({ x: 0, y: 1, z: 0 });
            order.push("pre");
            return false;
        });
        world.step(1 / 60);
        expect(order).toEqual(["material", "pre"]);
        expect(world.getContactEvents().beginEvents).toHaveLength(0);
    } finally {
        world.destroy();
    }
});

test("a throwing pre-solve completes step bookkeeping and views before rethrowing", () => {
    const world = contactWorld(true);
    const control = contactWorld(true);
    const error = new Error("bookkeeping probe");
    try {
        kernel(world.state.ecsState).bodySetEntity(world.state.worldId, 1, 42);
        kernel(control.state.ecsState).bodySetEntity(control.state.worldId, 1, 42);
        control.setPreSolveCallback(() => false);
        world.setPreSolveCallback(() => {
            throw error;
        });
        control.step(1 / 60);
        const expected = {
            stepIndex: control.state.stepIndex,
            syncCount: control.state.bodyStore.syncCount,
            moves: control.getBodyEvents().count,
            transform: control.getBodyEvents().moveEvents[0].transform,
        };
        expect(() => world.step(1 / 60)).toThrow(error);
        expect({
            stepIndex: world.state.stepIndex,
            syncCount: world.state.bodyStore.syncCount,
            moves: world.getBodyEvents().count,
            transform: world.getBodyEvents().moveEvents[0].transform,
        }).toEqual(expected);
    } finally {
        world.destroy();
        control.destroy();
    }
});

test("destroying a world releases its callback references without changing the caller's functions", () => {
    const world = contactWorld(true);
    const callback = () => true;
    world.setCustomFilterCallback(callback);
    world.setPreSolveCallback(callback);
    world.destroy();
    expect(world.state.customFilterCallback).toBeNull();
    expect(world.state.preSolveCallback).toBeNull();
    expect(callback()).toBe(true);
});

test("a collision callback observes locked-world guards and cannot capture or reenter a partial step", () => {
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        const sensor = world
            .createBody()
            .createSphere(
                { isSensor: true, enableSensorEvents: true, enableCustomFiltering: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 2 },
            );
        world
            .createBody({ type: BodyType.Dynamic })
            .createSphere(
                { enableSensorEvents: true },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        world.step(1 / 60);
        expect(sensor.getSensorOverlaps()).toHaveLength(1);
        const saved = snapshot(world);
        let calls = 0;
        world.setCustomFilterCallback(() => {
            calls++;
            expect(sensor.getSensorOverlaps()).toHaveLength(0);
            expect(world.getContactEvents()).toEqual({
                beginEvents: [],
                endEvents: [],
                hitEvents: [],
            });
            expect(world.getSensorEvents()).toEqual({ beginEvents: [], endEvents: [] });
            expect(world.getBodyEvents()).toEqual({ count: 0, moveEvents: [] });
            expect(world.getJointEvents()).toHaveLength(0);
            expect(() => snapshot(world)).toThrow("while it is stepping");
            expect(() => restore(world, saved)).toThrow("while it is stepping");
            expect(() => world.step(1 / 60)).toThrow("shared kernel instance");
            world.destroy();
            expect(world.isValid()).toBe(true);
            return true;
        });
        world.step(1 / 60);
        expect(calls).toBe(1);
        expect(world.state.stepIndex).toBe(2);
        expect(sensor.getSensorOverlaps()).toHaveLength(1);
    } finally {
        world.destroy();
    }
});

test("collision callbacks refuse interleaving a sibling query on the same kernel", () => {
    const world = contactWorld();
    const sibling = new PhysicsWorld();
    const saved = snapshot(sibling);
    try {
        world.setCustomFilterCallback(() => {
            expect(() => snapshot(sibling)).toThrow(
                "cannot re-enter the shared kernel instance from a step callback",
            );
            expect(() => restore(sibling, saved)).toThrow(
                "cannot re-enter the shared kernel instance from a step callback",
            );
            sibling.overlapAABB(
                { lowerBound: { x: -1, y: -1, z: -1 }, upperBound: { x: 1, y: 1, z: 1 } },
                () => true,
            );
            return true;
        });
        expect(() => world.step(1 / 60)).toThrow(
            "cannot re-enter the shared kernel instance from a step callback",
        );
    } finally {
        world.destroy();
        sibling.destroy();
    }
});

for (const pre of [false, true]) {
    test(`${pre ? "pre-solve" : "custom filter"} restores caller callback identity and its kernel gate`, () => {
        const world = contactWorld(pre);
        const target = new PhysicsWorld();
        let calls = 0;
        const callback = () => {
            calls++;
            return false;
        };
        try {
            if (pre) world.setPreSolveCallback(callback);
            else world.setCustomFilterCallback(callback);
            const saved = snapshot(world);
            if (pre) world.setPreSolveCallback(null);
            else world.setCustomFilterCallback(null);
            for (const destination of [world, target]) {
                restore(destination, saved);
                expect(
                    pre
                        ? destination.state.preSolveCallback
                        : destination.state.customFilterCallback,
                ).toBe(callback);
                const before = calls;
                destination.step(1 / 60);
                expect(calls - before).toBe(1);
            }
        } finally {
            world.destroy();
            target.destroy();
        }
    });
    test(`${pre ? "pre-solve" : "custom filter"} exceptions return through the kernel before being rethrown`, () => {
        const world = contactWorld(pre);
        const error = new Error("callback failure");
        try {
            if (pre)
                world.setPreSolveCallback(() => {
                    throw error;
                });
            else
                world.setCustomFilterCallback(() => {
                    throw error;
                });
            expect(() => world.step(1 / 60)).toThrow(error);
            expect(world.state.locked).toBe(false);
            if (pre) world.setPreSolveCallback(null);
            else world.setCustomFilterCallback(null);
            expect(() => world.step(1 / 60)).not.toThrow();
        } finally {
            world.destroy();
        }
    });
}
