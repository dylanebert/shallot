import { expect, test } from "bun:test";
import { BodyType, makeBoxHull, PhysicsWorld } from "./index";

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
            expect(a.isValid() && b.isValid()).toBe(true);
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

for (const pre of [false, true]) {
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
        } finally {
            world.destroy();
        }
    });
}
