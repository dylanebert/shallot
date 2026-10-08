import { expect, setDefaultTimeout, test } from "bun:test";
import {
    createApp,
    GlobalTransform,
    type Plugin,
    type Resource,
    type World,
} from "@dylanebert/shallot";
import {
    Devices,
    InputPlugin,
    pointerMove,
    pressKey,
    releaseKey,
    touchPoint,
} from "@dylanebert/shallot/input";
import { Body, Hulls, ShapeKind } from "@dylanebert/shallot/physics";
import {
    hashPhysics,
    physicsWorld,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);
await setupGlobals();
const Score: Resource<{ value: number }> = { create: () => ({ value: 0 }) };
const Gameplay: Plugin = {
    name: "Gameplay",
    recovery: (world: World) => ({
        snapshot: () => world.resource(Score).value,
        restore: (value: number) => {
            world.resource(Score).value = value;
        },
    }),
    systems: [
        {
            name: "clock",
            group: "fixed",
            update(world) {
                world.resource(Score).value += world.time.elapsed;
            },
        },
    ],
};
test("world recovery includes gameplay, identity, allocation, clock and physics including post-capture bodies", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin, Gameplay] });
    const world = app.world;
    try {
        const body = world.create();
        world.add(body, Body, { position: [0, 3, 0, 0] });
        const ref = world.ref(body);
        const storage = world.storage(Body);
        world.tick();
        const saved = world.snapshot();
        const capturedScore = world.resource(Score).value;
        const advance = () => {
            world.destroy(body);
            const reused = world.create();
            world.add(reused, Body, { position: [0, 5, 0, 0] });
            const spawned = world.create();
            world.add(spawned, Body, { position: [2, 4, 0, 0] });
            world.tick();
            world.tick();
            return {
                reused,
                spawned,
                ref: world.ref(reused),
                old: world.resolve(ref),
                y: storage.position.y.get(reused),
                placement: world.storage(GlobalTransform).translation.y.get(reused),
                members: [...world.query([Body])],
                entities: world.entities(),
                tick: world.time.fixedTick,
                score: world.resource(Score).value,
                hash: hashPhysics(world),
            };
        };
        const first = advance();
        world.restore(saved);
        expect(world.resolve(ref)).toBe(body);
        expect(world.resource(Score).value).toBe(capturedScore);
        expect(world.resolve(first.ref)).toBe(0);
        expect(world.exists(first.spawned)).toBe(false);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(advance()).toEqual(first);
        world.restore(saved);
        expect(advance()).toEqual(first);
    } finally {
        app.dispose();
    }
});
test("recovery restores hull authoring read by fixed sync, so failed bodies do not marshal from future hulls", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const hulls = world.resource(Hulls);
        const body = world.create();
        world.add(body, Body, { shape: ShapeKind.Hull, halfExtents: [1, 1, 1, 1] });
        world.tick();
        const saved = world.snapshot();
        world.tick();
        const expected = hashPhysics(world);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);
        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        expect(hulls.register({ ...cube, name: "future-hull" })).toBe(1);
        world.restore(saved);
        world.tick();
        expect(hashPhysics(world)).toBe(expected);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(0);
        expect(world.resource(Hulls)).toBe(hulls);
        expect(hulls.size).toBe(1);
        expect(hulls.id("future-hull")).toBeUndefined();
        expect(hulls.register({ ...cube, name: "replayed-hull" })).toBe(1);
    } finally {
        app.dispose();
    }
});

test("Input recovery preserves accepted facts and retained device handles, not browser handles", async () => {
    const app = await createApp({ defaults: false, plugins: [InputPlugin] });
    try {
        const world = app.world;
        const devices = world.resource(Devices);
        const keys = devices.keys;
        const pointer = devices.pointer;
        pressKey(world, "Space");
        pointerMove(world, 12, 8, 3, 4);
        touchPoint(world, 1, 10, 20);
        const saved = world.snapshot();
        world.tick();
        releaseKey(world, "Space");
        pointerMove(world, 42, 18, 8, 9);
        touchPoint(world, 1, 30, 40);
        world.restore(saved);
        expect(world.resource(Devices)).toBe(devices);
        expect(devices.keys).toBe(keys);
        expect(devices.pointer).toBe(pointer);
        expect(keys.held.has("Space")).toBe(true);
        expect(keys.tickPressed.has("Space")).toBe(true);
        expect(keys.tickReleased.size).toBe(0);
        expect(pointer.x).toBe(12);
        world.tick();
        expect(keys.tickPressed.size).toBe(0);
        world.restore(saved);
        expect(keys.tickPressed.has("Space")).toBe(true);
    } finally {
        app.dispose();
    }
});

test.each(["declarative", "initialize"])(
    "snapshot refuses an enabled %s fixed plugin without recovery by name",
    async (registration) => {
        let captures = 0;
        let counter = 0;
        const app = await createApp({
            defaults: false,
            plugins: [
                {
                    name: "Registered",
                    recovery: () => ({
                        snapshot() {
                            captures++;
                        },
                        restore() {},
                    }),
                },
                {
                    name: "UnregisteredGameplay",
                    systems:
                        registration === "declarative"
                            ? [{ name: "fixed", group: "fixed", update() {} }]
                            : [],
                    initialize(world) {
                        if (registration === "initialize")
                            world.addSystem({
                                name: "UnregisteredFixed",
                                group: "fixed",
                                update() {
                                    counter++;
                                },
                            });
                    },
                },
            ],
        });
        try {
            app.world.tick();
            if (registration === "initialize") expect(counter).toBe(1);
            expect(() => app.world.snapshot()).toThrow(
                registration === "declarative" ? "UnregisteredGameplay" : "UnregisteredFixed",
            );
            expect(captures).toBe(0);
        } finally {
            app.dispose();
        }
    },
);
