import { expect, setDefaultTimeout, test } from "bun:test";
import {
    createApp,
    GlobalTransform,
    type Plugin,
    type Resource,
    type World,
} from "@dylanebert/shallot";
import {
    audioContextState,
    Devices,
    InputPlugin,
    pointerMove,
    pressKey,
    releaseKey,
    touchPoint,
} from "@dylanebert/shallot/input";
import {
    Body,
    Compounds,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    Shape,
    ShapeKind,
    ShapeMaterials,
} from "@dylanebert/shallot/physics";
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
test("recovery restores a Shape on another entity with its body reference and solver binding", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const body = world.create();
        world.add(body, Body);
        const collider = world.create();
        world.add(collider, Shape, { body, friction: 0.25 });
        world.tick();
        const saved = world.snapshot();
        const hash = hashPhysics(world);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);

        world.destroy(collider);
        world.tick();
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);
        world.restore(saved);

        expect(world.exists(collider)).toBe(true);
        expect(world.has(collider, Shape)).toBe(true);
        expect(world.storage(Shape).body.get(collider)).toBe(body);
        expect(world.storage(Shape).friction.get(collider)).toBe(0.25);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(1);
        expect(hashPhysics(world)).toBe(hash);
    } finally {
        app.dispose();
    }
});

test("core physics recovery restores every Shape geometry and material registry", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const mesh = world.resource(PhysicsMeshes);
        const heightField = world.resource(HeightFields);
        const compound = world.resource(Compounds);
        const materials = world.resource(ShapeMaterials);
        const meshId = mesh.register({ name: "recovery-mesh", data: { version: 1 } });
        const heightId = heightField.register({ name: "recovery-height", data: { version: 1 } });
        const compoundId = compound.register({ name: "recovery-compound", data: { version: 1 } });
        const materialId = materials.register({
            name: "recovery-materials",
            materials: [
                {
                    friction: 0.25,
                    restitution: 0.5,
                    rollingResistance: 0.1,
                    tangentVelocity: { x: 1, y: 2, z: 3 },
                    userMaterialId: 44n,
                    customColor: 0x123456,
                },
            ],
        });
        const saved = world.snapshot();
        mesh.register({ name: "recovery-mesh", data: { version: 2 } });
        heightField.delete("recovery-height");
        compound.register({ name: "recovery-compound", data: { version: 2 } });
        materials.delete("recovery-materials");
        world.restore(saved);

        expect(mesh.id("recovery-mesh")).toBe(meshId);
        expect(mesh.get("recovery-mesh")?.data).toEqual({ version: 1 });
        expect(heightField.id("recovery-height")).toBe(heightId);
        expect(heightField.get("recovery-height")?.data).toEqual({ version: 1 });
        expect(compound.id("recovery-compound")).toBe(compoundId);
        expect(compound.get("recovery-compound")?.data).toEqual({ version: 1 });
        expect(materials.id("recovery-materials")).toBe(materialId);
        expect(materials.get("recovery-materials")?.materials[0]?.userMaterialId).toBe(44n);
    } finally {
        app.dispose();
    }
});

test("recovery restores Shape authoring and its hull registry before fixed sync", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const hulls = world.resource(Hulls);
        const body = world.create();
        world.add(body, Body);
        world.add(body, Shape, { kind: ShapeKind.Hull, geometry: 1, scale: [1, 1, 1, 0] });
        world.tick();
        const saved = world.snapshot();
        world.tick();
        const expected = hashPhysics(world);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);
        const cube = structuredClone(hulls.get(hulls.name(0)!)!);
        expect(hulls.register({ ...cube, name: "future-hull" })).toBe(1);
        world.restore(saved);
        world.tick();
        expect(hashPhysics(world)).toBe(expected);
        expect(physicsWorld(world)!.getCounters().bodyCount).toBe(1);
        expect(physicsWorld(world)!.getCounters().shapeCount).toBe(0);
        expect(world.resource(Hulls)).toBe(hulls);
        expect(hulls.size).toBe(1);
        expect(hulls.id("future-hull")).toBeUndefined();
        expect(hulls.register({ ...cube, name: "replayed-hull" })).toBe(1);
    } finally {
        app.dispose();
    }
});

test("snapshot and restore leave device state outside recovery", async () => {
    const app = await createApp({ defaults: false, plugins: [InputPlugin] });
    try {
        const world = app.world;
        const devices = world.resource(Devices);
        const keys = devices.keys;
        const pointer = devices.pointer;
        pressKey(world, "Space");
        pointerMove(world, 12, 8, 3, 4);
        touchPoint(world, 1, 10, 20);
        touchPoint(world, 2, 30, 20);
        audioContextState(world, "suspended");
        const saved = world.snapshot();
        releaseKey(world, "Space");
        pointerMove(world, 42, 18, 8, 9);
        touchPoint(world, 1, 30, 40);
        audioContextState(world, "running");
        world.restore(saved);
        expect(world.resource(Devices)).toBe(devices);
        expect(devices.keys).toBe(keys);
        expect(devices.pointer).toBe(pointer);
        expect(keys.held.has("Space")).toBe(false);
        expect(keys.released.has("Space")).toBe(true);
        expect(pointer.x).toBe(42);
        expect(pointer.y).toBe(18);
        expect(pointer.deltaX).toBe(11);
        expect(pointer.deltaY).toBe(13);
        expect(devices.touch.count).toBe(2);
        expect(devices.touch.deltaX).toBe(10);
        expect(devices.touch.deltaY).toBe(10);
        expect(devices.audio.context).toBe("running");
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
