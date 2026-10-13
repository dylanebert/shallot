import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "@dylanebert/shallot/testing";
import { createTestGpuApp as createApp } from "../../../scripts/gpu.fixture";
import {
    Body,
    BodyType,
    Camera,
    Character,
    CharacterPlugin,
    GlobalTransform,
    Player,
    PlayerPlugin,
    Shape,
    ShapeKind,
    StandardPhysicsPlugin,
    Time,
    Transform,
} from "../../index";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("a Player capsule retains its collider geometry and rests above the floor", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, CharacterPlugin, PlayerPlugin],
        setup(world) {
            const eye = world.create();
            world.add(eye, Camera);
            world.add(eye, Transform, { translation: [0, 1.5, 5, 0] });
            const player = world.create();
            world.add(player, Body, {
                position: [0, 2, 0, 0],
                type: BodyType.Kinematic,
            });
            world.add(player, Shape, { kind: ShapeKind.Capsule });
            world.storage(Shape).capsuleA.set(player, 0, -0.6, 0, 0);
            world.storage(Shape).capsuleB.set(player, 0, 0.6, 0, 0.3);
            world.add(player, Character);
            world.add(player, Player, { camera: eye });
            const floor = world.create();
            world.add(floor, Body, { position: [0, 0, 0, 0] });
            world.add(floor, Shape, { scale: [10, 0.5, 10, 0] });
        },
    });
    try {
        const eid = app.world.only([Character]);
        const shape = app.world.storage(Shape);
        expect(shape.capsuleB.y.get(eid)).toBeCloseTo(0.6);
        expect(shape.capsuleB.w.get(eid)).toBeCloseTo(0.3);
        for (let i = 0; i < 120; i++) app.world.step(Time.FIXED_DT);
        // Gravity's per-tick displacement balances the spring slightly below its rest length.
        const omega = 2 * Math.PI * 4;
        // Floor top, hover, capsule half-height and radius.
        const equilibrium =
            0.5 +
            app.world.storage(Player).hover.get(eid) +
            0.9 -
            (app.world.storage(Player).gravity.get(eid) *
                Time.FIXED_DT *
                (2 * 0.7 * omega * Time.FIXED_DT + (omega * Time.FIXED_DT) ** 2)) /
                (omega * omega * Time.FIXED_DT);
        expect(app.world.storage(GlobalTransform).translation.y.get(eid)).toBeCloseTo(
            equilibrium,
            3,
        );
    } finally {
        app.dispose();
    }
});
