import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    Camera,
    Character,
    CharacterPlugin,
    GlobalTransform,
    Shape,
    ShapeKind,
    StandardPhysicsPlugin,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";
import { Player, PlayerPlugin } from "./player";

setDefaultTimeout(TIMEOUT.node);

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
            world.add(player, Transform, { translation: [0, 2, 0] });
            world.add(player, Body, { type: BodyType.Kinematic });
            world.add(player, Shape, { kind: ShapeKind.Capsule });
            world.storage(Shape).capsuleCenter1.set(player, 0, -0.6, 0, 0);
            world.storage(Shape).capsuleCenter2.set(player, 0, 0.6, 0, 0);
            world.storage(Shape).radius.set(player, 0.3);
            world.add(player, Character);
            world.add(player, Player, { camera: eye });
            const floor = world.create();
            world.add(floor, Transform, { translation: [0, 0, 0] });
            world.add(floor, Body);
            world.add(floor, Shape, { scale: [20, 1, 20, 0] });
        },
    });
    try {
        const eid = app.world.only([Character]);
        const shape = app.world.storage(Shape);
        expect(shape.capsuleCenter2.y.get(eid)).toBeCloseTo(0.6);
        expect(shape.radius.get(eid)).toBeCloseTo(0.3);
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
