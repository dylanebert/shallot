import { setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    Body,
    Character,
    CharacterPlugin,
    createApp,
    Devices,
    InputPlugin,
    Player,
    PlayerPlugin,
    pointerLockChanged,
    pointerMove,
    pressKey,
    readBody,
    releaseKey,
    ShapeKind,
    StandardPhysicsPlugin,
    Time,
    Transform,
} from "@dylanebert/shallot";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

test("the public Player controller consumes held, released and neutral input to drive a Character and a linked Transform without a Camera, renderer or browser input", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [InputPlugin, CharacterPlugin, StandardPhysicsPlugin, PlayerPlugin],
    });
    try {
        const world = app.world;
        const floor = world.create();
        world.add(floor, Body);
        world.storage(Body).shape.set(floor, ShapeKind.Box);
        world.storage(Body).position.set(floor, 0, 0, 0, 0);
        world.storage(Body).halfExtents.set(floor, 4, 0.5, 4, 0);
        world.storage(Body).mass.set(floor, 0);

        const camera = world.create();
        world.add(camera, Transform);

        const player = world.create();
        world.add(player, Body);
        world.add(player, Character);
        world.add(player, Player);
        world.storage(Body).shape.set(player, ShapeKind.Capsule);
        world.storage(Body).position.set(player, 0, 1.3, 0, 0);
        world.storage(Body).halfExtents.set(player, 0, 0.5, 0, 0.3);
        world.storage(Body).mass.set(player, 0);
        world.storage(Player).speed.set(player, 6);
        world.storage(Player).sprint.set(player, 1);
        world.storage(Player).sensitivity.set(player, 1.5);
        world.storage(Player).camera.set(player, camera);
        world.storage(Player).jumpSpeed.set(player, 7);
        world.storage(Player).gravity.set(player, 30);

        // Establish the floor contact before the supplied jump edge arrives.
        world.step(Time.FIXED_DT);
        const initial = readBody(world, player);
        if (!initial) throw new Error("Player body did not enter the CPU physics world");
        const initialYaw = world.storage(Player).yaw.get(player);
        const initialPitch = world.storage(Player).pitch.get(player);

        pointerLockChanged(world, true);
        pointerMove(world, 0, 0, 12, -4);
        pressKey(world, "KeyW");
        pressKey(world, "Space");
        world.step(Time.FIXED_DT);
        const lookScale = world.storage(Player).sensitivity.get(player) / 1080;
        if (
            Math.abs(world.storage(Player).yaw.get(player) - (initialYaw - 12 * lookScale)) >
            0.000001
        )
            throw new Error("Player did not consume the supplied locked look sensitivity");
        if (
            Math.abs(world.storage(Player).pitch.get(player) - (initialPitch + 4 * lookScale)) >
            0.000001
        )
            throw new Error("Player did not consume the supplied vertical look sensitivity");
        const expectedYaw = initialYaw - 12 * lookScale;
        const expectedPitch = initialPitch + 4 * lookScale;
        const halfYaw = expectedYaw * 0.5;
        const halfPitch = expectedPitch * 0.5;
        const expectedCameraY = Math.sin(halfYaw) * Math.cos(halfPitch);
        if (Math.abs(world.storage(Transform).rotation.y.get(camera) - expectedCameraY) > 0.000001)
            throw new Error("Player did not apply look to the public camera Transform.rot");
        if (!world.resource(Devices).keys.held.has("KeyW"))
            throw new Error("Player lost the held move fact");

        // Observe the composed fixed-tick consumer, not a private intent store.
        world.step(Time.FIXED_DT);
        const moved = readBody(world, player);
        if (
            !moved ||
            Math.hypot(
                moved.position[0] - initial.position[0],
                moved.position[2] - initial.position[2],
            ) < 0.001
        )
            throw new Error("Character did not apply Player's supplied movement intent");
        if (!moved || moved.position[1] <= initial.position[1] + 0.01)
            throw new Error("Character did not apply Player's supplied jump edge");

        releaseKey(world, "KeyW");
        releaseKey(world, "Space");
        world.step(Time.FIXED_DT); // Release removes acceleration, not momentum.
        const beforeNeutral = readBody(world, player);
        if (!beforeNeutral) throw new Error("Player body disappeared after release");
        world.step(Time.FIXED_DT); // Friction continues damping the prior velocity.
        const neutral = readBody(world, player);
        if (!neutral) throw new Error("Player body disappeared on the neutral step");
        for (let tick = 0; tick < 120; tick++) world.step(Time.FIXED_DT);
        const resting = readBody(world, player)!;
        world.step(Time.FIXED_DT); // Friction has brought the released player to rest.
        const settled = readBody(world, player);
        if (!settled) throw new Error("Player body disappeared on the settled neutral step");
        if (
            Math.hypot(
                settled.position[0] - resting.position[0],
                settled.position[2] - resting.position[2],
            ) > 0.0001
        )
            throw new Error("released Player movement was replayed after the neutral step");
    } finally {
        app.dispose();
    }
});
