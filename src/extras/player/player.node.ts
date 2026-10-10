import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import {
    Body,
    BodyType,
    Character,
    CharacterPlugin,
    createApp,
    Devices,
    DrivePlayerSystem,
    GlobalTransform,
    GroundState,
    LocalPlayer,
    Player,
    PlayerInput,
    PlayerPlugin,
    pointerLockChanged,
    pointerMove,
    pressKey,
    ShapeKind,
    StandardPhysicsPlugin,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { PlayerMotion } from "./motion";

await setupGlobals();

interface Scene {
    app: Awaited<ReturnType<typeof createApp>>;
    world: Awaited<ReturnType<typeof createApp>>["world"];
    players: number[];
    cameras: number[];
}

async function scene(localPlayers: readonly number[] = [], count = 1): Promise<Scene> {
    const players: number[] = [];
    const cameras: number[] = [];
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, CharacterPlugin, PlayerPlugin],
        setup(world) {
            const floor = world.create();
            world.add(floor, Body, {
                type: BodyType.Static,
                shape: ShapeKind.Box,
                halfExtents: [20, 0.5, 20, 0],
            });
            for (let index = 0; index < count; index++) {
                const camera = world.create();
                world.add(camera, Transform);
                cameras.push(camera);

                const player = world.create();
                world.add(player, Body, {
                    type: BodyType.Kinematic,
                    shape: ShapeKind.Capsule,
                    halfExtents: [0, 0.5, 0, 0.3],
                    position: [index * 4, 1.3, 0, 0],
                });
                world.add(player, Character);
                world.add(player, Player, {
                    camera,
                    acceleration: 100,
                    gravity: 30,
                    jumpSpeed: 7,
                    sprint: 2,
                });
                if (localPlayers.includes(index)) world.add(player, LocalPlayer);
                players.push(player);
            }
        },
    });
    return { app, world: app.world, players, cameras };
}

function writeInput(
    world: Scene["world"],
    eid: number,
    values: {
        move: readonly [number, number];
        sprint?: number;
        yaw?: number;
        pitch?: number;
        jumpPresses?: number;
    },
): void {
    if (!world.has(eid, PlayerInput)) {
        world.add(eid, PlayerInput, {
            yaw: world.storage(Player).yaw.get(eid),
            pitch: world.storage(Player).pitch.get(eid),
        });
    }
    const input = world.storage(PlayerInput);
    input.move.set(eid, values.move[0], values.move[1]);
    input.sprint.set(eid, values.sprint ?? 0);
    input.yaw.set(eid, values.yaw ?? 0);
    input.pitch.set(eid, values.pitch ?? 0);
    input.jumpPresses.set(eid, values.jumpPresses ?? 0);
}

function playerMotionState(world: Scene["world"], eid: number) {
    const player = world.storage(Player);
    const motion = world.storage(PlayerMotion);
    const global = world.storage(GlobalTransform);
    const offset = eid * 4;
    return {
        player: [player.yaw.get(eid), player.pitch.get(eid)],
        motion: {
            carry: Array.from(motion.carry.column.slice(eid * 4, eid * 4 + 4)),
            coyote: motion.coyote.get(eid),
            buffer: motion.buffer.get(eid),
            lastJumpPresses: motion.lastJumpPresses.get(eid),
        },
        body: {
            position: [
                global.translation.column[offset],
                global.translation.column[offset + 1],
                global.translation.column[offset + 2],
            ],
            rotation: [
                global.rotation.column[offset],
                global.rotation.column[offset + 1],
                global.rotation.column[offset + 2],
                global.rotation.column[offset + 3],
            ],
            linearVelocity: [
                global.linearVelocity.column[offset],
                global.linearVelocity.column[offset + 1],
                global.linearVelocity.column[offset + 2],
            ],
        },
    };
}

test("two players follow different records and diverge", async () => {
    const { app, world, players } = await scene([], 2);
    try {
        const [forward, right] = players;
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        writeInput(world, forward, { move: [0, 1] });
        writeInput(world, right, { move: [1, 0] });
        const beforeForward = world.storage(GlobalTransform).translation.z.get(forward);
        const beforeRight = world.storage(GlobalTransform).translation.x.get(right);

        world.tick();

        const inputs = world.storage(PlayerInput);
        expect(inputs.move.column.slice(forward * 2, forward * 2 + 2)).toEqual(
            new Float32Array([0, 1]),
        );
        expect(inputs.move.column.slice(right * 2, right * 2 + 2)).toEqual(
            new Float32Array([1, 0]),
        );
        expect(world.storage(GlobalTransform).translation.z.get(forward)).toBeLessThan(
            beforeForward,
        );
        expect(world.storage(GlobalTransform).translation.x.get(right)).toBeGreaterThan(
            beforeRight,
        );
    } finally {
        app.dispose();
    }
});

test("a player without LocalPlayer ignores the keyboard", async () => {
    const { app, world, players } = await scene();
    try {
        const player = players[0];
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        const before = world.storage(GlobalTransform).translation;
        const beforeX = before.x.get(player);
        const beforeZ = before.z.get(player);
        pressKey(world, "KeyW");
        world.step(Time.FIXED_DT);
        const after = world.storage(GlobalTransform).translation;

        expect(world.has(player, LocalPlayer)).toBe(false);
        expect(after.x.get(player)).toBeCloseTo(beforeX);
        expect(after.z.get(player)).toBeCloseTo(beforeZ);
    } finally {
        app.dispose();
    }
});

test("world.tick drives move, sprint, look and jump from records without device input", async () => {
    const { app, world, players } = await scene();
    try {
        const player = players[0];
        world.storage(Player).yaw.set(player, 0.3);
        world.storage(Player).pitch.set(player, 0.2);
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.OnGround);
        expect(world.has(player, PlayerInput)).toBe(true);
        expect(world.storage(Player).yaw.get(player)).toBeCloseTo(0.3);
        expect(world.storage(Player).pitch.get(player)).toBeCloseTo(0.2);

        writeInput(world, player, {
            move: [0, 1],
            sprint: 1,
            yaw: Math.PI / 2,
            pitch: 0.25,
            jumpPresses: 1,
        });
        world.tick();

        expect(world.storage(Player).yaw.get(player)).toBeCloseTo(Math.PI / 2);
        expect(world.storage(Player).pitch.get(player)).toBeCloseTo(0.25);
        expect(world.storage(Character).velocity.x.get(player)).toBeLessThan(-8);
        expect(world.storage(Character).velocity.y.get(player)).toBeGreaterThan(6);
        expect(world.resource(Devices).keys.held.size).toBe(0);
    } finally {
        app.dispose();
    }
});

test("a jump press before a zero-tick step jumps on the next tick", async () => {
    const { app, world, players } = await scene([0]);
    try {
        const player = players[0];
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.OnGround);

        pressKey(world, "Space");
        world.step(Time.FIXED_DT / 2);
        expect(world.time.fixedSteps).toBe(0);
        world.tick();

        expect(world.storage(Character).velocity.y.get(player)).toBeGreaterThan(4);
        expect(world.storage(PlayerInput).jumpPresses.get(player)).toBe(1);
    } finally {
        app.dispose();
    }
});

test("a jump press before a catch-up step launches once instead of refilling the jump buffer", async () => {
    const { app, world, players } = await scene([0]);
    try {
        const player = players[0];
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        world.storage(Player).gravity.set(player, 60);
        world.storage(Player).jumpSpeed.set(player, 10);
        const launchVelocity: number[] = [];
        world.addSystem({
            name: "ground-for-each-catch-up-tick",
            group: "fixed",
            before: [DrivePlayerSystem],
            update() {
                world.storage(Character).groundState.set(player, GroundState.OnGround);
            },
        });
        world.addSystem({
            name: "observe-catch-up-launches",
            group: "fixed",
            after: [DrivePlayerSystem],
            before: CharacterPlugin.systems,
            update() {
                launchVelocity.push(world.storage(Character).velocity.y.get(player));
            },
        });

        pressKey(world, "Space");
        world.step(Time.FIXED_DT * 4);

        expect(world.time.fixedSteps).toBe(4);
        expect(launchVelocity).toHaveLength(4);
        expect(launchVelocity[0]).toBeGreaterThan(8);
        expect(launchVelocity.slice(1).every((value) => value < 0)).toBe(true);
        expect(world.storage(PlayerInput).jumpPresses.get(player)).toBe(1);
    } finally {
        app.dispose();
    }
});

test("pointer motion before a step turns that step's first tick movement direction", async () => {
    const { app, world, players } = await scene([0]);
    try {
        const player = players[0];
        world.storage(Player).yaw.set(player, 0.4);
        world.storage(Player).pitch.set(player, 0.2);
        world.storage(LocalPlayer).sensitivity.set(player, 2.5);
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        pointerLockChanged(world, true);
        pointerMove(world, 0, 0, 1080, 100);
        pressKey(world, "KeyW");
        world.step(Time.FIXED_DT);

        expect(world.storage(Character).velocity.x.get(player)).toBeGreaterThan(4);
        const sensitivity = world.storage(LocalPlayer).sensitivity.get(player) / 1080;
        expect(world.storage(Player).yaw.get(player)).toBeCloseTo(0.4 - 1080 * sensitivity);
        expect(world.storage(Player).pitch.get(player)).toBeCloseTo(0.2 - 100 * sensitivity);
        expect(world.storage(LocalPlayer).viewYaw.get(player)).toBeCloseTo(
            0.4 - 1080 * sensitivity,
        );
        expect(world.storage(LocalPlayer).viewPitch.get(player)).toBeCloseTo(
            0.2 - 100 * sensitivity,
        );
    } finally {
        app.dispose();
    }
});

test("written records replay identically across a zero-tick step and a catch-up step after restore", async () => {
    const { app, world, players } = await scene();
    try {
        const player = players[0];
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        const snapshot = world.snapshot();
        const run = () => {
            writeInput(world, player, {
                move: [0, 1],
                sprint: 0,
                yaw: 0.2,
                pitch: 0.1,
                jumpPresses: 1,
            });
            world.tick();
            writeInput(world, player, {
                move: [1, 0],
                sprint: 1,
                yaw: 0.4,
                pitch: -0.1,
                jumpPresses: 2,
            });
            world.step(0);
            writeInput(world, player, {
                move: [-1, 0],
                sprint: 0,
                yaw: -0.3,
                pitch: 0,
                jumpPresses: 2,
            });
            world.step(Time.FIXED_DT * 4);
            return playerMotionState(world, player);
        };

        const first = run();
        world.restore(snapshot);
        const replay = run();

        expect(first.player[0]).toBeCloseTo(-0.3);
        expect(first.player[1]).toBeCloseTo(0);
        expect(first.motion.lastJumpPresses).toBe(2);
        expect(replay).toEqual(first);
    } finally {
        app.dispose();
    }
});

test("a jump pressed during a pause does not jump after resume, while look still turns the camera", async () => {
    const { app, world, players, cameras } = await scene([0]);
    try {
        const player = players[0];
        const camera = cameras[0];
        for (let i = 0; i < 4; i++) world.step(Time.FIXED_DT);
        const initialRotation = world.storage(Transform).rotation.y.get(camera);
        const initialYaw = world.storage(Player).yaw.get(player);

        world.pause();
        pointerLockChanged(world, true);
        pointerMove(world, 0, 0, 720, 0);
        pressKey(world, "Space");
        world.step(Time.FIXED_DT);
        expect(world.storage(PlayerInput).jumpPresses.get(player)).toBe(0);
        expect(world.storage(Player).yaw.get(player)).toBe(initialYaw);
        expect(world.storage(Transform).rotation.y.get(camera)).not.toBeCloseTo(initialRotation);

        world.resume();
        world.step(Time.FIXED_DT);
        expect(world.storage(PlayerInput).jumpPresses.get(player)).toBe(0);
        expect(world.storage(Character).velocity.y.get(player)).toBeLessThan(4);
    } finally {
        app.dispose();
    }
});
