import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    Camera,
    Character,
    CharacterPlugin,
    GlobalTransform,
    GroundState,
    pressKey,
    releaseKey,
    Shape,
    ShapeKind,
    setTargetTransform,
    setTransform,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";
import { DrivePlayerSystem, LocalPlayer, Player, PlayerPlugin } from "./player";

setDefaultTimeout(TIMEOUT.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

async function scene(y = 2, offset = 0) {
    let player = 0,
        floor = 0;
    const app = await createApp({
        defaults: false,
        plugins: [PlayerPlugin],
        setup(world) {
            const camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            floor = world.create();
            world.add(floor, Transform, { translation: [offset, offset, 0] });
            world.add(floor, Body, { type: BodyType.Kinematic });
            world.add(floor, Shape, { scale: [8, 1, 8, 0] });
            player = world.create();
            world.add(player, Transform, { translation: [offset, offset + y, 0] });
            world.add(player, Body, { type: BodyType.Kinematic });
            world.add(player, Shape, { kind: ShapeKind.Capsule });
            world.storage(Shape).capsuleCenter1.set(player, 0, -0.6, 0, 0);
            world.storage(Shape).capsuleCenter2.set(player, 0, 0.6, 0, 0);
            world.storage(Shape).radius.set(player, 0.3);
            world.add(player, Character);
            world.add(player, Player, { camera });
            world.add(player, LocalPlayer);
        },
    });
    return { app, world: app.world, player, floor };
}

// The tick of first touchdown when the player drops from 4 m, pressing Space at `press` (or never).
async function dropAndPress(press: number) {
    const { app, world, player } = await scene(4);
    try {
        const c = world.storage(Character);
        let landed = -1;
        for (let tick = 0; tick < 120; tick++) {
            if (tick === press) pressKey(world, "Space");
            world.step(Time.FIXED_DT);
            if (tick === press) releaseKey(world, "Space");
            if (landed < 0 && c.groundState.get(player) === GroundState.OnGround) landed = tick;
            if (landed >= 0 && c.velocity.y.get(player) > 4) return { landed, jumped: true };
            if (landed >= 0 && tick > landed + 2) break;
        }
        return { landed, jumped: false };
    } finally {
        app.dispose();
    }
}

test("a jump pressed within 0.1 seconds before landing fires on touchdown; an earlier one expires", async () => {
    const { landed } = await dropAndPress(-1);
    expect(landed).toBeGreaterThan(8);
    expect((await dropAndPress(landed - 5)).jumped).toBe(true);
    expect((await dropAndPress(landed - 8)).jumped).toBe(false);
});

async function leaveLedgeAndPress(airTicks: number) {
    const { app, world, player } = await scene();
    try {
        for (let tick = 0; tick < 60; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.OnGround);
        setTransform(world, player, [8, 2, 0], [0, 0, 0, 1]);
        for (let tick = 0; tick < airTicks; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.InAir);
        pressKey(world, "Space");
        world.step(Time.FIXED_DT);
        return world.storage(Character).velocity.y.get(player);
    } finally {
        app.dispose();
    }
}

test("a jump within 0.1 seconds after leaving a ledge fires; a later one does not", async () => {
    expect(await leaveLedgeAndPress(4)).toBeGreaterThan(4);
    expect(await leaveLedgeAndPress(8)).toBeLessThan(0);
});

// Minecraft Java's walk is 4.317 m/s, and its 0.42 block/tick jump under 0.08 block/tick² gravity and 0.98 drag
// peaks at 1.2522 m after 6 of its 20 Hz ticks, landing after 12. Ground response is snappier than its spin-up.
test("default movement reaches walk speed within 0.1 s, falls below 5% of it within 0.15 s and jumps like Minecraft Java", async () => {
    const { app, world, player } = await scene();
    try {
        const c = world.storage(Character);
        const y = world.storage(GlobalTransform).translation.y;
        const speed = () => Math.hypot(c.velocity.x.get(player), c.velocity.z.get(player));
        for (let tick = 0; tick < 120; tick++) world.step(Time.FIXED_DT);
        const rest = y.get(player);
        pressKey(world, "KeyW");
        for (let tick = 0; tick < 6; tick++) world.step(Time.FIXED_DT);
        expect(speed()).toBeGreaterThan(4.317 * 0.98);
        expect(speed()).toBeLessThanOrEqual(4.317);
        releaseKey(world, "KeyW");
        for (let tick = 0; tick < 9; tick++) world.step(Time.FIXED_DT);
        expect(speed()).toBeLessThan(4.317 * 0.05);
        pressKey(world, "Space");
        let apex = 0,
            apexTick = 0,
            down = -1;
        for (let tick = 1; tick <= 60 && down < 0; tick++) {
            world.step(Time.FIXED_DT);
            if (tick === 1) releaseKey(world, "Space");
            const height = y.get(player) - rest;
            if (height > apex) {
                apex = height;
                apexTick = tick;
            }
            if (apexTick > 0 && height < 0.05) down = tick;
        }
        expect(Math.abs(apex - 1.2522)).toBeLessThan(0.02);
        expect(Math.abs(apexTick * Time.FIXED_DT - 0.3)).toBeLessThanOrEqual(2 * Time.FIXED_DT);
        expect(Math.abs(down * Time.FIXED_DT - 0.6)).toBeLessThanOrEqual(2 * Time.FIXED_DT);
    } finally {
        app.dispose();
    }
});

async function checkDiagonalCarry(offset: number, velocityPrecision: number) {
    const { app, world, player, floor } = await scene(1.4, offset);
    try {
        world.storage(Player).gravity.set(player, 0);
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.OnGround);
        let tick = 0;
        world.addSystem({
            name: "platform",
            group: "fixed",
            before: [DrivePlayerSystem, ...CharacterPlugin.systems!],
            update() {
                const distance = 2 * (tick + 1) * Time.FIXED_DT;
                setTargetTransform(
                    world,
                    floor,
                    [offset + distance, offset + distance, 0],
                    [0, 0, 0, 1],
                );
                tick++;
            },
        });
        for (let i = 0; i < 30; i++) {
            world.step(Time.FIXED_DT);
            if (i < 2) continue;
            const c = world.storage(Character);
            expect(c.groundState.get(player)).toBe(GroundState.OnGround);
            expect(c.groundVelocity.x.get(player)).toBeCloseTo(2, velocityPrecision);
            expect(c.groundVelocity.y.get(player)).toBeCloseTo(2, velocityPrecision);
            expect(c.velocity.x.get(player)).toBeCloseTo(2, velocityPrecision);
            expect(c.velocity.y.get(player)).toBeCloseTo(2, velocityPrecision);
        }
    } finally {
        app.dispose();
    }
}

test("a rider keeps a kinematic platform's reported velocity rather than accumulating carry", () =>
    checkDiagonalCarry(0, 5));

test("a rider stays grounded on a diagonal platform about 1000 m from the origin", () =>
    checkDiagonalCarry(1000, 2));

// The capsule floats Player.hover above contact, and steep contacts are walls, so walking climbs only the edges its
// rounding meets below maxSlope: about 0.1 m.
async function walkAtStep(height: number): Promise<boolean> {
    let player = 0;
    const app = await createApp({
        defaults: false,
        plugins: [PlayerPlugin],
        setup(world) {
            const camera = world.create();
            world.add(camera, Transform);
            const floor = world.create();
            world.add(floor, Body, { type: BodyType.Static });
            world.add(floor, Shape, { scale: [40, 1, 40, 0] });
            const step = world.create();
            world.add(step, Transform, { translation: [0, 0.5 + height / 2, -3] });
            world.add(step, Body, { type: BodyType.Static });
            world.add(step, Shape, { scale: [8, height, 2, 0] });
            player = world.create();
            world.add(player, Transform, { translation: [0, 2, 0] });
            world.add(player, Body, { type: BodyType.Kinematic });
            world.add(player, Shape, { kind: ShapeKind.Capsule });
            world.storage(Shape).capsuleCenter1.set(player, 0, -0.6, 0, 0);
            world.storage(Shape).capsuleCenter2.set(player, 0, 0.6, 0, 0);
            world.storage(Shape).radius.set(player, 0.3);
            world.add(player, Character);
            world.add(player, Player, { camera });
            world.add(player, LocalPlayer);
        },
    });
    try {
        for (let tick = 0; tick < 60; tick++) app.world.step(Time.FIXED_DT);
        pressKey(app.world, "KeyW");
        let furthest = 0;
        for (let tick = 0; tick < 90; tick++) {
            app.world.step(Time.FIXED_DT);
            furthest = Math.min(
                furthest,
                app.world.storage(GlobalTransform).translation.z.get(player),
            );
        }
        return furthest < -4;
    } finally {
        app.dispose();
    }
}

test("walking climbs a 0.08 m step but not a 0.15 m one", async () => {
    expect(await walkAtStep(0.08)).toBe(true);
    expect(await walkAtStep(0.15)).toBe(false);
});
