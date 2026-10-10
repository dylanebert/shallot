import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    Camera,
    Character,
    CharacterPlugin,
    createApp,
    DrivePlayerSystem,
    GroundState,
    LocalPlayer,
    Player,
    PlayerPlugin,
    pressKey,
    releaseKey,
    ShapeKind,
    setTargetTransform,
    setTransform,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

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
            world.add(floor, Body, {
                type: BodyType.Kinematic,
                position: [offset, offset, 0, 0],
                halfExtents: [4, 0.5, 4, 0],
            });
            player = world.create();
            world.add(player, Body, {
                type: BodyType.Kinematic,
                shape: ShapeKind.Capsule,
                halfExtents: [0, 0.6, 0, 0.3],
                position: [offset, offset + y, 0, 0],
            });
            world.add(player, Character);
            world.add(player, Player, { camera });
            world.add(player, LocalPlayer);
        },
    });
    return { app, world: app.world, player, floor };
}

test("a jump pressed less than 0.2 seconds before landing fires on touchdown", async () => {
    const { app, world, player } = await scene(2.5);
    try {
        pressKey(world, "Space");
        world.step(Time.FIXED_DT);
        releaseKey(world, "Space");
        const c = world.storage(Character);
        let landed = false,
            jumped = false;
        for (let tick = 0; tick < 12; tick++) {
            world.step(Time.FIXED_DT);
            if (c.groundState.get(player) === GroundState.OnGround) landed = true;
            if (landed && c.velocity.y.get(player) > 4) {
                jumped = true;
                break;
            }
        }
        expect(landed).toBe(true);
        expect(jumped).toBe(true);
    } finally {
        app.dispose();
    }
});

test("a jump within 0.15 seconds after leaving a ledge fires", async () => {
    const { app, world, player } = await scene();
    try {
        for (let tick = 0; tick < 60; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.OnGround);
        setTransform(world, player, [8, 2, 0], [0, 0, 0, 1]);
        for (let tick = 0; tick < 6; tick++) world.step(Time.FIXED_DT);
        expect(world.storage(Character).groundState.get(player)).toBe(GroundState.InAir);
        pressKey(world, "Space");
        world.step(Time.FIXED_DT);
        expect(world.storage(Character).velocity.y.get(player)).toBeGreaterThan(4);
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
