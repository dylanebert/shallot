import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import {
    Body,
    Camera,
    Character,
    CharacterPlugin,
    createApp,
    PhysicsPlugin,
    Player,
    PlayerPlugin,
    readBody,
    ShapeKind,
    Time,
    Transform,
} from "../../index";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a Player capsule retains its collider geometry and rests above the floor", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, PlayerPlugin],
        setup(world) {
            const eye = world.create();
            world.add(eye, Camera);
            world.add(eye, Transform, { translation: [0, 1.5, 5, 0] });
            const player = world.create();
            world.add(player, Body, {
                position: [0, 1, 0, 0],
                shape: ShapeKind.Capsule,
                halfExtents: [0, 0.6, 0, 0.3],
                mass: 0,
            });
            world.add(player, Character);
            world.add(player, Player, { camera: eye });
            const floor = world.create();
            world.add(floor, Body, {
                position: [0, 0, 0, 0],
                halfExtents: [10, 0.5, 10, 0],
                mass: 0,
            });
        },
    });
    try {
        const eid = app.world.only([Character]);
        const body = app.world.storage(Body);
        expect(body.halfExtents.y.get(eid)).toBeCloseTo(0.6);
        expect(body.halfExtents.w.get(eid)).toBeCloseTo(0.3);
        for (let i = 0; i < 5; i++) app.world.step(Time.FIXED_DT);
        expect(readBody(app.world, eid)?.position[1]).toBeCloseTo(1.4, 3);
    } finally {
        app.dispose();
    }
});
