import { expect, setDefaultTimeout, test } from "bun:test";
import { CEILING } from "../../../scripts/test-tiers";
import {
    Body,
    Character,
    CharacterPlugin,
    createApp,
    PhysicsPlugin,
    PlayerPlugin,
    readBody,
    Time,
} from "../../index";

setDefaultTimeout(CEILING.node);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a scene-authored Player capsule retains its collider geometry and rests above the floor", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, PlayerPlugin],
        scene: `<scene>
<a id="eye" camera transform="translation: 0 1.5 5" />
<a id="player" body="position: 0 1 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0" character player="camera: @eye" />
<a body="position: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
</scene>`,
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
