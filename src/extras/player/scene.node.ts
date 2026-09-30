import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    build,
    Character,
    CharacterPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    readBody,
    Time,
} from "../../index";

setDefaultTimeout(20_000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a scene-authored Player capsule retains its collider geometry and rests above the floor", async () => {
    const app = await build({
        defaults: false,
        plugins: [PhysicsPlugin, CharacterPlugin, PlayerPlugin],
        scene: `<scene>
<a id="eye" camera transform="pos: 0 1.5 5" />
<a id="player" body="pos: 0 1 0; shape: 2; half-extents: 0 0.6 0 0.3; mass: 0" character player="camera: @eye" />
<a body="pos: 0 0 0; half-extents: 10 0.5 10; mass: 0" />
</scene>`,
    });
    try {
        const eid = app.state.only([Character]);
        const body = app.state.of(Body);
        expect(body.halfExtents.y.get(eid)).toBeCloseTo(0.6);
        expect(body.halfExtents.w.get(eid)).toBeCloseTo(0.3);
        for (let i = 0; i < 5; i++) app.state.step(Time.FIXED_DT);
        expect(readBody(app.state, eid)?.pos[1]).toBeCloseTo(1.4, 3);
    } finally {
        app.dispose();
    }
});
