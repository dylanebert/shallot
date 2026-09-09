// The tumble.js `BulletVsStack` sample (`samples/src/samples/continuous.ts`) reproduced near-verbatim
// through the escape-hatch `World` API. A backing wall and an 8-box stack, plus a bullet fired at the
// `speed` knob's rate — marked `isBullet`, it sweeps against the stack so it strikes rather than tunnels.
//
// The sample's `launch()` also refires on a knob-panel button and `act()`; the gold trajectory only ever
// sees the one launch build() makes at scene creation, so the button is dropped here (the host's
// `knobParams` already drops it — buttons are transport-only).
//
// Creation order is load-bearing for the hash: ground, back wall, the 8 stack boxes, then the bullet — the
// sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Bullet vs Stack scene into `world`, reading the `speed` knob (the bullet's launch speed): a
 * static ground box, a static backing wall, an 8-box density-1 stack, and a density-10 bullet sphere fired
 * from x=20 at `-speed` along x.
 */
export function buildBulletVsStack(world: World, params: SampleParams): void {
    const speed = params.speed as number;

    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const back = world.createBody({ type: BodyType.Static, position: { x: -6, y: 5, z: 0 } });
    back.createHull({}, makeBoxHull(0.2, 5, 6));

    const box = makeBoxHull(0.5, 0.5, 0.5);
    for (let row = 0; row < 8; ++row) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 0.5 + 1.1 * row, z: 0 },
        });
        body.createHull({ density: 1 }, box);
    }

    const bullet = world.createBody({
        type: BodyType.Dynamic,
        isBullet: true,
        position: { x: 20, y: 4.5, z: 0 },
        linearVelocity: { x: -speed, y: 0, z: 0 },
    });
    bullet.createSphere({ density: 10 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.25 });
}
