// The tumble.js `Filter` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Two boxes dropped onto two static platforms; a filter joint suppresses the
// contact between the left box and its platform, so it falls straight through to the floor, while the
// right box (no filter) rests on top. The joint carries no constraint — it only mutes a collision pair.
//
// Creation order is load-bearing for the hash: ground, left platform, right platform, the filtered box,
// then the colliding box — the sample's exact order. The name labels are the sample's `render()` overlay
// ({@link renderFilter}) — projected HTML, outside the gold contract.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { Overlay, SampleRender } from "./tumble-overlay";

let labeled: { body: Body; name: string }[] = [];

/**
 * Author the Filter scene into `world`: two static platforms and two dynamic density-1 boxes dropped on
 * them, the left pair joined by a filter joint so that box falls through.
 */
export function buildFilter(world: World): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const platform = (x: number): Body => {
        const p = world.createBody({ type: BodyType.Static, position: { x, y: 2.5, z: 0 } });
        p.createHull({}, makeBoxHull(1.2, 0.2, 1.2));
        return p;
    };

    const left = platform(-2.5);
    platform(2.5);
    const through = world.createBody({
        type: BodyType.Dynamic,
        position: { x: -2.5, y: 6, z: 0 },
    });
    through.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));
    world.createFilterJoint(left, through);

    const onTop = world.createBody({ type: BodyType.Dynamic, position: { x: 2.5, y: 6, z: 0 } });
    onTop.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));

    labeled = [
        { body: through, name: "filtered" },
        { body: onTop, name: "collides" },
    ];
}

/** Label the filtered vs colliding box (the sample's `render()`). */
export const renderFilter: SampleRender = (draw: Overlay) => {
    for (const { body, name } of labeled) {
        const p = body.getPosition();
        draw.string3d({ x: p.x, y: p.y + 1, z: p.z }, name, 0xffffff);
    }
};
