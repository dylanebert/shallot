// The tumble.js `BoxPyramid` sample (`samples/src/samples/stacks.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Boxes stacked into a pyramid, settling on the ground under gravity — the
// canonical resting-contact test. The `rows` knob picks the pyramid's base row count.
//
// Creation order is load-bearing for the hash: ground, then each row's boxes left to right, base row
// first — the sample's exact nested-loop order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Box Pyramid scene into `world`, reading the `rows` knob (base row count, 3-14).
 */
export function buildBoxPyramid(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const baseCount = params.rows as number;
    const half = 0.5;
    const spacing = 1.02;
    const hull = makeBoxHull(half, half, half);
    for (let row = 0; row < baseCount; ++row) {
        const count = baseCount - row;
        const y = 1 + half + row * (2 * half);
        const x0 = -0.5 * (count - 1) * spacing;
        for (let i = 0; i < count; ++i) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: x0 + i * spacing, y, z: 0 },
            });
            body.createHull({ density: 1 }, hull);
        }
    }
}
