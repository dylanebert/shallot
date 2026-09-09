// The tumble.js `ShapeSoup` sample (`samples/src/samples/shapes.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Spheres, capsules, and boxes dropped together in a grid, cycling through every
// convex primitive — the `rows` knob picks the grid's row count. No `update()`.
//
// Creation order is load-bearing for the hash: ground, then each row's five bodies left to right, base row
// first — the sample's exact nested-loop order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Shape Soup scene into `world`, reading the `rows` knob (grid row count, 2-8).
 */
export function buildShapeSoup(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const box = makeBoxHull(0.4, 0.4, 0.4);
    const rows = params.rows as number;
    const cols = 5;
    for (let row = 0; row < rows; ++row) {
        for (let col = 0; col < cols; ++col) {
            const x = (col - (cols - 1) / 2) * 1.4;
            const y = 2 + row * 1.6;
            const kind = (row + col) % 3;
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y, z: 0 },
            });
            if (kind === 0) {
                body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
            } else if (kind === 1) {
                body.createCapsule(
                    { density: 1 },
                    {
                        center1: { x: -0.4, y: 0, z: 0 },
                        center2: { x: 0.4, y: 0, z: 0 },
                        radius: 0.35,
                    },
                );
            } else {
                body.createHull({ density: 1 }, box);
            }
        }
    }
}
