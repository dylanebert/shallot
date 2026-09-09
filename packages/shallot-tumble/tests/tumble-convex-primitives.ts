// The tumble.js `ConvexPrimitives` sample (`samples/src/samples/geometry.ts`) reproduced near-verbatim
// through the escape-hatch `World` API. A drop of the built-in convex generators — cylinders, cones, and
// rocks — arranged in a 3x5 grid. No knobs.
//
// Creation order is load-bearing for the hash: ground, then each body row by row, column by column — the
// sample's exact order.

import {
    BodyType,
    createCone,
    createCylinder,
    createRock,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";

/**
 * Author the Convex Primitives scene into `world`: a 3x5 grid of bodies cycling cylinder/cone/rock hulls.
 */
export function buildConvexPrimitives(world: World): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const cols = 5;
    for (let row = 0; row < 3; ++row) {
        for (let col = 0; col < cols; ++col) {
            const x = (col - (cols - 1) / 2) * 1.8;
            const y = 2 + row * 2;
            const kind = (row + col) % 3;
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y, z: 0 },
            });
            const hull =
                kind === 0
                    ? createCylinder(1.0, 0.5, -0.5, 12)
                    : kind === 1
                      ? createCone(1.2, 0.6, 0.05, 12)
                      : createRock(0.7);
            body.createHull({ density: 1 }, hull);
        }
    }
}
