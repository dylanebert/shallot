// The tumble.js `Restitution` sample (`samples/src/samples/shapes.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A row of bodies dropped from the same height with restitution ramping 0 → 1
// across the row: the left ones stick, the right ones bounce nearly back to the drop height. The `shape`
// knob picks sphere vs box; `count` picks the row length. No `update()`.
//
// Creation order is load-bearing for the hash: ground, then each body left to right in ascending
// restitution — the sample's exact order.

import {
    BodyType,
    defaultSurfaceMaterial,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Restitution scene into `world`, reading the `shape` knob (sphere/box) and `count` knob
 * (row length, 4-40).
 */
export function buildRestitution(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const count = params.count as number;
    const box = makeBoxHull(0.5, 0.5, 0.5);
    const isSphere = (params.shape as string) === "sphere";
    const dr = 1 / (count > 1 ? count - 1 : 1);
    const dx = 2;
    let x = -1 * (count - 1);
    for (let i = 0; i < count; ++i) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 1 + 39, z: 0 },
        });
        const material = { ...defaultSurfaceMaterial(), restitution: i * dr };
        if (isSphere) {
            body.createSphere(
                { density: 1, baseMaterial: material },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        } else {
            body.createHull({ density: 1, baseMaterial: material }, box);
        }
        x += dx;
    }
}
