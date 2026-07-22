// The tumble.js `OverlapBox` sample (`samples/src/samples/collision.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A static ground plus a 6x6 grid of resting dynamic boxes. The circling
// query box is the sample's `render()` overlay ({@link renderOverlapBox}): a per-frame `overlapAABB` that
// outlines every box it covers + counts them — a read-only query outside the gold contract, so only
// `build()` feeds the gold (it doesn't read `size`, the render-time knob, matching the sample). No `update()`.
//
// Creation order is load-bearing for the hash: ground, then the grid in its nested-loop order (ix outer,
// iz inner) — the sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

/**
 * Author the Overlap Box scene into `world`: a static ground and a 6x6 grid of resting boxes. `params` is
 * unused (the `size` knob only sizes the render-time query box), kept for the shared `SampleBuild`
 * signature.
 */
export function buildOverlapBox(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const box = makeBoxHull(0.4, 0.4, 0.4);
    for (let ix = 0; ix < 6; ++ix) {
        for (let iz = 0; iz < 6; ++iz) {
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: -5 + 2 * ix, y: 1, z: -5 + 2 * iz },
            });
            body.createHull({ density: 1 }, box);
        }
    }
}

/**
 * The circling query box (the sample's `render()`): an AABB of half-size `size` orbiting above the grid;
 * every box whose fat AABB it overlaps is outlined and the HUD counts them — the broad-phase overlap query,
 * live.
 */
export const renderOverlapBox: SampleRender = (
    draw: Overlay,
    world: World,
    params: SampleParams,
    stepCount: number,
) => {
    const s = params.size as number;
    const half = { x: s, y: 1.5, z: s };
    const t = 0.02 * stepCount;
    const c = { x: 6 * Math.cos(t), y: 2.5, z: 6 * Math.sin(t) };
    const min = { x: c.x - half.x, y: c.y - half.y, z: c.z - half.z };
    const max = { x: c.x + half.x, y: c.y + half.y, z: c.z + half.z };

    let count = 0;
    world.overlapAABB({ lowerBound: min, upperBound: max }, (shape) => {
        const p = shape.getBody().getPosition();
        draw.aabb(
            { x: p.x - 0.45, y: p.y - 0.45, z: p.z - 0.45 },
            { x: p.x + 0.45, y: p.y + 0.45, z: p.z + 0.45 },
            0xffd040,
        );
        count += 1;
        return true;
    });
    draw.aabb(min, max, 0x40a0ff);
    draw.text(`overlapping ${count}`);
};
