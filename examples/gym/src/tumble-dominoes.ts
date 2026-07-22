// The tumble.js `Dominoes` sample (`samples/src/samples/stacks.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Concentric rings of thin boxes; the first is kicked with a linear impulse and
// the whole spiral topples in a chain reaction. The `rings` knob picks the ring count.
//
// Creation order is load-bearing for the hash: ground, then each ring's dominoes in ascending angle order,
// ring by ring — the sample's exact nested-loop order. The impulse on the very first domino (ring 0,
// alpha 0) fires inline in the same loop, matching the sample.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Dominoes scene into `world`, reading the `rings` knob (concentric ring count, 1-8).
 */
export function buildDominoes(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const rings = params.rings as number;
    const hull = makeBoxHull(0.2, 0.8, 0.05);
    for (let ring = 0; ring < rings; ++ring) {
        const radius = 7 + 1.1 * ring;
        for (let alpha = 0; alpha <= 360; alpha += 2) {
            const rad = (alpha * Math.PI) / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            const t = alpha / 630;
            const x = radius * cos - t * cos;
            const z = radius * sin - t * sin;
            const half = -rad / 2;
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x, y: 1 + 0.8, z },
                rotation: { v: { x: 0, y: Math.sin(half), z: 0 }, s: Math.cos(half) },
            });
            body.createHull({}, hull);

            if (ring === 0 && alpha === 0) {
                body.applyLinearImpulse({ x: 0, y: 0, z: 25 }, { x, y: 1 + 1.6, z }, true);
            }
        }
    }
}
