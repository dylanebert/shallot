// The tumble.js `Pendulum` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A chain of boxes on revolute hinges, swinging out and settling under gravity
// — the simplest joint island. The `links` knob is the chain length.
//
// Creation order is load-bearing for the hash: ground, the static anchor, then each link left to right
// (hinged to the previous) — the sample's exact order.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/**
 * Author the Pendulum scene into `world`, reading the `links` knob (chain length). A static anchor and a
 * chain of density-1 links joined by revolute hinges, hung horizontally and released.
 */
export function buildPendulum(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const count = params.links as number;
    const half = 0.4;
    const gap = 0.05;
    const hull = makeBoxHull(half, half * 0.5, half * 0.5);
    const y = 8;
    const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y, z: 0 } });

    let prev: Body = anchor;
    for (let i = 0; i < count; ++i) {
        const x = (i + 1) * (2 * half + gap);
        const link = world.createBody({ type: BodyType.Dynamic, position: { x, y, z: 0 } });
        link.createHull({ density: 1 }, hull);
        const pivotX = x - half - 0.5 * gap;
        world.createRevoluteJoint(prev, link, {
            localFrameA: { p: { x: pivotX - prev.getPosition().x, y: 0, z: 0 }, q: IDENT },
            localFrameB: { p: { x: -half - 0.5 * gap, y: 0, z: 0 }, q: IDENT },
        });
        prev = link;
    }
}
