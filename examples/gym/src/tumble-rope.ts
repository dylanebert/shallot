// The tumble.js `Rope` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A rope of capsules on spherical (ball-and-socket) joints, hung horizontally
// and released — it swings down and coils, free to rotate in every direction at each link. The `links`
// knob is the chain length.
//
// Creation order is load-bearing for the hash: ground, the static anchor, then each capsule link left to
// right (hinged to the previous) — the sample's exact order.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/**
 * Author the Rope scene into `world`, reading the `links` knob (chain length). A static anchor and a
 * chain of density-1 capsule links joined by spherical joints, hung horizontally and released.
 */
export function buildRope(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const anchor = world.createBody({ type: BodyType.Static, position: { x: 0, y: 8, z: 0 } });
    const count = params.links as number;
    const seg = 0.35;
    let prev: Body = anchor;
    for (let i = 0; i < count; ++i) {
        const cx = (i + 1) * (2 * seg);
        const link = world.createBody({
            type: BodyType.Dynamic,
            position: { x: cx, y: 8, z: 0 },
            angularDamping: 0.1,
        });
        link.createCapsule(
            { density: 1 },
            { center1: { x: -seg, y: 0, z: 0 }, center2: { x: seg, y: 0, z: 0 }, radius: 0.12 },
        );
        const pivotX = cx - seg;
        world.createSphericalJoint(prev, link, {
            localFrameA: { p: { x: pivotX - prev.getPosition().x, y: 0, z: 0 }, q: IDENT },
            localFrameB: { p: { x: -seg, y: 0, z: 0 }, q: IDENT },
        });
        prev = link;
    }
}
