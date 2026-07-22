// The tumble.js `Cantilever` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A row of boxes welded end to end, jutting horizontally from a wall — a
// diving board. The `stiffness` knob is the weld's angular spring: soft droops, stiff holds level.
//
// Creation order is load-bearing for the hash: ground, wall, then the six welded links left to right —
// the sample's exact order.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/**
 * Author the Cantilever scene into `world`, reading the `stiffness` knob (the weld's angular spring
 * hertz). A static ground box, a static wall, and six density-2 links welded end to end jutting from it.
 */
export function buildCantilever(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const hz = params.stiffness as number;
    const half = 0.4;
    const gap = 0.02;
    const y = 6;
    const wall = world.createBody({ type: BodyType.Static, position: { x: 0, y, z: 0 } });
    wall.createHull({}, makeBoxHull(0.3, 1.2, 1.2));

    let prev: Body = wall;
    const count = 6;
    for (let i = 0; i < count; ++i) {
        const cx = 0.3 + half + gap + i * (2 * half + gap);
        const link = world.createBody({ type: BodyType.Dynamic, position: { x: cx, y, z: 0 } });
        link.createHull({ density: 2 }, makeBoxHull(half, half * 0.6, half));
        const pivotX = cx - half - 0.5 * gap;
        world.createWeldJoint(prev, link, {
            localFrameA: { p: { x: pivotX - prev.getPosition().x, y: 0, z: 0 }, q: IDENT },
            localFrameB: { p: { x: -half - 0.5 * gap, y: 0, z: 0 }, q: IDENT },
            angularHertz: hz,
            angularDampingRatio: 1,
        });
        prev = link;
    }
}
