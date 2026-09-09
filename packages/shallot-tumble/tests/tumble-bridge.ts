// The tumble.js `Bridge` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A suspension bridge of heavy planks, each hinged to its neighbour (and the
// end posts) by two spherical joints along the shared edge — the pair of pivots lets planks fold about
// the span axis but not twist. Four boxes dropped on the deck sag it.
//
// Creation order is load-bearing for the hash: the ground, the two end posts, then each plank left to
// right (hinged to the previous), then the four dropped boxes — the sample's exact order.

import {
    type Body,
    BodyType,
    makeBoxHull,
    type Transform,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

/** A joint frame pinning `worldPoint` in `body`'s local frame (identity rotation). */
function pin(body: Body, worldPoint: Vec3): Transform {
    return { p: body.getLocalPoint(worldPoint), q: IDENT };
}

/**
 * Author the Bridge scene into `world`, reading the `planks` knob (deck plank count). A static ground
 * box, two static end posts, a run of density-20 planks each paired-spherical-hinged to its neighbour
 * along the shared edge, and four loose density-1 boxes dropped on the deck.
 */
export function buildBridge(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const count = params.planks as number;
    const px = 0.125;
    const pz = 0.5;
    const step = 2 * px;
    const deckY = 6;
    const startX = -(count * step) / 2;

    const post = (x: number): Body => {
        const b = world.createBody({ type: BodyType.Static, position: { x, y: deckY, z: 0 } });
        b.createHull({}, makeBoxHull(0.15, 0.4, 0.7));
        return b;
    };
    const leftPost = post(startX - step);
    const rightPost = post(startX + count * step);

    // Two spherical joints at the plank's shared edge with `prev`, front and back (z = ±pz).
    const hinge = (a: Body, b: Body, edgeX: number): void => {
        for (const z of [-pz, pz]) {
            const pivot = { x: edgeX, y: deckY, z };
            world.createSphericalJoint(a, b, {
                localFrameA: pin(a, pivot),
                localFrameB: pin(b, pivot),
                constraintHertz: 1000,
                enableSpring: true,
                hertz: 2,
                dampingRatio: 1,
            });
        }
    };

    let prev: Body = leftPost;
    let lastX = startX;
    for (let i = 0; i < count; ++i) {
        const cx = startX + i * step + px;
        const plank = world.createBody({
            type: BodyType.Dynamic,
            position: { x: cx, y: deckY, z: 0 },
        });
        plank.createHull({ density: 20 }, makeBoxHull(px, px, pz));
        hinge(prev, plank, cx - px);
        prev = plank;
        lastX = cx;
    }
    hinge(prev, rightPost, lastX + px);

    for (let i = 0; i < 4; ++i) {
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: startX + ((i + 1) * count * step) / 5, y: deckY + 2 + i, z: 0 },
        });
        b.createHull({ density: 1 }, makeBoxHull(0.4, 0.4, 0.4));
    }
}
