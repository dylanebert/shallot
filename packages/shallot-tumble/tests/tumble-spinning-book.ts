// The tumble.js `SpinningBook` sample (`samples/src/samples/bodies.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. Three flat "books" spinning in free fall (gravityScale 0), each about one
// principal axis — the one spun about its intermediate axis of inertia tumbles chaotically (the
// Dzhanibekov effect / tennis-racket theorem). No knobs, no update.
//
// Creation order is load-bearing for the hash: ground, then the three books in x = -2, 0, 2 order — the
// sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Spinning Book scene into `world`: a static ground box and three density-1 flat-box bodies in
 * free fall (gravityScale 0), each given an initial angular velocity about a different principal axis.
 */
export function buildSpinningBook(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const hull = makeBoxHull(0.35, 0.08, 0.5);
    const spin: [number, [number, number, number]][] = [
        [-2, [5, 0.01, 0.01]],
        [0, [0.01, 5, 0.01]],
        [2, [0.01, 0.01, -5]],
    ];
    for (const [x, [wx, wy, wz]] of spin) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 4, z: 0 },
            gravityScale: 0,
            angularVelocity: { x: wx, y: wy, z: wz },
        });
        body.createHull({ density: 1 }, hull);
    }
}
