// The tumble.js `InclinedPlane` sample (`samples/src/samples/shapes.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A tilted, high-friction plane with five boxes released across it, friction
// ramping (i+1)²·0.04 low → high: the low-friction box slides off, the high-friction box grips. No knobs,
// no `update()`.
//
// Creation order is load-bearing for the hash: ground, then the plane, then the five boxes low-to-high
// friction — the sample's exact order.

import {
    BodyType,
    defaultSurfaceMaterial,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Inclined Plane scene into `world`. No knobs — `params` is unused, kept for the shared
 * `SampleBuild` signature.
 */
export function buildInclinedPlane(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const tilt = (40 * Math.PI) / 180;
    const rot = { v: { x: Math.sin(tilt / 2), y: 0, z: 0 }, s: Math.cos(tilt / 2) };
    const plane = world.createBody({
        type: BodyType.Static,
        position: { x: 0, y: 8.5, z: -5 },
        rotation: rot,
    });
    plane.createHull(
        { baseMaterial: { ...defaultSurfaceMaterial(), friction: 1 } },
        makeBoxHull(16, 0.5, 10),
    );

    const box = makeBoxHull(1, 1, 1);
    for (let i = 0; i < 5; ++i) {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -10 + 5 * i, y: 16.75, z: -10.6 },
        });
        const friction = (i + 1) * (i + 1) * 0.04;
        body.createHull(
            { density: 1, baseMaterial: { ...defaultSurfaceMaterial(), friction } },
            box,
        );
    }
}
