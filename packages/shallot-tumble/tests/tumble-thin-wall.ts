// The tumble.js `ThinWall` sample (`samples/src/samples/continuous.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. Three fast projectiles (sphere, capsule, box) hurled at a thin static
// wall, each moving far more per step than the wall is thick — continuous sweeping stops them at the
// surface. No knobs.
//
// Creation order is load-bearing for the hash: ground, wall, then sphere/capsule/box shot in that order —
// the sample's exact order.

import {
    type Body,
    BodyType,
    defaultSurfaceMaterial,
    makeBoxHull,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Thin Wall scene into `world`: a static ground box, a thin static wall stood upright, and
 * three fast dynamic projectiles (sphere, capsule, box) fired at it — CCD stops them at the surface.
 */
export function buildThinWall(world: World, _params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const wall = world.createBody({
        type: BodyType.Static,
        position: { x: 0, y: 10, z: 0 },
        rotation: { v: { x: Math.SQRT1_2, y: 0, z: 0 }, s: Math.SQRT1_2 },
    });
    wall.createHull({}, makeBoxHull(10, 0.1, 10));

    const material = { ...defaultSurfaceMaterial(), rollingResistance: 0.1 };
    const shoot = (x: number, create: (b: Body) => void, spin: Vec3): void => {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 10, z: 20 },
            linearVelocity: { x: 0, y: 0, z: -180 },
            angularVelocity: spin,
        });
        create(body);
    };

    shoot(
        -5,
        (b) =>
            b.createSphere(
                { baseMaterial: material },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.1 },
            ),
        { x: 20, y: 0, z: 0 },
    );
    shoot(
        0,
        (b) =>
            b.createCapsule(
                { baseMaterial: material },
                { center1: { x: -0.3, y: 0, z: 0 }, center2: { x: 0.3, y: 0, z: 0 }, radius: 0.1 },
            ),
        { x: 20, y: -5, z: 0 },
    );
    shoot(5, (b) => b.createHull({ baseMaterial: material }, makeBoxHull(0.4, 0.1, 0.1)), {
        x: 20,
        y: 5,
        z: 0,
    });
}
