// The tumble.js `RayCurtain` sample (`samples/src/samples/collision.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. Four kinematic spinning obstacles — sphere, capsule, box hull, rock mesh —
// laid out for the sweeping ray curtain to sample. The curtain is the sample's `render()` overlay
// ({@link renderRayCurtain}): a per-frame `castRayClosest` sweep drawn as lines + hit points — a read-only
// query, outside the gold contract, so only `build()` feeds it. No knobs, no `update()`.
//
// Creation order is load-bearing for the hash: sphere, capsule, box, rock — the sample's exact order.

import {
    BodyType,
    createRock,
    makeBoxHull,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

/**
 * Author the Ray Curtain scene into `world`: four kinematic spinning shapes in a row. No knobs — `params`
 * is unused, kept for the shared `SampleBuild` signature.
 */
export function buildRayCurtain(world: World, _params: SampleParams): void {
    const spin = { x: 0.8, y: 0.4, z: 0.8 };
    const at = (x: number): { type: BodyType; position: Vec3; angularVelocity: Vec3 } => ({
        type: BodyType.Kinematic,
        position: { x, y: 3, z: 0 },
        angularVelocity: spin,
    });

    world.createBody(at(-6)).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.9 });
    world
        .createBody(at(-2))
        .createCapsule(
            {},
            { center1: { x: -0.5, y: 0, z: 0 }, center2: { x: 0.5, y: 0, z: 0 }, radius: 0.8 },
        );
    world.createBody(at(2)).createHull({}, makeBoxHull(0.7, 0.7, 0.7));
    world.createBody(at(6)).createHull({}, createRock(0.9));
}

/**
 * The sweeping ray curtain (the sample's `render()`): a rank of downward rays across the row, offset in Z by
 * a slow sine of `stepCount`. Each ray's closest hit draws a surface point + normal; a clear ray draws its
 * full span dim.
 */
export const renderRayCurtain: SampleRender = (
    draw: Overlay,
    world: World,
    _params: SampleParams,
    stepCount: number,
) => {
    const offset = 2.5 * Math.sin(0.02 * stepCount);
    for (let x = -8; x <= 8; x += 0.25) {
        const origin = { x, y: 8, z: offset };
        const translation = { x: 0, y: -8, z: 0 };
        const r = world.castRayClosest(origin, translation);
        if (r.hit) {
            const tip = {
                x: r.point.x + 0.5 * r.normal.x,
                y: r.point.y + 0.5 * r.normal.y,
                z: r.point.z + 0.5 * r.normal.z,
            };
            draw.line(r.point, tip, 0x30d030);
            draw.point(r.point, 4, 0x30d030);
        } else {
            draw.line(origin, { x, y: 0, z: offset }, 0x404850);
        }
    }
};
