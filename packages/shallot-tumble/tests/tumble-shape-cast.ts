// The tumble.js `ShapeCast` sample (`samples/src/samples/collision.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. Three rows of the same four kinematic spinning obstacles (sphere, capsule,
// box hull, rock mesh) for the swept-sphere cast to sample. The cast is the sample's `render()` overlay
// ({@link renderShapeCast}): a rank of spheres swept in +Z, each resolved to its first contact — a
// read-only query, outside the gold contract, so only `build()` feeds it. No knobs, no `update()`.
//
// Creation order is load-bearing for the hash: each row's sphere, capsule, box, rock — the sample's exact
// nested-loop order.

import {
    BodyType,
    type CastHit,
    createRock,
    makeBoxHull,
    type ShapeProxy,
    type Vec3,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

// closest-hit collector for a shape cast: return the running fraction so the query keeps only the nearest
// contact (the sample's `castClosest`, `b3RayCastClosestCallback`).
type Nearest = { point: Vec3; normal: Vec3; fraction: number; hit: boolean };
function castClosest(world: World, origin: Vec3, proxy: ShapeProxy, translation: Vec3): Nearest {
    const best: Nearest = { point: origin, normal: { x: 0, y: 0, z: 0 }, fraction: 1, hit: false };
    world.castShape(origin, proxy, translation, (h: CastHit) => {
        best.point = h.point;
        best.normal = h.normal;
        best.fraction = h.fraction;
        best.hit = true;
        return h.fraction;
    });
    return best;
}

/**
 * Author the Shape Cast scene into `world`: three rows of four kinematic spinning shapes. No knobs —
 * `params` is unused, kept for the shared `SampleBuild` signature.
 */
export function buildShapeCast(world: World, _params: SampleParams): void {
    const spin = { x: 0.6, y: 0.5, z: 0.7 };
    for (let i = 0; i < 3; ++i) {
        const y = 2 + 2 * i;
        const at = (x: number): { type: BodyType; position: Vec3; angularVelocity: Vec3 } => ({
            type: BodyType.Kinematic,
            position: { x, y, z: 0 },
            angularVelocity: spin,
        });
        world.createBody(at(-6)).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.9 });
        world
            .createBody(at(-2))
            .createCapsule(
                {},
                { center1: { x: -0.5, y: 0, z: 0 }, center2: { x: 0.5, y: 0, z: 0 }, radius: 0.7 },
            );
        world.createBody(at(2)).createHull({}, makeBoxHull(0.7, 0.7, 0.7));
        world.createBody(at(6)).createHull({}, createRock(0.8));
    }
}

/**
 * The swept-sphere cast (the sample's `render()`): a 4×3 rank of spheres swept +Z through the spinning
 * shapes. Each cast draws the mover at its start (green), and — where it strikes — at the contact fraction
 * (red) with the surface point + normal; a clear path draws the mover at the far end (gray).
 */
export const renderShapeCast: SampleRender = (draw: Overlay, world: World) => {
    const radius = 0.3;
    const proxy: ShapeProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius };
    const translation = { x: 0, y: 0, z: 10 };
    for (let col = 0; col < 4; ++col) {
        for (let row = 0; row < 3; ++row) {
            const origin = { x: -6 + 4 * col, y: 2 + 2 * row, z: -5 };
            const sphere = { center: origin, radius };
            draw.solidSphere({ p: origin, q: IDENT }, sphere, 0x30a030);
            const hit = castClosest(world, origin, proxy, translation);
            if (hit.hit) {
                const end = {
                    x: origin.x + hit.fraction * translation.x,
                    y: origin.y + hit.fraction * translation.y,
                    z: origin.z + hit.fraction * translation.z,
                };
                draw.solidSphere({ p: end, q: IDENT }, { center: end, radius }, 0xd03030);
                draw.point(hit.point, 4, 0xffe000);
                draw.line(
                    hit.point,
                    {
                        x: hit.point.x + 0.3 * hit.normal.x,
                        y: hit.point.y + 0.3 * hit.normal.y,
                        z: hit.point.z + 0.3 * hit.normal.z,
                    },
                    0xffe000,
                );
            } else {
                const end = { x: origin.x, y: origin.y, z: origin.z + translation.z };
                draw.solidSphere({ p: end, q: IDENT }, { center: end, radius }, 0x606060);
            }
        }
    }
};
