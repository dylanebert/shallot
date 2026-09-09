// The tumble.js `Suspension` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. A platform hung from four corner distance-joint springs. Dropping crates
// on it compresses the springs and it bobs — the distance joint in its soft-spring mode. The `stiffness`
// knob is the springs' hertz.
//
// Creation order is load-bearing for the hash: ground, static frame, platform, the four corner distance
// joints, then the three dropped crates — the sample's exact order.

import { BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams } from "./tumble-oracle";

/**
 * Author the Suspension scene into `world`, reading the `stiffness` knob (spring hertz on all four corner
 * distance joints). A static ground box, a static frame plate, a density-2 platform hung from it by four
 * soft distance-joint springs, and three density-3 crates dropped on top.
 */
export function buildSuspension(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const frameBody = world.createBody({ type: BodyType.Static, position: { x: 0, y: 8, z: 0 } });
    frameBody.createHull({}, makeBoxHull(2.2, 0.08, 2.2));
    const platform = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
    platform.createHull({ density: 2 }, makeBoxHull(2, 0.2, 2));

    const hz = params.stiffness as number;
    for (const [cx, cz] of [
        [-1.8, -1.8],
        [1.8, -1.8],
        [-1.8, 1.8],
        [1.8, 1.8],
    ]) {
        world.createDistanceJoint(frameBody, platform, {
            localFrameA: { p: { x: cx, y: 0, z: cz }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            localFrameB: { p: { x: cx, y: 0, z: cz }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            length: 3,
            enableSpring: true,
            hertz: hz,
            dampingRatio: 0.5,
        });
    }

    // Dropped from below the frame plate so they land on the platform, not the frame.
    for (let i = 0; i < 3; ++i) {
        const crate = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -1 + i, y: 6.4 + i * 0.5, z: 0 },
        });
        crate.createHull({ density: 3 }, makeBoxHull(0.5, 0.5, 0.5));
    }
}
