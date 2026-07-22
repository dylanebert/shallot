// The tumble.js `Parallel` sample (`samples/src/samples/joints.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Two hovering panels under the same constant torque, applied every step via
// `update()`. One is held to a static reference by a parallel joint — its orientation locked level while
// free to translate — so it barely turns while the free panel tumbles.
//
// Creation order is load-bearing for the hash: ground, static reference, held panel, free panel, then the
// parallel joint — the sample's exact order. No knobs.
//
// The sample's `render()` overlay ({@link renderParallel}) labels each panel — projected HTML, outside the
// gold + the debug-draw walk.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleUpdate } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

let labeled: { body: Body; name: string }[] = [];

/**
 * Author the Parallel scene into `world`. A static ground box, a static reference panel, and two
 * gravity-free dynamic panels — one held level to the reference by a parallel joint, one free — both
 * torqued each step in `update()`.
 */
export function buildParallel(world: World): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const ref = world.createBody({ type: BodyType.Static, position: { x: 0, y: 4, z: 0 } });

    const mk = (x: number) => {
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 4, z: 0 },
            gravityScale: 0,
            angularDamping: 0.2,
        });
        b.createHull({ density: 1 }, makeBoxHull(0.9, 0.12, 0.9));
        return b;
    };

    const held = mk(-2.5);
    const free = mk(2.5);
    world.createParallelJoint(ref, held, { maxTorque: 200, hertz: 4, dampingRatio: 1 });
    labeled = [
        { body: held, name: "parallel joint" },
        { body: free, name: "free" },
    ];
}

/** Apply the sample's constant torque to both panels every step (the sample's `update()`). */
export const updateParallel: SampleUpdate = () => {
    for (const { body } of labeled) body.applyTorque({ x: 1.5, y: 1.5, z: 0 }, true);
};

/** Label the held vs free panel (the sample's `render()`). */
export const renderParallel: SampleRender = (draw: Overlay) => {
    for (const { body, name } of labeled) {
        const p = body.getPosition();
        draw.string3d({ x: p.x, y: p.y + 1, z: p.z }, name, 0xffffff);
    }
};
