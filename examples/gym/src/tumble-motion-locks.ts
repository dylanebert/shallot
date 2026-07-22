// The tumble.js `MotionLocks` sample (`samples/src/samples/bodies.ts`) reproduced near-verbatim through
// the escape-hatch `World` API. Four hovering cubes (gravityScale 0) under a constant torque, each with a
// different set of angular locks — rotation appears only about the unlocked axes. No knobs.
//
// Creation order is load-bearing for the hash: free, then lock X,Z, then lock Y, then fully locked — the
// sample's exact order (no ground box — this sample never creates one). The per-cube name labels are the
// sample's `render()` overlay ({@link renderMotionLocks}) — projected HTML, outside the gold + the walk.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams, SampleUpdate } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

let labeled: { body: Body; name: string }[] = [];

/**
 * Author the Motion Locks scene into `world`: four density-1 cubes hovering at y=3 with gravityScale 0 and
 * angularDamping 0.5, each with a different `motionLocks` set (free, lock X+Z, lock Y, fully locked).
 */
export function buildMotionLocks(world: World, _params: SampleParams): void {
    labeled = [];
    const cube = makeBoxHull(0.75, 0.75, 0.75);
    const add = (name: string, x: number, locks: Partial<Record<string, boolean>>): void => {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 3, z: 0 },
            gravityScale: 0,
            angularDamping: 0.5,
            motionLocks: {
                linearX: false,
                linearY: false,
                linearZ: false,
                angularX: false,
                angularY: false,
                angularZ: false,
                ...locks,
            },
        });
        body.createHull({ density: 1 }, cube);
        labeled.push({ body, name });
    };

    add("free", -4.5, {});
    add("lock X,Z", -1.5, { angularX: true, angularZ: true });
    add("lock Y", 1.5, { angularY: true });
    add("locked", 4.5, { angularX: true, angularY: true, angularZ: true });
}

/** Apply a constant torque to every cube each step (the sample's `update()`). */
export const updateMotionLocks: SampleUpdate = (_world: World) => {
    for (const { body } of labeled) {
        body.applyTorque({ x: 2, y: 2, z: 2 }, true);
    }
};

/** Label each cube with its lock set above the body (the sample's `render()`). */
export const renderMotionLocks: SampleRender = (draw: Overlay) => {
    for (const { body, name } of labeled) {
        const p = body.getPosition();
        draw.string3d({ x: p.x, y: p.y + 1.2, z: p.z }, name, 0xffffff);
    }
};
