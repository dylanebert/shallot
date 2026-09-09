// The tumble.js `JointBreak` sample (`samples/src/samples/events.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. Six boxes hang from a ceiling by distance joints of rising force threshold; a
// downward load ramps up each step, and `update()` reads `getJointEvents()` to cut every joint that crossed
// its threshold last step — a world-mutating event read (joint destruction), load-bearing for the gold.
//
// Creation order is load-bearing for the hash: ground, ceiling, then the six hung boxes + their joints left
// to right — the sample's exact order.

import {
    type Body,
    BodyType,
    type Joint,
    makeBoxHull,
    type World,
} from "@dylanebert/shallot/tumble/core";
import type { SampleUpdate } from "./tumble-oracle";
import type { Overlay, SampleRender } from "./tumble-overlay";

const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

interface Hung {
    joint: Joint;
    body: Body;
    threshold: number;
    broken: boolean;
}

let hung: Hung[] = [];
let load = 0;

/**
 * Author the Joint Break scene into `world`: a ceiling with six boxes hung by distance joints (thresholds
 * 300, 600, …, 1800), taut at each joint's rest length.
 */
export function buildJointBreak(world: World): void {
    hung = [];
    load = 0;
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    const ceiling = world.createBody({ type: BodyType.Static, position: { x: 0, y: 12, z: 0 } });

    const count = 6;
    const length = 3;
    const box = makeBoxHull(0.5, 0.5, 0.5);
    for (let i = 0; i < count; ++i) {
        const x = -10 + 4 * i;
        const anchor = { x, y: 11, z: 0 };
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x, y: 11 - length, z: 0 },
            enableSleep: false,
        });
        body.createHull({ density: 1 }, box);
        const threshold = 300 + 300 * i;
        const joint = world.createDistanceJoint(ceiling, body, {
            localFrameA: { p: ceiling.getLocalPoint(anchor), q: IDENT },
            localFrameB: { p: { x: 0, y: length, z: 0 }, q: IDENT },
            length,
            forceThreshold: threshold,
            userData: i,
        });
        hung.push({ joint, body, threshold, broken: false });
    }
}

/**
 * Cut every joint whose reaction crossed its threshold last step (the sample's `update()`), then ramp the
 * load and apply it to every still-hanging box.
 */
export const updateJointBreak: SampleUpdate = (world: World) => {
    for (const e of world.getJointEvents()) {
        const i = e.userData as number;
        const h = hung[i];
        if (h !== undefined && h.broken === false && h.joint.isValid()) {
            h.joint.destroy();
            h.broken = true;
        }
    }

    load = Math.min(1900, load + 16);
    for (const h of hung) {
        if (h.broken === false) h.body.applyForceToCenter({ x: 0, y: -load, z: 0 }, true);
    }
};

/**
 * Label each box with its break threshold (or "cut" once broken) and read out the live load + broken count
 * (the sample's `render()`). The label per hung box is the demonstration this port restores.
 */
export const renderJointBreak: SampleRender = (draw: Overlay) => {
    let broken = 0;
    for (const h of hung) {
        if (h.broken) broken += 1;
        const p = h.body.getPosition();
        draw.string3d(
            { x: p.x, y: p.y + 0.9, z: p.z },
            h.broken ? "cut" : `${h.threshold | 0}`,
            0xffffff,
        );
    }
    draw.text(`load ${load | 0} N   broken ${broken}/${hung.length}`);
};
