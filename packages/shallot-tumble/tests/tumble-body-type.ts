// The tumble.js `BodyTypes` sample (`samples/src/samples/bodies.ts`) reproduced near-verbatim through the
// escape-hatch `World` API. A platform carries three dynamic crates; the `type` knob picks its BodyType
// (Kinematic/Dynamic/Static). A kinematic platform sweeps side to side in `update()`, carrying the crates —
// the seam `tumble-pilot.test.ts` proves red/green against this exact gold.
//
// The sample's live `setType` re-typing (its knob is `live: true`) is out of scope: the host renders every
// select knob as a `rebuild` knob (`tumble-sample.ts` `knobParams`), so a type change reloads the scene
// rather than re-typing it in place — the spec's live-knob residue, not a stage-4 concern.
//
// Creation order is load-bearing for the hash: ground, platform, then the three crates — the sample's
// exact order.

import { type Body, BodyType, makeBoxHull, type World } from "@dylanebert/shallot/tumble/core";
import type { SampleParams, SampleUpdate } from "./tumble-oracle";

const TYPES: Record<string, BodyType> = {
    kinematic: BodyType.Kinematic,
    dynamic: BodyType.Dynamic,
    static: BodyType.Static,
};

const PLATFORM = { x: 0, y: 5, z: 0 };
const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };

let platform: Body | null = null;

/**
 * Author the Body Type scene into `world`, reading the `type` knob (kinematic / dynamic / static). A
 * static ground box, a platform of the selected type, and three loose density-2 crates it may carry.
 */
export function buildBodyType(world: World, params: SampleParams): void {
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));

    platform = world.createBody({ type: TYPES[params.type as string], position: PLATFORM });
    platform.createHull({ density: 2 }, makeBoxHull(4, 0.5, 4));

    for (let i = 0; i < 3; i++) {
        const crate = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -2 + 2 * i, y: 7 + 1.5 * i, z: 0 },
        });
        crate.createHull({ density: 2 }, makeBoxHull(0.75, 0.75, 0.75));
    }
}

/**
 * Sweep the platform side to side when `type` is kinematic (the sample's `update()`): a sine-driven
 * `setTargetTransform` toward the swept x, reproducing the sample's kinematic carry. No-op for
 * dynamic/static platforms, matching the source.
 */
export const updateBodyType: SampleUpdate = (
    _world: World,
    params: SampleParams,
    dt: number,
    stepCount: number,
) => {
    if ((params.type as string) !== "kinematic" || platform === null) return;
    const t = stepCount * dt;
    const x = PLATFORM.x + 7 * Math.sin(1.5 * t);
    platform.setTargetTransform({ p: { x, y: PLATFORM.y, z: PLATFORM.z }, q: IDENT }, dt, true);
};
