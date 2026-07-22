// The gold-match oracle for the tumble sample host: rebuild a sample's world from the escape-hatch build,
// step it N times at knob defaults, and compare each step's `hashWorldState` against the committed gold
// trajectory (tests/tumble/samples/<slug>.json). Both sides are the same engine, so a match is bit-exact
// and only authoring can differ — a wrong axis, wrong joint, wrong shape, or wrong body set diverges the
// hash at the first divergent step, which this reports. Pure engine surface, no GPU: the host runs it at
// build time (before the render world) and the headless evidence script runs it standalone.

import { hashWorldState, init, World } from "@dylanebert/shallot/tumble/core";

/** the resolved knob values a sample's build reads — the gym {@link Params} shape, kept local so the oracle
 *  stays GPU-free. */
export type SampleParams = Record<string, number | boolean | string>;

/** authors a sample's scene into a live `World`, reproducing the tumble.js sample's `build()` — the one
 *  function the oracle replays and the host renders, so a rendered scene can't drift from its verified one. */
export type SampleBuild = (world: World, params: SampleParams) => void;

/** the per-step hook run just before each `world.step`, reproducing the tumble.js sample's `update(dt)`
 *  (`sample.ts` `step()`: `update` then `world.step`). `stepCount` is the pre-step count (0 on the first
 *  step, matching `Sample.stepCount`), so a sine sweep or a scheduled event lands on the same tick as the
 *  gold. The oracle and the live stepper MUST call it identically or the live view drifts from the gold. */
export type SampleUpdate = (
    world: World,
    params: SampleParams,
    dt: number,
    stepCount: number,
) => void;

/** the committed gold trajectory — one sample run headless at knob defaults (see the mint,
 *  `scripts/gen-tumble-sample-golds.ts`). The host consumes the imported JSON as this shape. */
export interface SampleGold {
    slug: string;
    name: string;
    timeStep: number;
    subStepCount: number;
    gravity: [number, number, number];
    enableSleep: boolean;
    enableContinuous: boolean;
    stepCount: number;
    camera: {
        pivot: [number, number, number];
        yaw: number;
        pitch: number;
        radius: number;
        fov: number;
        near: number;
        far: number;
    };
    knobs: Knob[];
    hashes: string[];
}

type Knob =
    | { kind: "slider"; key: string; default: number }
    | { kind: "toggle"; key: string; default: boolean }
    | { kind: "select"; key: string; default: string }
    | { kind: "button"; key: string };

/** the oracle result: bit-exact for every step, or the first divergent step index + the mismatch. */
export type OracleResult =
    | { pass: true; steps: number }
    | { pass: false; step: number; expected: string; got: string };

/** the knob defaults a sample was minted at — the params the gold trajectory assumes. */
export function goldParams(gold: SampleGold): SampleParams {
    const p: SampleParams = {};
    for (const k of gold.knobs) {
        if (k.kind !== "button") p[k.key] = k.default;
    }
    return p;
}

const toHex = (h: bigint): string => `0x${h.toString(16).padStart(16, "0")}`;

/**
 * Replay `build` (+ the optional per-step `update`) against the gold's world def at knob defaults and
 * compare each step's world-state hash to the gold. `update` runs before every `world.step`, exactly as the
 * mint's `sample.step()` did — a sample whose behavior lives in `update()` (a kinematic sweep, a scheduled
 * joint cut) only replays bit-exact through this hook. Single-threaded (the gold contract is
 * thread-count-independent; a fresh ST kernel matches the mint by construction). Destroys its world; safe to
 * call before the host builds its render world.
 */
export async function runOracle(
    gold: SampleGold,
    build: SampleBuild,
    update?: SampleUpdate,
): Promise<OracleResult> {
    await init({ threads: 0 });
    const params = goldParams(gold);
    const world = new World({
        gravity: { x: gold.gravity[0], y: gold.gravity[1], z: gold.gravity[2] },
        enableSleep: gold.enableSleep,
        enableContinuous: gold.enableContinuous,
    });
    try {
        build(world, params);
        for (let i = 0; i < gold.stepCount; i++) {
            update?.(world, params, gold.timeStep, i);
            world.step(gold.timeStep, gold.subStepCount);
            const got = toHex(hashWorldState(world.state));
            if (got !== gold.hashes[i]) {
                return { pass: false, step: i, expected: gold.hashes[i], got };
            }
        }
        return { pass: true, steps: gold.stepCount };
    } finally {
        world.destroy();
    }
}
