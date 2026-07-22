// The stage-3 host-layer gate, proven red-first (spec tumble-inline §3), plus the 6b/F3 grab-energy cap.
// Headless (GPU-free) checks over the gold-match oracle + the ported mouse-grab:
//   1. the faithful Paddle authoring reproduces its committed gold bit-exact for every step (GREEN);
//   2. a wrong-axis motor (spin about y, not z) diverges — the oracle catches the class of defect a
//      source-blind port ships (RED). An oracle that has never been red pins nothing;
//   3. a light grab on a settled stack does not launch it — the spring scales to the body's weight, so a
//      held-still grab injects no energy (the smoke suite's grab-regression bound);
//   4. a one-frame cursor whip cannot inject unbounded velocity into the grabbed body — the deterministic
//      red-first for the floor-vanish cap (6b/F3): uncapped the sustained whip runs the body to ~400 m/s,
//      the cap holds it under 250 (the bridge repro's intermittent trusted-input whip can't pin this).
//
// Each check owns one short-lived world (destroyed before the next); the kernel is a process singleton whose
// grow-only regions carry a high-water across worlds, so the divergent runs return early and the grab world
// stays small.

import { expect, test } from "bun:test";
import { type Body, BodyType, makeBoxHull, World } from "@dylanebert/shallot/tumble/core";
import bodyTypeJson from "../../../packages/shallot/tests/tumble/samples/bodies-body-type.json";
import goldJson from "../../../packages/shallot/tests/tumble/samples/joints-paddle.json";
import { buildBodyType, updateBodyType } from "./tumble-body-type";
import { beginGrab, driveGrab, endGrab, updateGrab } from "./tumble-grab";
import { runOracle, type SampleBuild, type SampleGold } from "./tumble-oracle";
import { buildPaddle } from "./tumble-paddle";

const gold = goldJson as unknown as SampleGold;

// A motor spinning about y, where the sample spins about z — every other authoring choice matches the gold,
// so the axis is the only field that can diverge the hash. The defect class the rescoped spec cites: a rotor
// that looks plausible but spins wrong because its physics was authored apart from a source oracle.
const buildWrongAxis: SampleBuild = (world: World, params) => {
    const speed = params.speed as number;
    const Ident = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));
    const pivot = { x: 0, y: 2, z: 0 };
    const anchor = world.createBody({ type: BodyType.Static, position: pivot });
    const paddle = world.createBody({ type: BodyType.Dynamic, position: pivot });
    paddle.createHull({ density: 1 }, makeBoxHull(2.2, 0.2, 0.4));
    world.createMotorJoint(anchor, paddle, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: Ident },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: Ident },
        angularVelocity: { x: 0, y: speed, z: 0 }, // WRONG: the sample spins about z
        maxVelocityTorque: 800,
        maxVelocityForce: 4000,
    });
    for (let i = 0; i < 8; ++i) {
        const b = world.createBody({
            type: BodyType.Dynamic,
            position: { x: -3 + (i % 4) * 2, y: 5 + Math.floor(i / 4), z: -1 + (i % 2) * 2 },
        });
        b.createHull({ density: 0.5 }, makeBoxHull(0.3, 0.3, 0.3));
    }
};

test("faithful Paddle authoring reproduces its gold bit-exact", async () => {
    const result = await runOracle(gold, buildPaddle);
    if (!result.pass) {
        throw new Error(
            `diverged at step ${result.step}: got ${result.got}, expected ${result.expected}`,
        );
    }
    expect(result.pass).toBe(true);
    expect(result.steps).toBe(gold.stepCount);
});

test("wrong-axis motor diverges — the oracle catches it (red-first)", async () => {
    const result = await runOracle(gold, buildWrongAxis);
    // The evidence the gate requires: the same oracle goes RED on the wrong authoring, reporting the first
    // divergent step. A green run of this build would mean the oracle can't tell a wrong axis from a right one.
    if (result.pass) throw new Error("wrong-axis build unexpectedly matched the gold");
    console.log(
        `[red-first] wrong-axis oracle diverged at step ${result.step}: got ${result.got}, expected ${result.expected}`,
    );
    expect(result.pass).toBe(false);
    // the divergence must land inside the recorded trajectory (not off the end) — the `if (result.pass)`
    // throw above is the load-bearing red check; this pins that `step` is a real trajectory index.
    expect(result.step).toBeLessThan(gold.stepCount);
});

test("a light grab does not launch a settled stack", async () => {
    // A minimal settled stack (lighter than the paddle world), grabbed with a held-still target. The spring
    // scales to the body's weight (100·mg), so it holds the box in place rather than yanking it — the flat
    // constant that the smoke suite's regression pins would rip a light body loose.
    const dt = gold.timeStep;
    const sub = gold.subStepCount;
    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    try {
        const ground = world.createBody({ type: BodyType.Static });
        ground.createHull({}, makeBoxHull(50, 1, 50));
        const boxes: Body[] = [];
        for (let i = 0; i < 3; i++) {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: 1.3 + i * 0.62, z: 0 },
            });
            b.createHull({ density: 0.5 }, makeBoxHull(0.3, 0.3, 0.3));
            boxes.push(b);
        }

        // settle
        for (let i = 0; i < 120; i++) world.step(dt, sub);
        const settledSpeed = Math.max(
            ...boxes.map((b) => {
                const v = b.getLinearVelocity();
                return Math.hypot(v.x, v.y, v.z);
            }),
        );
        expect(settledSpeed).toBeLessThan(0.1); // the stack is genuinely at rest before the grab

        // grab the top box with a downward pick ray, then hold the target still and step
        const origin = { x: 0, y: 6, z: 0 };
        const down = { x: 0, y: -10, z: 0 };
        const grab = beginGrab(world, origin, down);
        expect(grab).not.toBeNull();
        for (let i = 0; i < 40; i++) {
            updateGrab(grab!, origin, { x: 0, y: -1, z: 0 });
            driveGrab(grab!, dt);
            world.step(dt, sub);
        }
        const grabbedSpeed = Math.max(
            ...boxes.map((b) => {
                const v = b.getLinearVelocity();
                return Math.hypot(v.x, v.y, v.z);
            }),
        );
        endGrab(grab!);
        // a launch would be many m/s; a weight-scaled hold keeps the whole stack near rest.
        expect(grabbedSpeed).toBeLessThan(1.0);
    } finally {
        world.destroy();
    }
});

test("a one-frame cursor whip cannot inject unbounded velocity into the grabbed body", async () => {
    // The floor-vanish root cause (spec tumble-inline 6b): a one-frame cursor teleport (a frame hitch, or
    // trusted synthetic input) drives the kinematic grab anchor to an unbounded velocity, which the motor
    // joint injects into the grabbed body — flinging it (and its joint/contact neighbours) out of frustum.
    // `driveGrab`'s cap bounds the anchor's per-step move to MAX_DRAG_FRACTION·depth, so the injected velocity
    // stays bounded. Deterministic red-first: with the cap removed the same whip injects ~thousands of m/s.
    const dt = gold.timeStep;
    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    try {
        const ground = world.createBody({ type: BodyType.Static });
        ground.createHull({}, makeBoxHull(50, 1, 50));
        const box = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
        box.createHull({ density: 1 }, makeBoxHull(0.3, 0.3, 0.3));

        // grab the box from above (pick depth ≈ 6.7 → cap ≈ 0.25·6.7·60 ≈ 100 m/s anchor speed).
        const origin = { x: 0, y: 12, z: 0 };
        const grab = beginGrab(world, origin, { x: 0, y: -20, z: 0 });
        expect(grab).not.toBeNull();

        // a full-viewport whip held for a few frames (a frame hitch, or a cursor dragged off-screen fast):
        // the target sits 200 m away. Uncapped, the anchor teleports there and the maxed spring accelerates
        // the box unbounded across the gap; the cap keeps the anchor creeping near the box, so the box tracks
        // it at ≈ the anchor's capped speed. Measure the box's PEAK speed over the whip.
        let peak = 0;
        for (let i = 0; i < 30; i++) {
            grab!.target = { x: 200, y: 5, z: 0 };
            driveGrab(grab!, dt);
            world.step(dt, gold.subStepCount);
            const v = box.getLinearVelocity();
            peak = Math.max(peak, Math.hypot(v.x, v.y, v.z));
        }
        endGrab(grab!);
        console.log(
            `[whip-cap] grabbed-body peak speed over a 30-step whip: ${peak.toFixed(1)} m/s`,
        );
        // Bounded by the anchor-velocity cap: the anchor tops out ≈ 0.25·depth·60 ≈ 100 m/s and the sprung
        // body tracks below it. A 250 m/s ceiling clears the capped case with margin; uncapped, the maxed
        // spring across the 200 m gap runs the box far past it.
        expect(peak).toBeLessThan(250);
    } finally {
        world.destroy();
    }
});

test("the per-step update seam replays a sample whose behavior lives in update()", async () => {
    // Proof the seam works: the BodyType sample (bodies.ts) has an empty-ish build and a kinematic sine
    // sweep in update() — its gold only reproduces if update() runs before every step, exactly as
    // the mint's sample.step() did. A build-only host would leave this permanently red. `buildBodyType` /
    // `updateBodyType` (`tumble-body-type.ts`) are the real stage-4 port this seam proof now replays.
    const bodyGold = bodyTypeJson as unknown as SampleGold;

    const withSeam = await runOracle(bodyGold, buildBodyType, updateBodyType);
    if (!withSeam.pass) {
        throw new Error(
            `update-seam replay diverged at step ${withSeam.step}: got ${withSeam.got}, expected ${withSeam.expected}`,
        );
    }
    expect(withSeam.pass).toBe(true);

    // without the seam the same build goes red — proof the update() is what the gold encodes, not just the
    // build (the kinematic platform sits still, so the crates land differently).
    const withoutSeam = await runOracle(bodyGold, buildBodyType);
    expect(withoutSeam.pass).toBe(false);
});
