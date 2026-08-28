/**
 * Headless AVBD single-step differential: seed one fixed contact state into a standalone
 * `PhysicsStep`, execute exactly one step on lavapipe, read its solver state through
 * `probeBuffer`, and compare it with the existing f64 oracle advanced from the same state and
 * with the GPU's actual coloring.
 *
 * The envelopes are derived before observing this arm: one f32 solve performs 10 sweeps, and
 * each sweep's pose update is rounded at roughly one f32 ulp near this origin-scale fixture
 * (2^-23 m). The coupled 6x6 LDL solve plus contact stamp has fewer than 1600 rounded scalar
 * operations per sweep; charging one ulp to every operation gives 10 * 1600 * 2^-23 = 1.91e-3 m,
 * rounded outward to a 2e-3 m position envelope. BDF1 recovers velocity by dividing the pose delta
 * by dt=1/60, so its propagated position envelope is 60 * 2e-3 = 1.2e-1 m/s; the velocity envelope
 * is exactly that derived value. These bounds depend on the
 * oracle's operation count and fixed timestep, never on a lavapipe-recorded miss.
 *
 * Red proof (integer-division mutant class): replacing the fixed-step ratio
 * `const h = p.dt / this._substeps` in `src/standard/avbd/step.ts` with
 * `const h = Math.trunc(p.dt) / this._substeps` (the numerator is truncated as an integer before
 * division), then running `bun test ./packages/shallot/tests/avbd/differential.tier.ts`, exits 1:
 * body 1 position error is NaN. The correct floating-point ratio is restored in the tracked tree.
 */

import { describe, expect, test } from "bun:test";
import { requestGPU } from "../../src/engine/runtime/gpu";
import { probeBuffer } from "../../src/engine/runtime/probe";
import { BODY_VEC4, PhysicsStep } from "../../src/standard/avbd/step";
import type { Vec3 } from "./math";
import { type Body, body } from "./rigid";
import { makeSolver, step } from "./solver";

const CAPACITY = 8;
const DT = 1 / 60;
const ITERATIONS = 10;
const POSITION_TOLERANCE = 2e-3;
const VELOCITY_TOLERANCE = 60 * POSITION_TOLERANCE;

const scene = (): Body[] => [
    body([10, 1, 10], 0, 0.5, [0, 0, 0]),
    body([1, 1, 1], 1, 0.5, [0, 0.99, 0], [0.1, -0.2, 0.05]),
];

const clone = (b: Body): Body =>
    body([...b.size] as Vec3, b.mass, b.friction, [...b.posLin] as Vec3, [...b.velLin] as Vec3, [
        ...b.posAng,
    ]);

function seed(bodies: Body[]): Float32Array {
    const out = new Float32Array(CAPACITY * BODY_VEC4 * 4);
    const put = (i: number, col: number, values: readonly number[]): void => {
        out.set(values, (col * CAPACITY + i) * 4);
    };
    for (let i = 0; i < bodies.length; i++) {
        const b = bodies[i];
        put(i, 0, [...b.posLin, 0]);
        put(i, 1, b.posAng);
        put(i, 2, [...b.posLin, 0]);
        put(i, 3, b.posAng);
        put(i, 4, [...b.posLin, 0]);
        put(i, 5, b.posAng);
        put(i, 6, [...b.velLin, 0]);
        put(i, 7, [...b.velAng, 0]);
        put(i, 8, [...b.prevVelLin, 0]);
        put(i, 9, [...b.moment, b.mass]);
        put(i, 10, [b.size[0] / 2, b.size[1] / 2, b.size[2] / 2, b.friction]);
        put(i, 11, [0, 0, 0, 0]);
    }
    return out;
}

const lane3 = (state: Float32Array, col: number, i: number): Vec3 => {
    const at = (col * CAPACITY + i) * 4;
    return [state[at], state[at + 1], state[at + 2]];
};

const distance = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("headless AVBD GPU readback differential", () => {
    test("one fixed contact step matches the f64 CPU oracle", async () => {
        const { device } = await requestGPU();
        const physics = await PhysicsStep.create(device, CAPACITY, CAPACITY);
        try {
            const initial = scene();
            device.queue.writeBuffer(physics.bodies, 0, seed(initial));
            device.queue.writeBuffer(
                physics.colors,
                0,
                Uint32Array.from({ length: CAPACITY }, (_, i) => i),
            );
            physics.configure({
                dt: DT,
                gravity: -10,
                alpha: 0.99,
                penalty: 1e5,
                betaLin: 1e4,
                gamma: 0.999,
                iterations: ITERATIONS,
                maxColors: 8,
                smallN: 1024,
                ldsN: 64,
                substeps: 1,
            });
            physics.gateSetCount(initial.length);
            physics.cold();
            const encoder = device.createCommandEncoder();
            physics.record(encoder);
            device.queue.submit([encoder.finish()]);

            const [bodyProbe, colorProbe] = await Promise.all([
                probeBuffer(device, physics.bodies, {
                    offset: 0,
                    size: CAPACITY * BODY_VEC4 * 16,
                    label: "avbd-differential-bodies",
                }),
                probeBuffer(device, physics.colors, {
                    offset: 0,
                    size: CAPACITY * 4,
                    label: "avbd-differential-colors",
                }),
            ]);
            const gpu = new Float32Array(bodyProbe.bytes);
            const rawColors = new Uint32Array(colorProbe.bytes);
            const oracleBodies = initial.map(clone);
            const oracle = makeSolver(oracleBodies, {
                dt: DT,
                gravity: -10,
                iterations: ITERATIONS,
                alpha: 0.99,
                betaLin: 1e4,
                gamma: 0.999,
                penaltyStiffness: 1e5,
                layer: "warmstart",
                substeps: 1,
            });
            const colors = oracleBodies.map((_, i) =>
                rawColors[i] === 0xffffffff ? 0 : rawColors[i],
            );
            step(oracle, { kind: "colored", colors });

            for (let i = 0; i < oracleBodies.length; i++) {
                expect(
                    distance(lane3(gpu, 0, i), oracleBodies[i].posLin),
                    `body ${i} position`,
                ).toBeLessThanOrEqual(POSITION_TOLERANCE);
                expect(
                    distance(lane3(gpu, 6, i), oracleBodies[i].velLin),
                    `body ${i} velocity`,
                ).toBeLessThanOrEqual(VELOCITY_TOLERANCE);
            }
        } finally {
            physics.destroy();
            device.destroy();
        }
    });
});
