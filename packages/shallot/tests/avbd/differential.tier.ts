/**
 * Headless AVBD single-step differential: seed one fixed contact state into a standalone
 * `PhysicsStep`, execute exactly one step on lavapipe, read its state through `probeBuffer`, and
 * compare it with the existing f64 oracle advanced from the identical marshaled state and with the
 * GPU's actual coloring.
 *
 * The envelopes are derived before observing the GPU. The ground top is y=0.5 and the dynamic
 * box bottom is y=0.49, so their 0.01 m overlap exercises broadphase, SAT contact generation,
 * coloring, and all ten contact-solve iterations. With u=2^-24, each iteration's contact path has
 * at most 240 rounded scalar operations (basis/arms, C and J, LDL solve, pose update); all position
 * operands are below 11 m and the configured 1e5 penalty cancels through H^-1, so outward
 * first-order propagation is 10·γ240·11 < 1.58e-3 m. POSITION_TOLERANCE=2e-3 m rounds that bound
 * outward. BDF1 then divides the pose interval by dt=1/60; adding two rounded operations at a
 * result below 2 m/s gives 60·2e-3 + γ2·2 < 0.121 m/s, so VELOCITY_TOLERANCE=0.125 m/s is its
 * outward band. Every input to both arms is first materialized through one Float32Array.
 *
 * Red proof (shader integer-division mutant): in `inertialKernel`, replacing the `dt` in
 * `std.mul(vel, dt)` with `d.f32(idiv(d.u32(8), d.u32(8)))` makes the real resolved shader evaluate
 * integer `8u / 8u` before conversion. `bun test ./packages/shallot/tests/avbd/differential.tier.ts`
 * exits 1 with body 1 position error 1.059e-2 m (limit 2e-3): probe readback catches the corrupted
 * contact-step inertial target. The correct floating-point expression is restored in the tree.
 */

import { describe, expect, test } from "bun:test";
import { requestGPU } from "../../src/engine/runtime/gpu";
import { probeBuffer } from "../../src/engine/runtime/probe";
import { BODY_VEC4, PhysicsStep } from "../../src/standard/avbd/step";
import type { Quat, Vec3 } from "./math";
import { type Body, body } from "./rigid";
import { makeSolver, step } from "./solver";

const CAPACITY = 8;
const DT = Math.fround(1 / 60);
const ITERATIONS = 10;
const POSITION_TOLERANCE = 2e-3;
const VELOCITY_TOLERANCE = 0.125;

const scene = (): Body[] => [
    body([10, 1, 10], 0, 0.5, [0, 0, 0]),
    body([1, 1, 1], 1, 0.5, [0, 0.99, 0], [0.1, -0.2, 0.05]),
];

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

/** Reconstruct the oracle bodies from the exact f32 lanes uploaded to the GPU. */
function materialize(upload: Float32Array, count: number): Body[] {
    const at = (i: number, col: number) => (col * CAPACITY + i) * 4;
    const read3 = (i: number, col: number): Vec3 => {
        const o = at(i, col);
        return [upload[o], upload[o + 1], upload[o + 2]];
    };
    return Array.from({ length: count }, (_, i) => {
        const half = read3(i, 10);
        const q = at(i, 1);
        const b = body(
            [half[0] * 2, half[1] * 2, half[2] * 2],
            upload[at(i, 9) + 3],
            upload[at(i, 10) + 3],
            read3(i, 0),
            read3(i, 6),
            [upload[q], upload[q + 1], upload[q + 2], upload[q + 3]] as Quat,
        );
        b.moment = read3(i, 9);
        b.velAng = read3(i, 7);
        b.prevVelLin = read3(i, 8);
        return b;
    });
}

const lane3 = (state: Float32Array, col: number, i: number): Vec3 => {
    const at = (col * CAPACITY + i) * 4;
    return [state[at], state[at + 1], state[at + 2]];
};
const distance = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe("headless AVBD GPU readback differential", () => {
    test("one fixed step matches the f64 CPU oracle", async () => {
        const { device } = await requestGPU();
        const physics = await PhysicsStep.create(device, CAPACITY, CAPACITY);
        try {
            const authored = scene();
            const upload = seed(authored);
            const initial = materialize(upload, authored.length);
            device.queue.writeBuffer(physics.bodies, 0, upload);
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
            const oracleBodies = materialize(upload, initial.length);
            const oracle = makeSolver(oracleBodies, {
                dt: DT,
                gravity: -10,
                iterations: ITERATIONS,
                alpha: Math.fround(0.99),
                betaLin: 1e4,
                gamma: Math.fround(0.999),
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
