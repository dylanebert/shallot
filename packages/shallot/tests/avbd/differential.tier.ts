import { describe, expect, test } from "bun:test";
import { Compute, requestGPU } from "../../src/engine/runtime/gpu";
import { probeBuffer } from "../../src/engine/runtime/probe";
import {
    BODY_VEC4,
    CONSTRAINT_CONTACT,
    CONTACT_META,
    CONTACT_VEC4,
    PhysicsStep,
} from "../../src/standard/avbd/step";
import { type Body, body } from "./rigid";

// Trigger cone: `packages/shallot/src/standard/avbd/**/*.ts` and this tier's direct fixtures. Run from
// the shallot root with `bun test ./packages/shallot/tests/avbd/differential.tier.ts`.
// Red arm: the temporary raw TGSL integer-division mutation makes the geometry-band assertion fail
// after compilation and execution; the focused run recorded 0 pass / 1 fail / 3 expects before restore.

const CAPACITY = 8;
const DT = Math.fround(1 / 60);

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

describe("headless AVBD execution sentinel", () => {
    test("produces the intended contact and a geometry-bounded pose", async () => {
        const previousCompute = { ...Compute };
        let device: GPUDevice | undefined;
        let physics: PhysicsStep | undefined;
        try {
            ({ device } = await requestGPU());
            physics = await PhysicsStep.create(device, CAPACITY, CAPACITY);
            const authored = scene();
            device.queue.writeBuffer(physics.bodies, 0, seed(authored));
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
                iterations: 10,
                maxColors: 8,
                smallN: 1024,
                ldsN: 64,
                substeps: 1,
            });
            physics.gateSetCount(authored.length);
            physics.cold();
            const encoder = device.createCommandEncoder();
            physics.record(encoder);
            device.queue.submit([encoder.finish()]);

            const contactProbe = await probeBuffer(device, physics!.pairContacts, {
                offset: 0,
                size: physics!.recordCap * CONTACT_VEC4 * 16,
                label: "avbd-sentinel-contacts",
            });
            const contactWords = new Uint32Array(contactProbe.bytes);
            const live = Array.from({ length: physics!.recordCap }, (_, rec) => ({
                kind: contactWords[(CONTACT_META * physics!.recordCap + rec) * 4],
                a: contactWords[(CONTACT_META * physics!.recordCap + rec) * 4 + 1],
                b: contactWords[(CONTACT_META * physics!.recordCap + rec) * 4 + 2],
            })).filter(({ kind }) => kind === CONSTRAINT_CONTACT);
            expect(
                live.some(({ a, b }) => a === 1 && b === 0),
                "device contact store contains the authored dynamic-ground pair",
            ).toBe(true);

            const bodyProbe = await probeBuffer(device, physics.bodies, {
                offset: 0,
                size: CAPACITY * BODY_VEC4 * 16,
                label: "avbd-sentinel-bodies",
            });
            const state = new Float32Array(bodyProbe.bytes);
            const pos = [state[4], state[5], state[6]];
            expect(pos.every(Number.isFinite), "device-produced dynamic pose is finite").toBe(true);

            const freeTangentialTravel = Math.abs(authored[1].velLin[0]) * DT;
            expect(
                Math.abs(pos[0]),
                "x stays within one authored free tangential step",
            ).toBeLessThanOrEqual(freeTangentialTravel);
            const groundTop = authored[0].posLin[1] + authored[0].size[1] / 2;
            expect(pos[1], "center stays within the authored ground and box geometry").toBeWithin(
                groundTop,
                groundTop + authored[1].size[1],
            );
        } finally {
            try {
                physics?.destroy();
            } finally {
                try {
                    device?.destroy();
                } finally {
                    for (const key of Object.keys(Compute))
                        delete (Compute as unknown as Record<string, unknown>)[key];
                    Object.assign(Compute, previousCompute);
                }
            }
        }
    });
});
