import type { Plugin, State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import * as d from "typegpu/data";
import { PARTICLE_COUNT, Particle, SPAWN_Y } from "./kernel";
import { particlesStepped, readParticles } from "./particles";

const READY_MS = 5000;
const STRIDE = d.sizeOf(Particle) / 4; // f32 lanes per particle: posSeed.xyzw + vel.xyzw
const SAMPLE_EVERY = 64;
const SETTLE_FRAMES = 8;
const RISE = 1.0;
const CHECK = "particles rise off the spawn plane and fall back";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** the highest y a sampled particle reached, how many are on their way back down, and whether the
 *  buffer is the expected size and free of NaNs */
function survey(buffer: Float32Array): {
    peak: number;
    falling: number;
    lanes: number;
    finite: boolean;
} {
    let peak = Number.NEGATIVE_INFINITY;
    let falling = 0;
    let finite = true;
    for (let i = 0; i < PARTICLE_COUNT; i += SAMPLE_EVERY) {
        const at = i * STRIDE;
        const y = buffer[at + 1];
        const vy = buffer[at + 5];
        if (!Number.isFinite(y) || !Number.isFinite(vy)) finite = false;
        if (y > peak) peak = y;
        if (vy < 0 && y > SPAWN_Y) falling++;
    }
    return { peak, falling, lanes: buffer.length, finite };
}

export const Smoke: Plugin = {
    name: "GpuParticlesSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            // A frame *encodes* its dispatch; the render plugin submits at end of frame. So wait for a
            // dispatch from an earlier frame to have been submitted, or the readback is zero-filled and
            // every assertion below is meaningless.
            const start = performance.now();
            while (!particlesStepped() && performance.now() - start < READY_MS) await frame();
            if (!particlesStepped())
                return {
                    ok: false,
                    checks: [{ name: CHECK, ok: false, detail: `no dispatch in ${READY_MS}ms` }],
                };

            const before = survey(await readParticles());
            for (let i = 0; i < SETTLE_FRAMES; i++) await frame();
            const after = survey(await readParticles());

            const expectedLanes = PARTICLE_COUNT * STRIDE;
            const rose = after.peak > SPAWN_Y + RISE;
            const fell = after.falling > 0;
            const ok = after.lanes === expectedLanes && after.finite && rose && fell;
            return {
                ok,
                checks: [
                    {
                        name: CHECK,
                        ok,
                        detail: `peak y ${before.peak.toFixed(3)} -> ${after.peak.toFixed(3)} (needs > ${(SPAWN_Y + RISE).toFixed(2)}), ${after.falling} sampled particles descending, ${after.lanes} of ${expectedLanes} f32 lanes, finite ${after.finite}`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
