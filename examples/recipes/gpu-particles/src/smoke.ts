import { Compute, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import { Draws } from "@dylanebert/shallot/render/core";
import { Surfaces } from "@dylanebert/shallot/sear/core";
import * as d from "typegpu/data";
import { PARTICLE_BYTES, PARTICLE_COUNT, Particle, SPAWN_Y } from "./kernel";
import { particleState, particlesStepped } from "./particles";

const READY_MS = 5000;
const STRIDE = d.sizeOf(Particle) / 4; // f32 lanes per particle: posSeed.xyzw + vel.xyzw
const SAMPLE_EVERY = 64;
const SETTLE_FRAMES = 8;
const RISE = 1.0;
const MOVED = 1e-6;
const CHECK = "particles rise off the spawn plane and fall back";
const CHECK_BOUND = "the compute buffer is what the vertex stage binds";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Copy the simulation buffer back to the CPU. Nothing in the recipe's render path does this — the
 *  simulation state never crosses the bus — so it lives here, with the assertions that need it. */
async function readParticles(): Promise<Float32Array> {
    const live = particleState();
    if (!live) throw new Error("gpu-particles: no particle buffer to read");
    const staging = Compute.device.createBuffer({
        label: "particles-readback",
        size: PARTICLE_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let mapped = false;
    try {
        const enc = Compute.device.createCommandEncoder({ label: "particles-readback" });
        enc.copyBufferToBuffer(live.raw, 0, staging, 0, PARTICLE_BYTES);
        Compute.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        mapped = true;
        return new Float32Array(staging.getMappedRange().slice(0));
    } finally {
        if (mapped) staging.unmap();
        staging.destroy();
    }
}

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

/** how many sampled particles moved between two readbacks. Every `survey` clause is a property of one
 *  snapshot, and the first step already scatters the whole jet across its arc — so a simulation frozen
 *  after that step satisfies all of them. This is the clause that reads *motion*. */
function moved(before: Float32Array, after: Float32Array): number {
    let count = 0;
    for (let i = 0; i < PARTICLE_COUNT; i += SAMPLE_EVERY) {
        const at = i * STRIDE;
        const delta =
            Math.abs(after[at] - before[at]) +
            Math.abs(after[at + 1] - before[at + 1]) +
            Math.abs(after[at + 2] - before[at + 2]);
        if (delta > MOVED) count++;
    }
    return count;
}

/** Sear's forward pass resolves a surface layout's own bindings by name out of `Compute.typed`, and
 *  degrades a miss to a warn-once skip — a broken handoff is an empty screen, not an error. So assert it
 *  by identity: the buffer the compute pass writes is the one this draw's surface binds. */
function bound(): { ok: boolean; detail: string } {
    const live = particleState();
    const surface = live ? Surfaces.get(live.draw.surface) : undefined;
    const entries = (surface?.layout.entries ?? {}) as Record<string, unknown>;
    const ok =
        live !== null &&
        Compute.typed.get("particles") === live.typed &&
        Draws.get(live.draw.name) === live.draw &&
        "particles" in entries;
    return {
        ok,
        detail: `draw ${live ? `"${live.draw.name}" -> surface "${live.draw.surface}"` : "unregistered"}, surface bindings [${Object.keys(entries).join(", ")}], Compute.typed."particles" ${live && Compute.typed.get("particles") === live.typed ? "is" : "is not"} the simulation buffer`,
    };
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

            const beforeBuf = await readParticles();
            const before = survey(beforeBuf);
            for (let i = 0; i < SETTLE_FRAMES; i++) await frame();
            const afterBuf = await readParticles();
            const after = survey(afterBuf);

            const expectedLanes = PARTICLE_COUNT * STRIDE;
            const sampled = Math.ceil(PARTICLE_COUNT / SAMPLE_EVERY);
            const stepped = moved(beforeBuf, afterBuf);
            const rose = after.peak > SPAWN_Y + RISE;
            const fell = after.falling > 0;
            const flowing = stepped > sampled / 2;
            const handoff = bound();
            const arc = after.lanes === expectedLanes && after.finite && rose && fell && flowing;
            return {
                ok: arc && handoff.ok,
                checks: [
                    {
                        name: CHECK,
                        ok: arc,
                        detail: `peak y ${before.peak.toFixed(3)} -> ${after.peak.toFixed(3)} (needs > ${(SPAWN_Y + RISE).toFixed(2)}), ${after.falling} sampled particles descending, ${stepped} of ${sampled} sampled particles moved over ${SETTLE_FRAMES} frames, ${after.lanes} of ${expectedLanes} f32 lanes, finite ${after.finite}`,
                    },
                    { name: CHECK_BOUND, ok: handoff.ok, detail: handoff.detail },
                ],
            };
        };
    },
};

export default Smoke;
