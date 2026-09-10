import { FrameGpu } from "@dylanebert/shallot/render/core";
import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

export const PARTICLE_COUNT = 50_000;
export const PARTICLE_WORKGROUP = 64;

const GRAVITY = 9.8;
const UP_MIN = 6.5;
const UP_MAX = 8.0;
const SPREAD = 1.5;
export const SPAWN_Y = 0.05;
const GROUND_Y = 0.0;
const TAU = 6.2831853;
const GOLDEN = 0x9e3779b9;

/** The one CPU/GPU particle layout, shared by the compute writer and the Sear vertex reader. */
export const Particle = d
    .struct({
        posSeed: d.vec4f,
        vel: d.vec4f,
    })
    .$name("Particle");
export const ParticleArray = d.arrayOf(Particle, PARTICLE_COUNT);
export const PARTICLE_BYTES = d.sizeOf(ParticleArray);
export type ParticleBuffer = TgpuBuffer<typeof ParticleArray> & StorageFlag;

export const particleLayout = tgpu.bindGroupLayout({
    frame: { uniform: FrameGpu },
    particles: { storage: ParticleArray, access: "mutable" },
});

// A raw-WGSL body: the hash needs u32 wraparound on multiply, the shader's native arithmetic. JS spells
// that too (`Math.imul`), so this is GPU-only for want of a CPU caller, not for want of a JS twin.
const hashU32 = tgpu
    .fn(
        [d.u32],
        d.u32,
    )(
        /* wgsl */ `(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}`,
    )
    .$name("hashU32");

const rnd = tgpu
    .fn(
        [d.u32],
        d.f32,
    )((s) => {
        "use gpu";
        return d.f32(hashU32(s)) * d.f32(1.0 / 4294967296.0);
    })
    .$name("particleRnd");

/** One launch velocity: a random azimuth, a disc-uniform spread, a random speed up the y axis. */
const jet = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((seed) => {
        "use gpu";
        const ang = rnd(seed) * d.f32(TAU);
        const rad = std.sqrt(rnd(seed + d.u32(GOLDEN))) * d.f32(SPREAD);
        const up = std.mix(d.f32(UP_MIN), d.f32(UP_MAX), rnd(seed + d.u32(0x85ebca6b)));
        return d.vec3f(std.cos(ang) * rad, up, std.sin(ang) * rad);
    })
    .$name("particleJet");

/** `posSeed.w` is the seeded flag: an unseeded particle jumps to a random point along its own
 *  ballistic arc, so the jet is already full the frame the buffer is first stepped instead of
 *  climbing as one visible slug. */
const integrateParticle = tgpu
    .fn(
        [d.u32, d.vec4f, d.vec4f, FrameGpu],
        Particle,
    )((i, posSeed, vel, frame) => {
        "use gpu";
        const g = d.vec3f(0, -GRAVITY, 0);
        let pos = posSeed.xyz;
        let nextVel = vel.xyz;

        if (posSeed.w < d.f32(0.5)) {
            const v0 = jet(i + d.u32(0x01234567));
            const flight = std.div(std.mul(d.f32(2.0), v0.y), d.f32(GRAVITY));
            const t = rnd(i * d.u32(2) + d.u32(1)) * flight;
            pos = std.add(
                std.add(d.vec3f(0, SPAWN_Y, 0), std.mul(v0, t)),
                std.mul(std.mul(g, t), t * d.f32(0.5)),
            );
            nextVel = std.add(v0, std.mul(g, t));
        } else {
            const dt = frame.dt;
            nextVel = std.add(nextVel, std.mul(g, dt));
            pos = std.add(pos, std.mul(nextVel, dt));
            if (pos.y <= d.f32(GROUND_Y) && nextVel.y < d.f32(0.0)) {
                nextVel = jet(i ^ (frame.frame * d.u32(GOLDEN)));
                pos = d.vec3f(0, SPAWN_Y, 0);
            }
        }

        return Particle({ posSeed: d.vec4f(pos, d.f32(1)), vel: d.vec4f(nextVel, d.f32(0)) });
    })
    .$name("integrateParticle");

/** One invocation per particle, one dispatch per frame: the buffer is stepped in place, so the
 *  simulation state never crosses the PCI bus. */
export const integrateKernel = tgpu
    .computeFn({
        workgroupSize: [PARTICLE_WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= d.u32(PARTICLE_COUNT)) return;
        const particle = particleLayout.$.particles[i];
        particleLayout.$.particles[i] = integrateParticle(
            i,
            particle.posSeed,
            particle.vel,
            particleLayout.$.frame,
        );
    })
    .$name("particles-integrate");
