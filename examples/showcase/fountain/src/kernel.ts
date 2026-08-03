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
const SPAWN_Y = 0.05;
const GROUND_Y = 0.0;
const TAU = 6.2831853;
const GOLDEN = 0x9e3779b9;
const f32 = Math.fround;

/** The one CPU/GPU particle layout, shared by the compute writer and Sear reader. */
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

const hashU32Wgsl = tgpu
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
    .$name("hashU32Wgsl");

function hashU32Cpu(x: number): number {
    let h = x >>> 0;
    h = (h ^ (h >>> 16)) >>> 0;
    h = Math.imul(h, 0x7feb352d) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
    h = Math.imul(h, 0x846ca68b) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
}

function rndCpu(seed: number): number {
    return f32(f32(hashU32Cpu(seed)) * f32(1.0 / 4294967296.0));
}

function jetCpu(seed: number): d.v3f {
    const ang = f32(rndCpu(seed) * f32(TAU));
    const rad = f32(f32(Math.sqrt(rndCpu((seed + GOLDEN) >>> 0))) * f32(SPREAD));
    const mix = rndCpu((seed + 0x85ebca6b) >>> 0);
    const up = f32(f32(f32(UP_MIN) * f32(1 - mix)) + f32(f32(UP_MAX) * mix));
    return d.vec3f(f32(f32(Math.cos(ang)) * rad), up, f32(f32(Math.sin(ang)) * rad));
}

function integrateParticleCpu(
    i: number,
    posSeed: d.v4f,
    velocity: d.v4f,
    frame: d.Infer<typeof FrameGpu>,
): d.Infer<typeof Particle> {
    let px = posSeed.x;
    let py = posSeed.y;
    let pz = posSeed.z;
    let vx = velocity.x;
    let vy = velocity.y;
    let vz = velocity.z;

    if (posSeed.w < f32(0.5)) {
        const v0 = jetCpu((i + 0x01234567) >>> 0);
        const flight = f32(f32(f32(2) * v0.y) / f32(GRAVITY));
        const t = f32(rndCpu((Math.imul(i, 2) + 1) >>> 0) * flight);
        const halfGravityT2 = f32(f32(f32(f32(0.5) * f32(-GRAVITY)) * t) * t);
        px = f32(v0.x * t);
        py = f32(f32(f32(SPAWN_Y) + f32(v0.y * t)) + halfGravityT2);
        pz = f32(v0.z * t);
        vx = v0.x;
        vy = f32(v0.y + f32(f32(-GRAVITY) * t));
        vz = v0.z;
    } else {
        const dt = frame.dt;
        vy = f32(vy + f32(f32(-GRAVITY) * dt));
        px = f32(px + f32(vx * dt));
        py = f32(py + f32(vy * dt));
        pz = f32(pz + f32(vz * dt));
        if (py <= f32(GROUND_Y) && vy < f32(0)) {
            const relaunched = jetCpu((i ^ Math.imul(frame.frame, GOLDEN)) >>> 0);
            px = 0;
            py = f32(SPAWN_Y);
            pz = 0;
            vx = relaunched.x;
            vy = relaunched.y;
            vz = relaunched.z;
        }
    }

    return Particle({
        posSeed: d.vec4f(px, py, pz, f32(1)),
        vel: d.vec4f(vx, vy, vz, f32(0)),
    });
}

// JS bitwise arithmetic is signed and Number multiplication does not wrap at u32. The folded dual
// keeps the shader's native u32 hash while making the exported CPU arm an exact logic oracle.
export const hashU32 = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((x) => {
        "use gpu";
        return std.isBeingTranspiled() ? hashU32Wgsl(x) : hashU32Cpu(x);
    })
    .$name("hashU32");

const rndGpu = tgpu
    .fn(
        [d.u32],
        d.f32,
    )((s) => {
        "use gpu";
        return d.f32(hashU32(s)) * d.f32(1.0 / 4294967296.0);
    })
    .$name("fountainRndGpu");

const rnd = tgpu
    .fn(
        [d.u32],
        d.f32,
    )((s) => {
        "use gpu";
        return std.isBeingTranspiled() ? rndGpu(s) : rndCpu(s);
    })
    .$name("fountainRnd");

const jetGpu = tgpu
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
    .$name("fountainJetGpu");

export const jet = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((seed) => {
        "use gpu";
        return std.isBeingTranspiled() ? jetGpu(seed) : jetCpu(seed);
    })
    .$name("fountainJet");

const integrateParticleGpu = tgpu
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
    .$name("integrateParticleGpu");

export const integrateParticle = tgpu
    .fn(
        [d.u32, d.vec4f, d.vec4f, FrameGpu],
        Particle,
    )((i, posSeed, vel, frame) => {
        "use gpu";
        return std.isBeingTranspiled()
            ? integrateParticleGpu(i, posSeed, vel, frame)
            : integrateParticleCpu(i, posSeed, vel, frame);
    })
    .$name("integrateParticle");

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
    .$name("fountain-integrate");

const integrateKernelAddressMutated = tgpu
    .computeFn({
        workgroupSize: [PARTICLE_WORKGROUP],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        if (i >= d.u32(PARTICLE_COUNT)) return;
        const at = i + d.u32(1);
        const particle = particleLayout.$.particles[at];
        particleLayout.$.particles[at] = integrateParticle(
            i,
            particle.posSeed,
            particle.vel,
            particleLayout.$.frame,
        );
    })
    .$name("fountain-integrate");

export function fountainIntegrateWgsl(addressMutation = false): string {
    return tgpu.resolve(
        [
            hashU32Wgsl,
            hashU32,
            rndGpu,
            rnd,
            jetGpu,
            jet,
            integrateParticleGpu,
            integrateParticle,
            addressMutation ? integrateKernelAddressMutated : integrateKernel,
        ],
        { names: "strict" },
    );
}
