// The deterministic world-state hash — Box3D's b3HashWorldState (recording.c/recording.h, Erin
// Catto, MIT). This is the bit-exact regression contract: the fixture generator emits this FNV-1a
// hash after every step, and the port asserts equality. It walks live bodies in id order, mixing
// each body's transform (position + rotation) and, for awake bodies, its linear/angular velocity.
//
// The hash is a real u64 (wrapping multiply), so it runs in BigInt. Each f32 is mixed as its raw
// IEEE-754 bit pattern (zero-extended to u64), exactly as the C `memcpy(&bits, &f, 4)`.

import { getBodySim, getBodyState } from "./body";
import type { WorldState } from "./world";

const FNV_INIT = 14695981039346656037n;
const FNV_PRIME = 1099511628211n;
const U64_MASK = 0xffffffffffffffffn;

// Scratch views to read a float's raw 32-bit pattern (b3HashFloat's memcpy).
const floatScratch = new Float32Array(1);
const uintScratch = new Uint32Array(floatScratch.buffer);

function mixFloat(hash: bigint, value: number): bigint {
    floatScratch[0] = value;
    const bits = BigInt(uintScratch[0]);
    return ((hash ^ bits) * FNV_PRIME) & U64_MASK;
}

/** @returns the FNV-1a hash of every live body's transform + velocity (b3HashWorldState). */
export function hashWorldState(world: WorldState): bigint {
    let hash = FNV_INIT;

    const bodyCount = world.bodies.length;
    for (let i = 0; i < bodyCount; ++i) {
        const body = world.bodies[i];
        if (body.id !== i) {
            // Free or never-used slot
            continue;
        }

        const sim = getBodySim(world, body);

        hash = mixFloat(hash, sim.transform.p.x);
        hash = mixFloat(hash, sim.transform.p.y);
        hash = mixFloat(hash, sim.transform.p.z);
        hash = mixFloat(hash, sim.transform.q.v.x);
        hash = mixFloat(hash, sim.transform.q.v.y);
        hash = mixFloat(hash, sim.transform.q.v.z);
        hash = mixFloat(hash, sim.transform.q.s);

        const state = getBodyState(world, body);
        if (state !== null) {
            hash = mixFloat(hash, state.linearVelocity.x);
            hash = mixFloat(hash, state.linearVelocity.y);
            hash = mixFloat(hash, state.linearVelocity.z);
            hash = mixFloat(hash, state.angularVelocity.x);
            hash = mixFloat(hash, state.angularVelocity.y);
            hash = mixFloat(hash, state.angularVelocity.z);
        }
    }

    return hash;
}
