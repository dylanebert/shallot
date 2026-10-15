import {
    readSimTransform,
    readStateAngularVelocity,
    readStateLinearVelocity,
} from "../kernel/bodycolumns";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { kernel } from "../kernel/kernel";
// Box3D b3HashWorldState (recording.c/recording.h, Erin Catto, MIT).
// Hash raw f32 bits with wrapping u64 multiplication; numeric float conversion would change parity.

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

/** FNV-1a over live bodies in id order: every pose, plus velocities for bodies with an awake state
 * (b3HashWorldState). */
export function hashWorldState(world: WorldState): bigint {
    return hashWorldStateWithSeed(world, FNV_INIT);
}

/** Hash the whitebox lane with the additive B3_ORACLE_SENTINELS diagnostic seed. */
export function hashWorldStateOracleSentinel(world: WorldState): bigint {
    return hashWorldStateWithSeed(world, FNV_INIT ^ 0x9e3779b97f4a7c15n);
}

const transformScratch1 = {
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
};
const linearVelocityScratch8 = { x: 0, y: 0, z: 0 };
const angularVelocityScratch11 = { x: 0, y: 0, z: 0 };

function hashWorldStateWithSeed(world: WorldState, initial: bigint): bigint {
    let hash = initial;

    const bodyCount = kernel(world.ecsState).bodyLength(world.worldId);
    for (let i = 0; i < bodyCount; ++i) {
        const body = i;
        if (bodyField(world, body, BodyField.id) !== i) {
            continue;
        }

        const sim = getBodySim(world, body);

        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).p.x);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).p.y);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).p.z);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).q.v.x);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).q.v.y);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).q.v.z);
        hash = mixFloat(hash, readSimTransform(world, sim, transformScratch1).q.s);

        const state = getBodyState(world, body);
        if (state !== null) {
            hash = mixFloat(hash, readStateLinearVelocity(world, state, linearVelocityScratch8).x);
            hash = mixFloat(hash, readStateLinearVelocity(world, state, linearVelocityScratch8).y);
            hash = mixFloat(hash, readStateLinearVelocity(world, state, linearVelocityScratch8).z);
            hash = mixFloat(
                hash,
                readStateAngularVelocity(world, state, angularVelocityScratch11).x,
            );
            hash = mixFloat(
                hash,
                readStateAngularVelocity(world, state, angularVelocityScratch11).y,
            );
            hash = mixFloat(
                hash,
                readStateAngularVelocity(world, state, angularVelocityScratch11).z,
            );
        }
    }

    return hash;
}
