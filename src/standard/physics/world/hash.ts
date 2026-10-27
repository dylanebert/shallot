import { kernel } from "../kernel/kernel";
import type { WorldState } from "./world";

const FNV_INIT = 14695981039346656037n;

/** FNV-1a over live bodies in id order: every pose, plus velocities for bodies with an awake state
 * (b3HashWorldState). */
export function hashWorldState(world: WorldState): bigint {
    return hashWorldStateWithSeed(world, FNV_INIT);
}

/** Hash the whitebox lane with the additive B3_ORACLE_SENTINELS diagnostic seed. */
export function hashWorldStateOracleSentinel(world: WorldState): bigint {
    return hashWorldStateWithSeed(world, FNV_INIT ^ 0x9e3779b97f4a7c15n);
}

function hashWorldStateWithSeed(world: WorldState, initial: bigint): bigint {
    return BigInt.asUintN(64, kernel(world.ecsState).hashWorldState(world.worldId, initial));
}
