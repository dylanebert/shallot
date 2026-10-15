// Box3D ctz.h (Erin Catto, MIT).

/** Count leading zeros of a 32-bit value. Returns 32 for 0, matching b3CLZ32. */
export const clz32 = Math.clz32;

/**
 * High 32 bits of a u64, as an unsigned number.
 * Internals use u32 halves to keep BigInt operations off hot paths.
 */
export function hi32(x: bigint): number {
    return Number((x >> 32n) & 0xffffffffn);
}

/** Low 32 bits of a u64, as an unsigned number. */
export function lo32(x: bigint): number {
    return Number(x & 0xffffffffn);
}
