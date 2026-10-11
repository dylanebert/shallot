// Integer bit-twiddling helpers, ported from Box3D's ctz.h (Erin Catto, MIT).
//
// These are exact integer ops — no f32 arithmetic — so no fround discipline applies. JS
// `Math.clz32` matches b3CLZ32 exactly, including returning 32 for input 0.

/** Count leading zeros of a 32-bit value. Returns 32 for 0, matching b3CLZ32. */
export const clz32 = Math.clz32;

/**
 * High 32 bits of a u64, as an unsigned number. The public API carries real u64 bit sets as
 * `bigint`; internals split them into u32 halves so no hot path touches a BigInt.
 *
 * @example
 * const hi = hi32(0xdeadbeefcafef00dn); // 0xdeadbeef
 */
export function hi32(x: bigint): number {
    return Number((x >> 32n) & 0xffffffffn);
}

/** Low 32 bits of a u64, as an unsigned number. */
export function lo32(x: bigint): number {
    return Number(x & 0xffffffffn);
}
