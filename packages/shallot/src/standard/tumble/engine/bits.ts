// Integer bit-twiddling helpers, ported from Box3D's ctz.h (Erin Catto, MIT).
//
// These are exact integer ops — no f32 arithmetic — so no fround discipline applies. JS
// `Math.clz32` matches b3CLZ32 exactly, including returning 32 for input 0. Population count
// uses the SWAR bit trick (Box3D delegates to __popcnt/__builtin_popcount; the result is what
// matters, not the instruction).

/** Count leading zeros of a 32-bit value. Returns 32 for 0, matching b3CLZ32. */
export const clz32 = Math.clz32;

/** Population count of a 32-bit value. */
export function popCount32(x: number): number {
    let v = x >>> 0;
    v = v - ((v >>> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    v = (v + (v >>> 4)) & 0x0f0f0f0f;
    return (v * 0x01010101) >>> 24;
}

export function isPowerOf2(x: number): boolean {
    return (x & (x - 1)) === 0;
}

/** ceil(log2(x)): the smallest exponent e with 2^e >= x. */
export function boundingPowerOf2(x: number): number {
    if (x <= 1) {
        return 1;
    }
    return 32 - clz32((x - 1) >>> 0);
}

/** Smallest power of two >= x. */
export function roundUpPowerOf2(x: number): number {
    if (x <= 1) {
        return 1;
    }
    return 1 << (32 - clz32((x - 1) >>> 0));
}

/** floor(log2(x)): position of the most significant set bit. Requires x > 0. */
export function lowerPowerOf2Exponent(x: number): number {
    return 31 - clz32(x >>> 0);
}

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
