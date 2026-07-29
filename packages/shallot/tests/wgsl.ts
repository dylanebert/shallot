import { expect } from "bun:test";

// Shared helpers for the emitted-WGSL structural tests — the device-free seam every ported kernel
// exposes as a `*Wgsl()` function. Kept here rather than duplicated per test file: by the end of the
// TypeGPU port there is one of these seams per kernel across render, sear, physics and the BVH.

/** the emitted body of one function or entry point, brace-matched from its signature */
export function body(src: string, signature: string): string {
    const start = src.indexOf(signature);
    expect(start).toBeGreaterThanOrEqual(0);
    let depth = 0;
    for (let i = src.indexOf("{", start); i < src.length; i++) {
        if (src[i] === "{") depth++;
        else if (src[i] === "}" && --depth === 0) return src.slice(start, i + 1);
    }
    throw new Error(`unterminated body for ${signature}`);
}

/** collapse runs of whitespace so an assertion pins the emitted *code*, not typegpu's formatting */
export function flat(src: string): string {
    return src.replace(/\s+/g, " ");
}

/**
 * assert a kernel keeps TGSL's two silent-wrong integer classes out of its emitted WGSL: a bare JS
 * literal seeds an `i32` (flipping the arithmetic signed and spraying conversions), and a bare `/` on
 * integer operands transpiles to `f32(a) / f32(b)` — a fractional quotient, wrong for any inexact
 * division, not merely above 2²⁴. Both look correct in the TypeScript.
 */
export function integerDiscipline(src: string): void {
    // conversions, and i32-suffixed literals in arithmetic position (an array subscript is legal and
    // has no arithmetic consequence, so `arr[0i]` is allowed through)
    expect(flat(src)).not.toMatch(/\bi32\(/);
    expect(flat(src).replace(/\[[^\]]*\]/g, "[]")).not.toMatch(/\b\d+i\b/);
}
