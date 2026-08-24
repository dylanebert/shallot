import { expect } from "bun:test";

// Shared helpers for the emitted-WGSL structural tests — the device-free seam every ported kernel
// exposes as a `*Wgsl()` function. Kept here rather than duplicated per test file: by the end of the
// TypeGPU port there is one of these seams per kernel across render, sear, physics and the BVH.

/** the emitted body of one function or entry point, brace-matched from its signature */
export function body(src: string, signature: string): string {
    const start = src.indexOf(signature);
    expect(start).toBeGreaterThanOrEqual(0);
    const openBrace = src.indexOf("{", start);
    if (openBrace < 0) throw new Error(`no opening brace after signature: ${signature}`);
    let depth = 0;
    for (let i = openBrace; i < src.length; i++) {
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
 * assert no integer division slipped past `idiv`. TGSL transpiles `a / b` on integer operands to
 * `f32(a) / f32(b)` — a *fractional* quotient, wrong for any inexact division, not merely above 2²⁴,
 * and it looks correct in the TypeScript. Three assertions, because one shape does not cover it: the
 * conversion-pair catches the division wherever BOTH operands convert, the mixed-operand shape catches a
 * runtime value divided against a bare literal that folds straight to an `f32` suffix with no `f32(...)`
 * wrapper on that side (red-proven: `lane / PAIRS_PER_BODY` where `PAIRS_PER_BODY` is a bare TS constant
 * emits `f32(lane) / 8f`, invisible to the conversion-pair alone), and the bare-slash sweep catches an
 * operand shape the first two would have to guess at. A real float division (a normalize, a reciprocal)
 * trips the third one, so a kernel that legitimately divides floats asserts the first two alone.
 */
export function noIntegerDivision(src: string): void {
    // the operands are parenthesized whenever they're compound — `u32((f32((a + b)) / f32((b + 1u))))`
    // — so the pattern between the two conversions has to be non-greedy rather than paren-free
    expect(flat(src)).not.toMatch(/f32\(.*?\) \/ f32\(/);
    // the mixed shape: one operand converted, the other a bare numeric literal with the `f` suffix
    expect(flat(src)).not.toMatch(/f32\(.*?\) \/ \d+f\b/);
    expect(flat(src)).not.toMatch(/\d+f \/ f32\(/);
}

/**
 * assert no bare `/` at all, integer or float — the stronger form of {@link noIntegerDivision} for a
 * kernel whose every index is a shift, mask or multiply-add. `idiv`'s own leaf body is the one legal
 * division, so pass `src` with it stripped.
 */
export function noDivision(src: string): void {
    noIntegerDivision(src);
    // the left-operand class covers a subscript too (`arr[i] / 2`), which a word/paren-only class misses
    expect(flat(src).replace(IDIV_LEAF, "")).not.toMatch(/[)\]\w] ?\/ ?/);
}

/** `idiv`'s emitted leaf — the one place a `/` on integers is correct. */
export const IDIV_LEAF = "fn idivWgsl(a: u32, b: u32) -> u32 { return a / b; }";

/**
 * assert every `&x` in the emitted WGSL points at a `var`, never a `let`. TGSL declares a local `let`
 * unless the body assigns it directly, so a local written ONLY through a pointer (a double-buffer the
 * callee fills, a `bestSep` accumulated by a helper) comes out `let` — and `&` on a `let` is a WGSL type
 * error the transpiler never sees, so only a real shader compile catches it. A `d.ref(x)` argument marks
 * the local for the transpiler, so this only bites where `d.ref` can't be used — it refuses a scalar.
 * Red-proven against `capsulePoly`'s `bestSep` / `n` (both bare, both needing a forcing self-assign).
 *
 * Scoped per emitted function body, not the module-flat text: a `let n` in one spliced function and a
 * `(&n)` in an unrelated one share no scope, so a flat match false-positives across them (`hullWgsl()`'s
 * `capsulePoly`'s `var n` vs `supportPoly`'s `let n = polyVertCount(p)`).
 */
export function pointerDiscipline(src: string): void {
    for (const fn of functionBodies(src)) {
        const flattened = flat(fn);
        for (const [, name] of flattened.matchAll(/\(&([A-Za-z_]\w*)\)/g)) {
            // a pointer to a function parameter is already a reference, so only locals are at risk
            if (new RegExp(`\\blet ${name} =`).test(flattened))
                throw new Error(`&${name} takes the address of a \`let\` — force a \`var\``);
        }
    }
}

/**
 * brace-matched top-level `fn name(...) { ... }` blocks, in emission order. Falls back to the whole
 * source when it names no function (a caller passing an already-scoped body with no `fn` header).
 */
function functionBodies(src: string): string[] {
    const bodies: string[] = [];
    for (const m of src.matchAll(/\bfn\s+[A-Za-z_]\w*\s*\(/g)) {
        const start = m.index;
        let depth = 0;
        for (let i = src.indexOf("{", start); i < src.length; i++) {
            if (src[i] === "{") depth++;
            else if (src[i] === "}" && --depth === 0) {
                bodies.push(src.slice(start, i + 1));
                break;
            }
        }
    }
    return bodies.length > 0 ? bodies : [src];
}

/**
 * assert a kernel's integer locals stay unsigned. A bare JS literal seeds an `i32`, which flips the
 * arithmetic signed and sprays conversions — and at bit 31 walks into signed overflow, which WGSL
 * leaves indeterminate. An array subscript is exempt: `arr[0i]` is legal and has no arithmetic
 * consequence.
 *
 * Three assertions, because the first two miss a shift. A bare literal shifted by a runtime `u32`
 * materializes as `i32` and emits **no** conversion and no `i` suffix for them to see — `~(3 << s)`
 * looks clean and is signed overflow at `s = 30` (red-proven: the BVH trail's mask, which needed
 * `d.u32(3)`). Its sibling `(1 << n) - 1` *does* emit `i32(...)`, which is why one shape's escape went
 * unnoticed. A suffixed literal (`3u << s`) has a non-space after the digits, so it never matches.
 */
export function integerDiscipline(src: string): void {
    expect(flat(src)).not.toMatch(/\bi32\(/);
    expect(flat(src).replace(/\[[^\]]*\]/g, "[]")).not.toMatch(/\b\d+i\b/);
    expect(flat(src)).not.toMatch(/[^\w.]\d+ ?<</);
}
