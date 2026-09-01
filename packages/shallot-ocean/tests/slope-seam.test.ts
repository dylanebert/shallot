// CPU-only regression proof of `slope-seam.ts`'s own math — self-contained, no GPU adapter needed.
// Every "expected" value below is a LITERAL or a CLOSED FORM written directly in this file,
// derived from a position-encoded fixture (every source texel and channel a distinct value) —
// never a call into `slope.ts` or `slope-seam.ts` on the same inputs (I3g-r2's re-verdict: two
// prior rounds each built a "second mirror" by transcribing `expectedFromPublished` — a
// `reduceSlopeMip` copy with `Math.fround` discipline inserted — or compared `slopePostKernel`'s
// own output at level 0 against `expectedLevel0`'s rounding of itself (`x === f16Round(x)`), and
// both are one derivation under two names, not independence). The seam claim itself — that a
// published GPU texel is one of the two f16 neighbours of the value it was rounded FROM — has no
// CPU instance anywhere in this package: it lives entirely on the adapter, in the `ocean-slope`
// gym scenario (`bun bench --scenario ocean-slope`), which reads the actual published texture and
// buffers rather than a JS-simulated storage round trip. What THIS file proves is narrower and
// CPU-checkable: that `expectedLevel0` and `expectedFromPublished` — the two closed-form
// "reference" functions the device-side seam comparison calls — themselves compute the formula
// their docblocks claim, mutation-proven against literals nothing else in the package can move.
import { expect, test } from "bun:test";
import {
    expectedFromPublished,
    expectedLevel0,
    f16Neighbors,
    f16NextDown,
    f16NextUp,
    f16Round,
    f16StepDistance,
} from "../src/slope-seam";

test("f16 neighbour helpers round-trip and bracket correctly across binades", () => {
    expect(f16Round(0)).toBe(0);
    expect(f16Neighbors(0)).toEqual([0, 0]);
    // 1.0 sits at a binade boundary: the step above (in [1,2)) is 2^-10, the step below (in
    // [0.5,1)) is 2^-11 — the two are NOT equal, which is exactly why neighbours are derived from
    // the f16 bit pattern rather than from a single "step at this binade" formula.
    const [lo, hi] = f16Neighbors(1.0);
    expect(lo).toBe(1.0);
    expect(hi).toBe(1.0);
    const justAbove = 1.0 + 2 ** -12;
    const [loAbove, hiAbove] = f16Neighbors(justAbove);
    expect(loAbove).toBe(1.0);
    expect(hiAbove).toBeCloseTo(1.0 + 2 ** -10, 6);
    const justBelow = 1.0 - 2 ** -13;
    const [loBelow, hiBelow] = f16Neighbors(justBelow);
    expect(hiBelow).toBe(1.0);
    expect(loBelow).toBeCloseTo(1.0 - 2 ** -11, 6);
    expect(f16NextUp(f16NextDown(1.0))).toBe(1.0);
    expect(f16StepDistance(1.0, 1.0)).toBe(0);
    expect(f16StepDistance(1.0, hiAbove)).toBe(1);
});

/**
 * `expectedLevel0` against a hand-picked, position-encoded fixture: each of the 3 texels carries
 * distinct `x`/`z` values chosen to be EXACT in binary floating point (0.25, 0.0625, 4.0625,
 * 16.25, ... — every square and every sum below lands on an exact binary fraction), so the
 * expected array is a closed-form literal computed by hand from the formula the docblock states
 * (`energy = fround(fround(sx*sx) + fround(sz*sz))`, `residual = 0`) — never by calling
 * `expectedLevel0` a second time or by re-deriving it through any other package export.
 *
 * Mutation table (each mutation applied directly to `expectedLevel0` in `src/slope-seam.ts`, this
 * test re-run, then reverted to the pre-mutation text — the function has no other file to diff
 * against since it is itself the CPU reference, so there is no `git show HEAD:<path>` to revert
 * through; the exact pre-mutation source is preserved by hand instead):
 *
 *   1. swapped-channel (`out[i*4] = sz; out[i*4+1] = sx;`, output x/z channels swapped): exit 1.
 *      Texel 0 read `[3, 1, 10, 0]` against the fixture's expected `[1, 3, 10, 0]` — the first two
 *      channels mismatched at every texel, `toEqual` failed on the whole array.
 *   2. dropped z-term (`energy = fround(fround(sx * sx))`, `sz*sz` never added): exit 1. Texel 0's
 *      energy channel read `1` against the fixture's expected `10` (`sz=3` contributes `9`) —
 *      every texel's energy channel mismatched, `toEqual` failed on the whole array.
 */
test("expectedLevel0 computes the literal energy formula on a position-encoded fixture", () => {
    const xReal = new Float32Array([1, -2, 0.5]);
    const zReal = new Float32Array([3, 0.25, -4]);
    // biome-ignore format: one row per texel is the readable layout for a hand-derived table
    const expected = new Float32Array([
        1, 3, 10, 0, // texel 0: energy = 1*1 + 3*3 = 10
        -2, 0.25, 4.0625, 0, // texel 1: energy = 4 + 0.0625 = 4.0625
        0.5, -4, 16.25, 0, // texel 2: energy = 0.25 + 16 = 16.25
    ]);
    const actual = expectedLevel0(xReal, zReal);
    expect(Array.from(actual)).toEqual(Array.from(expected));
});

/**
 * `expectedFromPublished` against a position-encoded fixture: a 4x4 parent level where
 * `parent[i] = i + 1` for every source texel and channel (every one of the 64 values distinct),
 * reduced to a 2x2 child level. The expected array below is hand-derived closed form — each
 * child texel's mean is `(sum of its 4 parent texels' channel values) / 4`, residual is
 * `max(0, second - meanX^2 - meanZ^2)` — computed directly from the fixture's own index
 * arithmetic, never by calling `expectedFromPublished`, `reduceSlopeMip`, or any other package
 * export. Every value here is a small integer exactly representable in f32, so `Math.fround` is a
 * no-op throughout and the closed form is exact, not approximate.
 *
 * Mutation table (each mutation applied directly to `expectedFromPublished` in `src/slope-seam.ts`,
 * this test re-run, then reverted to the pre-mutation text by hand, the same reasoning as
 * `expectedLevel0`'s table above — the function under test IS the CPU reference, so there is no
 * separate production file to `git show HEAD:<path>` against):
 *
 *   1. weight (`* 0.25` -> `* 0.2` in `meanOf`): exit 1. Child (0,0)'s X channel read `8.8`
 *      (`44 * 0.2`) against the fixture's expected `11` — every channel of every child texel
 *      mismatched, `toEqual` failed on the whole array.
 *   2. dropped-texel (`meanOf`'s `Math.fround(c + e)` term dropped, only `c` kept — the fourth
 *      source texel's contribution never added): exit 1. Child (0,0)'s X channel read `5.75`
 *      (`(1+5+17)/4`, texel `e`'s value `21` never summed) against expected `11` — every channel
 *      of every child texel mismatched.
 *   3. wrong-offset (`offsets[0]`'s formula changed to `offsets[1]`'s — `(2*y*size+2*x+1)*4`
 *      instead of `(2*y*size+2*x)*4`, duplicating one source texel and dropping another): exit 1.
 *      Child (0,0)'s X channel read `12` (`(5+5+17+21)/4`, texel 1 counted twice, texel 0 never
 *      read) against expected `11` — every channel of every child texel mismatched.
 *   4. swapped-channel (`out[index] = meanZ; out[index+1] = meanX;`, output x/z channels swapped):
 *      exit 1. Child (0,0) read `[12, 11, 13, 0]` against the fixture's expected `[11, 12, 13, 0]`
 *      — the first two channels mismatched at every child texel.
 */
test("expectedFromPublished computes the literal mean/residual formula on a position-encoded fixture", () => {
    const parentSize = 4;
    const parent = new Float32Array(parentSize * parentSize * 4);
    for (let i = 0; i < parent.length; i++) parent[i] = i + 1;
    // biome-ignore format: one row per output texel is the readable layout for a hand-derived table
    const expected = new Float32Array([
        11, 12, 13, 0, // child (0,0): parent texels 0,1,4,5
        19, 20, 21, 0, // child (0,1): parent texels 2,3,6,7
        43, 44, 45, 0, // child (1,0): parent texels 8,9,12,13
        51, 52, 53, 0, // child (1,1): parent texels 10,11,14,15
    ]);
    const actual = expectedFromPublished(parent, parentSize);
    expect(Array.from(actual)).toEqual(Array.from(expected));
});
