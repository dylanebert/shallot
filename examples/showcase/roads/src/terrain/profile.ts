// The longitudinal road profile: a low-pass, grade-limited elevation control-point sequence sampled along
// each polyline's own centreline — the spec's Locked decision for stage 8 ("low-pass the longitudinal
// profile, don't linearize it"). `flatten.ts`'s `setNetwork` calls {@link buildPolylineProfile} once per
// polyline to build the GPU segment list's per-endpoint heights; `networkCore` (flatten.ts) then
// interpolates between them by the clamped projection parameter `t` it already computes, instead of
// re-sampling raw `heightAt` at the projected point (stage 6/7's defect: every noise octave at the 4 m
// vertex scale rode straight into the road's own surface).
//
// Pure CPU module (the device-free split `overlay/document.ts`/`overlay/tiles.ts` use) — `heightAtCpu`
// below is a from-scratch JS re-authoring of `noise.ts`'s TGSL `perlin2`/`fbm2`/`heightAt` (not a call
// into them: those are `tgpu.fn`s bound through `noiseLayout`'s storage binding, which has no meaning
// outside a dispatched kernel). A small CPU/GPU numerical drift (f64 vs f32, ULP-level) is expected and
// harmless here — the profile's own smoothing changes the height by decimetres to metres by design, which
// dwarfs it — but the *algorithm* has to match, or the falloff terminus (`flatten.ts`'s cosine ease, which
// blends back to `heightAt` computed by the real GPU kernel) would show a visible seam.

const SQRT1_2 = Math.SQRT1_2;

// Arc-length step: the vertex grid's own spacing (`grid.ts`'s SPACING, imported below) — profile detail
// finer than the mesh itself can resolve is invisible in the emitted heightfield, and the longitudinal
// oracle (`flatten.test.ts`) requires sampling at <= the vertex spacing, which this satisfies exactly by
// construction rather than by a separately chosen number.
import { SPACING } from "./grid";
import { GROUND_LEVEL, HFREQ, LACUNARITY, OCTAVES, PERSISTENCE, RELIEF } from "./noise";

export const PROFILE_STEP = SPACING; // 4 m

// Smoothing strength: a live control (`boot.ts`'s bracket-key idiom, mirroring the F9 seed control) —
// the number of samples averaged on each side of a profile point. Not mine to pick a final value for
// (the spec's taste handover); MIN/MAX bound the control's range, DEFAULT is the shipped starting point.
export const MIN_SMOOTH_RADIUS = 0; // no box averaging — grade-clamp (below) still always applies
export const MAX_SMOOTH_RADIUS = 8; // a 17-sample, ~68 m window at PROFILE_STEP
export const DEFAULT_SMOOTH_RADIUS = 3; // a 7-sample, ~28 m window

// Grade limit: AASHTO's "A Policy on Geometric Design of Highways and Streets" (the Green Book) puts 12%
// as the practical ceiling for a low-speed local road in rolling/mountainous terrain — the steepest a real
// road standard permits, not a value fitted to this terrain's own relief.
export const MAX_GRADE = 0.12; // rise/run

// Jitter bound: independent of MAX_GRADE, deliberately — a bound derived from the grade limit alone (via
// the triangle inequality on two grade-bounded steps) is satisfied automatically by any profile that
// already passes the grade check, since |Δ²h| <= |Δh_i+1| + |Δh_i| whenever both terms are individually
// grade-bounded; that would make the jitter check redundant, never discriminating anything the grade check
// didn't already catch. A profile that zigzags direction every sample while staying under MAX_GRADE at
// every single step is exactly the case a rider still feels as choppy, so the standard for *this* axis is
// the one road-design practice actually uses for it: whether a grade change needs its own vertical curve.
// AASHTO guidance treats an algebraic grade break under about 1% as unnoticeable at low design speeds —
// under that, no vertical curve is needed to ride smoothly; over it, the break itself reads as a bump.
export const MAX_GRADE_BREAK = 0.01; // dimensionless — the change in grade (not the grade itself) allowed
// between two adjacent steps, one step-pair to the next.

/** the seeded permutation table's hash, mirroring `noise.ts`'s `grad2` — a from-scratch switch instead of
 *  an if-chain (a different code shape over the same 8-direction gradient set, same independent-derivation
 *  spirit stage 5/6 use for distance). */
function grad2Cpu(hash: number, x: number, z: number): number {
    switch (hash & 7) {
        case 0:
            return x;
        case 1:
            return -x;
        case 2:
            return z;
        case 3:
            return -z;
        case 4:
            return SQRT1_2 * x + SQRT1_2 * z;
        case 5:
            return -SQRT1_2 * x + SQRT1_2 * z;
        case 6:
            return SQRT1_2 * x - SQRT1_2 * z;
        default:
            return -SQRT1_2 * x - SQRT1_2 * z;
    }
}

function fade(t: number): number {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

function mixCpu(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

/** CPU mirror of `noise.ts`'s `perlin2` — improved-noise 2D gradient lattice, same permutation table
 *  format (`makePermutation`'s doubled length-512 array). */
function perlin2Cpu(x: number, y: number, perm: Uint32Array): number {
    const fx = Math.floor(x);
    const fy = Math.floor(y);
    const X = fx & 255;
    const Y = fy & 255;
    const xf = x - fx;
    const yf = y - fy;
    const u = fade(xf);
    const v = fade(yf);
    const A = perm[X] + Y;
    const B = perm[X + 1] + Y;
    const v00 = grad2Cpu(perm[A], xf, yf);
    const v10 = grad2Cpu(perm[B], xf - 1, yf);
    const v01 = grad2Cpu(perm[A + 1], xf, yf - 1);
    const v11 = grad2Cpu(perm[B + 1], xf - 1, yf - 1);
    return mixCpu(mixCpu(v00, v10, u), mixCpu(v01, v11, u), v);
}

/** CPU mirror of `noise.ts`'s `fbm2` — same octave/persistence/lacunarity loop, over the same exported
 *  constants (one source, `noise.ts`), so a tuning change there doesn't need a second edit here. */
function fbm2Cpu(x: number, y: number, perm: Uint32Array): number {
    let amp = 1;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let i = 0; i < OCTAVES; i++) {
        sum += perlin2Cpu(x * freq, y * freq, perm) * amp;
        norm += amp;
        amp *= PERSISTENCE;
        freq *= LACUNARITY;
    }
    return sum / norm;
}

/** CPU mirror of `noise.ts`'s `heightAt` — the natural terrain height at world (x, z), for the profile
 *  builder to sample without a device. */
export function heightAtCpu(x: number, z: number, perm: Uint32Array): number {
    return GROUND_LEVEL + fbm2Cpu(x * HFREQ, z * HFREQ, perm) * RELIEF;
}

/** a box filter (unweighted moving average) over `heights`, radius samples each side, clamped at the
 *  sequence's own ends (no wrap, no zero-padding — both would pull the endpoint average toward a value
 *  the profile never actually has). `radius <= 0` is a no-op copy. */
export function boxFilter(heights: readonly number[], radius: number): number[] {
    if (radius <= 0) return heights.slice();
    const n = heights.length;
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        let count = 0;
        for (let k = -radius; k <= radius; k++) {
            const j = Math.min(n - 1, Math.max(0, i + k));
            sum += heights[j];
            count++;
        }
        out[i] = sum / count;
    }
    return out;
}

/**
 * grade-limit `heights` so no adjacent pair (spaced `steps[i]` metres apart, `steps.length === heights.length - 1`)
 * differs by more than `maxGrade * steps[i]` — a forward-then-backward envelope clamp (a standard
 * slope-limiting technique for a sampled profile: each pass alone is sufficient to satisfy the bound
 * against its own direction's neighbour, moving the *smaller* correction needed; running both keeps the
 * result closer to the input than either pass alone would, since each pass can only pull a point toward
 * its neighbour's own already-adjusted value). The backward pass is what fixes the final adjacent-pair
 * bound: by the time it sets `out[i]`, `out[i + 1]` is already final, so `|out[i] - out[i + 1]| <= maxDelta`
 * holds by construction for every `i`, which is exactly the grade the longitudinal oracle checks.
 * @example clampGrade([0, 10], [1], 1) // [0, 1] — a 10 m rise over 1 m (1000% grade) clipped to 100%
 */
export function clampGrade(
    heights: readonly number[],
    steps: readonly number[],
    maxGrade: number,
): number[] {
    const out = heights.slice();
    for (let i = 1; i < out.length; i++) {
        const maxDelta = maxGrade * steps[i - 1];
        out[i] = Math.min(Math.max(out[i], out[i - 1] - maxDelta), out[i - 1] + maxDelta);
    }
    for (let i = out.length - 2; i >= 0; i--) {
        const maxDelta = maxGrade * steps[i];
        out[i] = Math.min(Math.max(out[i], out[i + 1] - maxDelta), out[i + 1] + maxDelta);
    }
    return out;
}

/** one profile control point: world (x, z) plus its smoothed, grade-limited height. */
export interface ProfilePoint {
    readonly x: number;
    readonly z: number;
    readonly height: number;
}

/**
 * limit a height sequence on *both* difference axes: the grade (first difference, {@link MAX_GRADE}) and
 * the grade break (second difference, {@link MAX_GRADE_BREAK}) — not one clamp on heights, which only
 * reaches the first difference. A hard height clamp that repeatedly saturates against noisy input (climb
 * to the grade cap, then immediately back down) produces exactly the sharp alternating-direction ramps the
 * grade-break axis exists to catch (measured: `flatten.test.ts`'s own longitudinal-oracle sweep, an
 * envelope-clamped-only profile failed the grade-break check on effectively every road). Working in *grade
 * space* instead fixes it: convert heights to a grade sequence, clamp its own adjacent differences with
 * the same forward-backward envelope technique {@link clampGrade} uses (grade values now play the role
 * heights did, one grade-break apart per adjacent pair), clip to the absolute grade bound, then integrate
 * back to heights anchored at the sequence's first sample.
 */
function limitProfile(
    heights: readonly number[],
    steps: readonly number[],
    maxGrade: number,
): number[] {
    const grade: number[] = [];
    for (let i = 0; i < heights.length - 1; i++) {
        grade.push(steps[i] > 1e-9 ? (heights[i + 1] - heights[i]) / steps[i] : 0);
    }
    const clippedGrade = grade.map((g) => Math.max(-maxGrade, Math.min(maxGrade, g)));
    const breakSteps = new Array(Math.max(0, clippedGrade.length - 1)).fill(1);
    const smoothedGrade = clampGrade(clippedGrade, breakSteps, MAX_GRADE_BREAK).map((g) =>
        Math.max(-maxGrade, Math.min(maxGrade, g)),
    );

    const out = [heights[0]];
    for (let i = 0; i < smoothedGrade.length; i++) out.push(out[i] + smoothedGrade[i] * steps[i]);
    return out;
}

/**
 * resample `points` (a polyline's own consecutive vertices — 2 for every road this stage's generator
 * produces, `overlay/network.ts`) at <= {@link PROFILE_STEP} arc-length increments, sample natural height
 * at each (`heightAtCpu`), box-filter by `radius` ({@link boxFilter}), then limit on both difference axes
 * ({@link limitProfile}, always active regardless of `radius` — the correctness bound, not the taste
 * dial). Consecutive input segments share their joint vertex (no duplicate sample there), so the whole
 * polyline resamples as one continuous sequence, not per-segment independently — a multi-point future
 * polyline would stay smooth across its own interior joints instead of re-starting the filter at each one.
 */
export function buildPolylineProfile(
    points: ReadonlyArray<readonly [number, number]>,
    perm: Uint32Array,
    radius: number,
): ProfilePoint[] {
    const raw: { x: number; z: number }[] = [{ x: points[0][0], z: points[0][1] }];
    for (let i = 0; i < points.length - 1; i++) {
        const [ax, az] = points[i];
        const [bx, bz] = points[i + 1];
        const len = Math.hypot(bx - ax, bz - az);
        const n = Math.max(1, Math.ceil(len / PROFILE_STEP));
        for (let s = 1; s <= n; s++) {
            const t = s / n;
            raw.push({ x: ax + (bx - ax) * t, z: az + (bz - az) * t });
        }
    }

    const steps: number[] = [];
    for (let i = 0; i < raw.length - 1; i++) {
        steps.push(Math.hypot(raw[i + 1].x - raw[i].x, raw[i + 1].z - raw[i].z));
    }

    const natural = raw.map((p) => heightAtCpu(p.x, p.z, perm));
    const smoothed = boxFilter(natural, radius);
    const limited = limitProfile(smoothed, steps, MAX_GRADE);

    return raw.map((p, i) => ({ x: p.x, z: p.z, height: limited[i] }));
}

/** {@link longitudinalOracle}'s verdict: whether `profile` satisfies the spec's longitudinal Validation
 *  criterion on each axis, plus the worst observed value on each for diagnostics. */
export interface LongitudinalCheck {
    readonly gradeOk: boolean;
    readonly jitterOk: boolean;
    readonly maxGrade: number;
    readonly maxJitter: number;
}

/**
 * the longitudinal flattening oracle (spec's Validation, added by the release look, 2026-08-18): checks a
 * sampled centreline profile's first difference (grade) against {@link MAX_GRADE}, and its second
 * difference — the change *in* grade from one step-pair to the next — against {@link MAX_GRADE_BREAK}, a
 * bound independent of the grade limit itself (see the constant's own doc for why sharing one derivation
 * between the two axes would make the second check redundant). Neither bound is fitted to any one
 * network's own output. Generic over any `x, z, height` sequence, not just {@link buildPolylineProfile}'s
 * own output: this is what `flatten.test.ts`'s/`profile.test.ts`'s mutation proof runs against both the
 * smoothed profile (should pass) and a reconstruction of stage 6's raw, unsmoothed one (should fail).
 */
export function longitudinalOracle(profile: readonly ProfilePoint[]): LongitudinalCheck {
    const steps: number[] = [];
    for (let i = 1; i < profile.length; i++) {
        steps.push(Math.hypot(profile[i].x - profile[i - 1].x, profile[i].z - profile[i - 1].z));
    }

    // signed grade, not magnitude: a V-shaped kink (steep up, then equally steep down) has identical
    // |grade| on both sides, so an unsigned jitter check would read it as zero change — exactly the sharp
    // direction reversal this axis exists to catch. gradeOk still checks magnitude (a limit on steepness,
    // not direction); jitterOk needs the signed change to see a reversal at all.
    const signedGrades: number[] = [];
    let maxGrade = 0;
    let gradeOk = true;
    for (let i = 0; i < steps.length; i++) {
        const signed = steps[i] > 1e-9 ? (profile[i + 1].height - profile[i].height) / steps[i] : 0;
        signedGrades.push(signed);
        maxGrade = Math.max(maxGrade, Math.abs(signed));
        if (Math.abs(signed) > MAX_GRADE + 1e-9) gradeOk = false;
    }

    let maxJitter = 0;
    let jitterOk = true;
    for (let i = 1; i < signedGrades.length; i++) {
        const jitter = Math.abs(signedGrades[i] - signedGrades[i - 1]);
        maxJitter = Math.max(maxJitter, jitter);
        if (jitter > MAX_GRADE_BREAK + 1e-9) jitterOk = false;
    }

    return { gradeOk, jitterOk, maxGrade, maxJitter };
}
