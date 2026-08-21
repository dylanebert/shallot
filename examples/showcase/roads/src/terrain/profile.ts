// The longitudinal road profile: a straight chord per polyline — the spec's Locked decision half two
// (stage 17, the 2026-08-19 rescope). `flatten.ts`'s `setNetwork` calls {@link buildPolylineProfile} once
// per polyline to build the GPU segment list's per-endpoint heights; `networkCore` (flatten.ts) then
// interpolates between them by the clamped projection parameter `t` it already computes. On a straight
// segment the longitudinal station `s` is an affine function of `(x, z)`, so a linear profile in `s`
// makes the flatten target an affine function of `(x, z)` — barycentric interpolation over a triangle
// reproduces an affine field exactly, so the in-corridor reconstruction error vanishes identically rather
// than shrinking with cell size. The resample-and-limit chain this replaced (stage 8's "low-pass, don't
// linearize") made the target a curved function of station; that curvature was the entire remaining
// defect, and the whole apparatus was deleted at stage 19 — `buildPolylineProfile` no longer calls any
// of it.
//
// Pure CPU module (the device-free split `overlay/document.ts`/`overlay/tiles.ts` use) — `heightAtCpu`
// below is a from-scratch JS re-authoring of `noise.ts`'s TGSL `perlin2`/`fbm2`/`heightAt` (not a call
// into them: those are `tgpu.fn`s bound through `noiseLayout`'s storage binding, which has no meaning
// outside a dispatched kernel). A small CPU/GPU numerical drift (f64 vs f32, ULP-level) is expected and
// harmless here — the chord's own height changes by decimetres to metres by design, which dwarfs it — but
// the *algorithm* has to match, or the falloff terminus (`flatten.ts`'s cosine ease, which blends back to
// `heightAt` computed by the real GPU kernel) would show a visible seam.

const SQRT1_2 = Math.SQRT1_2;

// Arc-length step: the vertex grid's own spacing (`grid.ts`'s SPACING, imported below) — profile detail
// finer than the mesh itself can resolve is invisible in the emitted heightfield, and the longitudinal
// oracle (`flatten.test.ts`) requires sampling at <= the vertex spacing, which this satisfies exactly by
// construction rather than by a separately chosen number.
import { SPACING } from "./grid";
import { GROUND_LEVEL, HFREQ, LACUNARITY, OCTAVES, PERSISTENCE, RELIEF } from "./noise";

export const PROFILE_STEP = SPACING; // 4 m

// Grade limit: AASHTO's "A Policy on Geometric Design of Highways and Streets" (the Green Book) puts 12%
// as the practical ceiling for a low-speed local road in rolling/mountainous terrain — the steepest a real
// road standard permits, not a value fitted to this terrain's own relief.
//
// Every reader is an oracle or a fixture (`longitudinalOracle` below, `flatness.ts`'s `gradeBound`,
// `dragCorpus.ts`'s corpus filter, test assertions): **no production path enforces or clamps it**, so a
// drag can hand the flattener a steeper chord and the flattener will build it. This is the repo's second
// grade ceiling and they are 8× apart — `posts.ts`'s `MAX_CHORD_GRADE = 1.0` is the analytic worst case
// over every chord the drag *admits*, and its docblock carries the consequence of the gap.
export const MAX_GRADE = 0.12; // rise/run

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

/** one profile control point: world (x, z) plus its natural terrain height. */
export interface ProfilePoint {
    readonly x: number;
    readonly z: number;
    readonly height: number;
}

/**
 * the straight-chord profile (stage 17, the spec's Locked decision half two): one {@link ProfilePoint} per
 * polyline endpoint — `points[0]` and `points[points.length - 1]` — with `height` the natural terrain
 * height there (`heightAtCpu`). `buildNetworkGeometry` (`flatten.ts`) emits exactly one `ProfileSegment`
 * per polyline spanning these two points, so `networkCore`'s `mix(aHeight, bHeight, t)` is a linear
 * function of station — affine in `(x, z)` on a straight segment — and barycentric interpolation
 * reproduces it exactly at any cell size and any road angle. `cutDepth` is measured along the chord as
 * today (`buildNetworkGeometry`'s own loop over the returned profile points).
 */
export function buildPolylineProfile(
    points: ReadonlyArray<readonly [number, number]>,
    perm: Uint32Array,
): ProfilePoint[] {
    if (points.length > 2) {
        throw new Error(
            `buildPolylineProfile: polyline has ${points.length} points, but the one-chord premise ` +
                `(spec Locked decision: one straight segment per road) supports exactly 2 — interior ` +
                `points are not supported and would ship a non-affine target with every gate green`,
        );
    }
    const first = points[0];
    const last = points[points.length - 1];
    return [
        { x: first[0], z: first[1], height: heightAtCpu(first[0], first[1], perm) },
        { x: last[0], z: last[1], height: heightAtCpu(last[0], last[1], perm) },
    ];
}

/** {@link longitudinalOracle}'s verdict: whether `profile` satisfies the spec's longitudinal Validation
 *  criterion, plus the worst observed grade for diagnostics. */
export interface LongitudinalCheck {
    readonly gradeOk: boolean;
    readonly maxGrade: number;
}

/**
 * the longitudinal flattening oracle (spec's Validation, added by the release look, 2026-08-18): checks a
 * sampled centreline profile's first difference (grade) against {@link MAX_GRADE}. The bound is not fitted
 * to any one network's own output. Generic over any `x, z, height` sequence, not just
 * {@link buildPolylineProfile}'s own output: this is what `flatten.test.ts`'s/`profile.test.ts`'s mutation
 * proof runs against both the chord profile (should pass) and a reconstruction of stage 6's raw one
 * (should fail).
 */
export function longitudinalOracle(profile: readonly ProfilePoint[]): LongitudinalCheck {
    const steps: number[] = [];
    for (let i = 1; i < profile.length; i++) {
        steps.push(Math.hypot(profile[i].x - profile[i - 1].x, profile[i].z - profile[i - 1].z));
    }

    let maxGrade = 0;
    let gradeOk = true;
    for (let i = 0; i < steps.length; i++) {
        const signed = steps[i] > 1e-9 ? (profile[i + 1].height - profile[i].height) / steps[i] : 0;
        maxGrade = Math.max(maxGrade, Math.abs(signed));
        if (Math.abs(signed) > MAX_GRADE + 1e-9) gradeOk = false;
    }

    return { gradeOk, maxGrade };
}
