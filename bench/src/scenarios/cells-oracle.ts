// Finite certificate for the 10×6 fill, not a general floating-point oracle.
// WGSL CRD 2026-08-31 §§15.7.2–15.7.5, 17.9.2: correctly rounded need not mean
// nearest-even; division permits 2.5 ULP; packing is floor(.5 + 255*clamp(a)).
// Reordering preserves the signed terms, existing factors and nonlinear composition.
// It does not insert arbitrary cancelling terms or new transcendental calls.
// With rho=(5/2)*2^-23, a monomial traversing n rounding/inverse factors differs
// by at most G(n)=n*rho/(1-n*rho). Sum ABSOLUTE monomial magnitudes to cover
// cancellation, distribution, common-denominator and reciprocal-multiply forms.
// Coordinate/complement paths need <=4 factors and absolute sum <2. Alpha
// through packing has terms .5 + 255 - 255*i/255 - 255*.5/255: sum <=265 and
// <=8 factors/path. Post-power RGB has .5 + 255*A*P - 255*B: sum <284 and <=8
// factors/path including coefficient conversions. Fusion cannot worsen accuracy.
// These are path bounds, not instruction counts or a sequential rounding model.
// The subnormal reserves below dominate the finite paths' residues and scales.
type Q = { n: bigint; d: bigint };
const q = (n: number | bigint, d: number | bigint = 1): Q => ({ n: BigInt(n), d: BigInt(d) });
const add = (a: Q, b: Q): Q => q(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a: Q, b: Q): Q => q(a.n * b.d - b.n * a.d, a.d * b.d);
const mul = (a: Q, b: Q): Q => q(a.n * b.n, a.d * b.d);
const div = (a: Q, b: Q): Q => q(a.n * b.d, a.d * b.n);
const cmp = (a: Q, b: Q): number => (a.n * b.d < b.n * a.d ? -1 : a.n * b.d > b.n * a.d ? 1 : 0);
const abs = (a: Q): Q => q(a.n < 0n ? -a.n : a.n, a.d);
const floor = (a: Q): bigint => (a.n >= 0n ? a.n / a.d : -((-a.n + a.d - 1n) / a.d));
const two = (e: number): Q => (e >= 0 ? q(1n << BigInt(e)) : q(1, 1n << BigInt(-e)));
function requireProof(ok: boolean, label: string): void {
    if (!ok) throw new Error(`cells certificate: ${label}`);
}

function f32(bits: number): Q {
    const exponent = ((bits >>> 23) & 255) - 127;
    const fraction = BigInt(bits & 0x7fffff);
    requireProof(bits >>> 31 === 0 && exponent !== 128, "positive finite f32");
    return exponent === -127
        ? mul(q(fraction), two(-149))
        : mul(q((1n << 23n) + fraction), two(exponent - 23));
}

// Exact bit-space search; no host floating-point rounding assumption.
function bracket(a: Q): [number, number] {
    requireProof(cmp(a, q(0)) >= 0 && cmp(a, q(256)) <= 0, "bracket domain");
    let lo = 0;
    let hi = 0x43800000; // 256, including pack's scale/add intermediate
    while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (cmp(f32(mid), a) <= 0) lo = mid;
        else hi = mid;
    }
    if (cmp(f32(hi), a) === 0) return [hi, hi];
    return [lo, cmp(f32(lo), a) === 0 ? lo : hi];
}
const directed = (a: Q, up: boolean): Q => f32(bracket(a)[up ? 1 : 0]);

function alphaWitness(i: number): [number, number] {
    const exact = q(2 * i + 1, 510);
    const [lo, hi] = bracket(exact);
    const allowance = mul(q(5, 2), sub(f32(lo + 1), f32(lo)));
    requireProof(cmp(abs(sub(f32(lo), exact)), allowance) <= 0, "lower quotient admissible");
    requireProof(cmp(abs(sub(f32(hi), exact)), allowance) <= 0, "upper quotient admissible");
    const pack = (quotient: number, up: boolean): number => {
        const a = directed(sub(q(1), f32(quotient)), up);
        const scaled = directed(mul(q(255), a), up);
        return Number(floor(directed(add(q(1, 2), scaled), up)));
    };
    const bytes: [number, number] = [pack(hi, false), pack(lo, true)];
    requireProof(bytes[0] === 254 - i && bytes[1] === 255 - i, "both alpha endpoints witnessed");
    return bytes;
}

// Enclose c^(5/12) using integer comparisons r^12 <= c^5, not libm.
function root(c: Q): [Q, Q] {
    if (c.n === 0n) return [q(0), q(0)];
    if (c.n === c.d) return [q(1), q(1)];
    const scale = 1n << 64n;
    const right = c.n ** 5n * scale ** 12n;
    const d5 = c.d ** 5n;
    let lo = 0n;
    let hi = scale;
    while (hi - lo > 1n) {
        const mid = (hi + lo) / 2n;
        if (mid ** 12n * d5 <= right) lo = mid;
        else hi = mid;
    }
    requireProof(lo ** 12n * d5 <= right && hi ** 12n * d5 >= right, "root enclosure");
    return [q(lo, scale), q(hi, scale)];
}

/** Rebuild the rational certificate; dimensions outside its finite population refuse. */
export function certifyFill(cols: number, rows: number) {
    requireProof(cols === 10 && rows === 6, "only the 10×6 expression population is certified");
    const u = two(-23);
    const rho = mul(q(5, 2), u);
    const gamma = (n: number): Q => div(mul(q(n), rho), sub(q(1), mul(q(n), rho)));
    const alphaError = add(mul(q(265), gamma(8)), two(-100));
    const inputError = add(mul(q(2), gamma(4)), two(-120));
    // c>=.1-Ec: log2 derivative <15 (ln2>.69), |log2(c)|<4.
    // log2 accuracy <=3*2^-21 also covers crossing .5. The exponent's either
    // f32 conversion differs from 5/12 by <2^-25 and stays below .417.
    const logError = add(mul(q(15), inputError), mul(q(3), two(-21)));
    const tError = add(add(mul(q(417, 1000), logError), mul(q(4), two(-25))), u);
    // exp2 argument in (-2,2^-15), derivative <1, output <2: <=7 ULP.
    const powerError = add(tError, mul(q(7), u));
    const rgbError = add(add(mul(q(255 * 211, 200), powerError), mul(q(284), gamma(8))), two(-100));
    requireProof(cmp(alphaError, q(1)) < 0, "alpha floor enclosure width <1");
    requireProof(cmp(inputError, q(1, 400000)) < 0, "input branch/domain preserved");
    requireProof(cmp(tError, two(-15)) < 0, "exp2 derivative domain preserved");
    requireProof(cmp(rgbError, q(6, 1000)) < 0, "RGB error <.006 byte units");
    const alpha = Array.from({ length: 10 }, (_, i) => {
        // Ideal pre-floor is the integer 255-i; Ea<1 admits only n-1,n.
        const n = q(255 - i);
        const bytes = [Number(floor(sub(n, alphaError))), Number(floor(add(n, alphaError)))];
        const witnesses = alphaWitness(i);
        requireProof(
            bytes.every((b, k) => b === witnesses[k]),
            "alpha enclosure equals witnessed set",
        );
        return witnesses;
    });
    const rgb = [];
    for (const denominator of [10, 6]) {
        for (let index = 0; index < denominator; index++) {
            for (const complement of [false, true]) {
                const c = q(complement ? denominator - index : index, denominator);
                // Zero selects the linear branch and packs zero; do not evaluate log2(0).
                const scaled =
                    c.n === 0n
                        ? [q(0), q(0)]
                        : root(c).map((p) => mul(q(255), sub(mul(q(211, 200), p), q(11, 200))));
                const byte = floor(add(scaled[0], q(1, 2)));
                for (const margin of [
                    sub(scaled[0], sub(q(byte), q(1, 2))),
                    sub(add(q(byte), q(1, 2)), scaled[1]),
                ]) {
                    requireProof(cmp(margin, q(9, 1000)) > 0, "ideal RGB boundary distance >.009");
                    requireProof(cmp(margin, rgbError) > 0, "RGB singleton after error expansion");
                }
                // Clamp is non-expansive, including c=0 and c=1.
                rgb.push({ denominator, index, complement, byte: Number(byte) });
            }
        }
    }
    requireProof(rgb.length === 32 && alpha.length === 10, "certificate population");
    return { alpha, rgb, population: { cells: 60, alpha: 120, rgb: 360 } };
}

/** Per-sample set membership, never a distance from an observed/CPU byte. */
export function admitsAlpha(set: readonly [number, number], byte: number): boolean {
    return byte === set[0] || byte === set[1];
}

// Uploaded finite dyadics. Different foreground/background assignments expose routing.
// All alpha products/additions are exactly representable, including the .5 boundary.
// RGB endpoints have singleton results by certifyFill's endpoint certificate.
export const packingInputs = [
    [1 / 4, 3 / 4],
    [0, 1],
    [1 / 2 - 1 / 1024, 1 / 2 + 1 / 1024],
    [1 / 2, 1 / 4],
] as const;

/** Independent word construction for the exact GPU packing fixture. */
export function packingWords(): Uint32Array {
    const words = new Uint32Array(packingInputs.length * 3);
    for (let i = 0; i < packingInputs.length; i++) {
        const bytes = packingInputs[i].map((a) => {
            const input = q(a * 1024, 1024);
            const scaled = mul(q(255), input);
            const shifted = add(q(1, 2), scaled);
            requireProof(cmp(directed(scaled, false), scaled) === 0, "exact control multiply");
            requireProof(cmp(directed(shifted, false), shifted) === 0, "exact control add");
            return Number(floor(shifted));
        });
        words.set(
            [17, (bytes[0] * 2 ** 24 + 0x00ff00) >>> 0, (bytes[1] * 2 ** 24 + 0xff00ff) >>> 0],
            i * 3,
        );
    }
    return words;
}
