// Display-transform tonemap operators — the curve glaze applies to HDR scene radiance before the
// sRGB encode. Analytic only (no LUT): the Khronos PBR Neutral default plus the Bevy set (Reinhard,
// Reinhard-luminance, ACES Fitted, SomewhatBoring) and the iolite/Filament analytic AgX. Khronos
// Neutral + AgX are ported from three.js (NeutralToneMapping / AgXToneMapping — already column-major,
// so the matrices land in `d.mat3x3f` verbatim); Reinhard-luminance + SomewhatBoring from Bevy
// (`tonemapping_shared.wgsl`). The `Tonemap` indices are the one source of truth, paired with the
// dispatch in {@link tonemap} — index 0 (Neutral) is the zero-config default, so a camera with no
// `Glaze` (a zeroed config → mode 0) tonemaps Neutral.
//
// Every operator is a TGSL function: one source that runs on the CPU (its unit tests call it directly)
// and resolves into the WGSL a composite splices ({@link tonemapWgsl}). The operand order is the shipped
// shader's, term for term — reassociating a float product changes the last bits, and the bench's
// framebuffer probes compare against the pre-port baseline.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils/core";

/**
 * tonemap operator for the {@link Glaze} `tonemap` field. `Neutral` (0) is the zero-config default:
 * color-faithful, rolls off only the highlights. `None` is a raw passthrough.
 *
 * @example
 * ```
 * Glaze.tonemap.set(camera, Tonemap.Aces);
 * ```
 */
export const Tonemap = {
    Neutral: 0,
    None: 1,
    Aces: 2,
    Reinhard: 3,
    ReinhardLuminance: 4,
    Agx: 5,
    SomewhatBoring: 6,
} as const;

/** Rec. 709 relative luminance.
 *  @example let y = tmLuma(color); */
export const tmLuma = tgpu.fn(
    [d.vec3f],
    d.f32,
)((v) => {
    "use gpu";
    return std.dot(v, d.vec3f(0.2126, 0.7152, 0.0722));
});

/** Khronos PBR Neutral (three.js `NeutralToneMapping` / modelviewer.dev) — preserves authored color,
 *  desaturating only past the compression knee. The default operator.
 *  @example let display = tmNeutral(radiance); */
export const tmNeutral = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((cin) => {
    "use gpu";
    const startCompression = 0.8 - 0.04;
    const desaturation = 0.15;
    const x = std.min(cin.x, std.min(cin.y, cin.z));
    const offset = std.select(0.04, x - 6.25 * x * x, x < 0.08);
    let c = std.sub(cin, offset);
    const peak = std.max(c.x, std.max(c.y, c.z));
    if (peak < startCompression) return c;
    // three.js calls this `d`; that name is the typegpu data namespace in this file
    const headroom = 1 - startCompression;
    const newPeak = 1 - (headroom * headroom) / (peak + headroom - startCompression);
    c = std.mul(c, newPeak / peak);
    const g = 1 - 1 / (desaturation * (peak - newPeak) + 1);
    return std.mix(c, d.vec3f(newPeak), g);
});

/** plain Reinhard — oversaturates, the simple baseline.
 *  @example let display = tmReinhard(radiance); */
export const tmReinhard = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    return std.div(c, std.add(d.vec3f(1), c));
});

/** luminance-space Reinhard — preserves hue (Bevy `tonemapping_reinhard_luminance`).
 *  @example let display = tmReinhardLuminance(radiance); */
export const tmReinhardLuminance = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const lOld = tmLuma(c);
    const lNew = lOld / (1 + lOld);
    return std.mul(c, lNew / std.max(lOld, 1e-5));
});

/** the Stephen Hill RRT+ODT rational fit {@link tmAces} applies between its two ACES matrices.
 *  @example let fit = tmRrtOdtFit(acesInput); */
export const tmRrtOdtFit = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((v) => {
    "use gpu";
    const a = std.sub(std.mul(v, std.add(v, 0.0245786)), 0.000090537);
    const b = std.add(std.mul(v, std.add(std.mul(0.983729, v), 0.432951)), 0.238081);
    return std.div(a, b);
});

/** ACES Fitted (Stephen Hill RRT+ODT fit, via three.js / Godot) — contrasty, saturation-boosting.
 *  @example let display = tmAces(radiance); */
export const tmAces = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((cin) => {
    "use gpu";
    // named `acesIn` / `acesOut`, not `input` / `output`: a composite splicing this chunk will have
    // declared bindings by those names, and a shadowing local resolves as `input_1` there
    const acesIn = d.mat3x3f(
        d.vec3f(0.59719, 0.076, 0.0284),
        d.vec3f(0.35458, 0.90834, 0.13383),
        d.vec3f(0.04823, 0.01566, 0.83777),
    );
    const acesOut = d.mat3x3f(
        d.vec3f(1.60475, -0.10208, -0.00327),
        d.vec3f(-0.53108, 1.10813, -0.07276),
        d.vec3f(-0.07367, -0.00605, 1.07602),
    );
    const c = tmRrtOdtFit(std.mul(acesIn, cin));
    // `clamp(x, 0, 1)` is what WGSL defines `saturate` as; spelled out because typegpu has no CPU arm
    // for the vector `saturate`, and these operators are unit-tested on the CPU
    return std.clamp(std.mul(acesOut, c), d.vec3f(0), d.vec3f(1));
});

/** the AgX sigmoid, a degree-6 polynomial on the log-encoded value.
 *  @example let contrasted = tmAgxContrast(logEncoded); */
export const tmAgxContrast = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((x) => {
    "use gpu";
    const x2 = std.mul(x, x);
    const x4 = std.mul(x2, x2);
    return std.sub(
        std.add(
            std.add(
                std.sub(
                    std.add(
                        std.sub(std.mul(std.mul(15.5, x4), x2), std.mul(std.mul(40.14, x4), x)),
                        std.mul(31.96, x4),
                    ),
                    std.mul(std.mul(6.868, x2), x),
                ),
                std.mul(0.4298, x2),
            ),
            std.mul(0.1191, x),
        ),
        0.00232,
    );
});

/** iolite/Filament analytic AgX (three.js `AgXToneMapping`) — soft, neutral, gentle desaturation.
 *  @example let display = tmAgx(radiance); */
export const tmAgx = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((cin) => {
    "use gpu";
    const srgbToRec2020 = d.mat3x3f(
        d.vec3f(0.6274, 0.0691, 0.0164),
        d.vec3f(0.3293, 0.9195, 0.088),
        d.vec3f(0.0433, 0.0113, 0.8956),
    );
    const rec2020ToSrgb = d.mat3x3f(
        d.vec3f(1.6605, -0.1246, -0.0182),
        d.vec3f(-0.5876, 1.1329, -0.1006),
        d.vec3f(-0.0728, -0.0083, 1.1187),
    );
    const inset = d.mat3x3f(
        d.vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
        d.vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        d.vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    const outset = d.mat3x3f(
        d.vec3f(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        d.vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        d.vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    // `d.f32` types these f32 rather than WGSL abstract-float, so `maxEv - minEv` folds in f32 exactly as
    // the shipped shader's f32 `let`s did. Folded at f64 instead it lands one f32 step low (16.499998
    // against 16.5), which shifts every AgX log-range remap
    const minEv = d.f32(-12.47393);
    const maxEv = d.f32(4.026069);
    let c = std.mul(inset, std.mul(srgbToRec2020, cin));
    c = std.log2(std.max(c, d.vec3f(1e-10)));
    c = std.clamp(std.div(std.sub(c, minEv), maxEv - minEv), d.vec3f(0), d.vec3f(1));
    c = tmAgxContrast(c);
    c = std.pow(std.max(std.mul(outset, c), d.vec3f(0)), d.vec3f(2.2));
    return std.clamp(std.mul(rec2020ToSrgb, c), d.vec3f(0), d.vec3f(1));
});

/** Rec. 709 RGB → YCbCr, the chroma basis {@link tmSomewhatBoring} desaturates in.
 *  @example let ycbcr = tmRgbToYcbcr(color); */
export const tmRgbToYcbcr = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const m = d.mat3x3f(0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5, 0.5, -0.4542, -0.0458);
    return std.mul(c, m);
});

/** the `1 - exp(-v)` roll-off {@link tmSomewhatBoring} curves through.
 *  @example let y = tmSbCurve(luma); */
export const tmSbCurve = tgpu.fn(
    [d.f32],
    d.f32,
)((v) => {
    "use gpu";
    return 1 - std.exp(-v);
});

/** {@link tmSbCurve} per channel.
 *  @example let rolled = tmSbCurve3(color); */
export const tmSbCurve3 = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((v) => {
    "use gpu";
    return d.vec3f(tmSbCurve(v.x), tmSbCurve(v.y), tmSbCurve(v.z));
});

/** SomewhatBoringDisplayTransform (Stachowiak, via Bevy) — chroma-aware highlight desaturation.
 *  @example let display = tmSomewhatBoring(radiance); */
export const tmSomewhatBoring = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((cin) => {
    "use gpu";
    const ycbcr = tmRgbToYcbcr(cin);
    const bt = tmSbCurve(std.length(ycbcr.yz) * 2.4);
    let desat = std.max((bt - 0.7) * 0.8, 0);
    desat = desat * desat;
    const desatCol = std.mix(cin, d.vec3f(ycbcr.x), desat);
    const tm0 = std.mul(cin, std.max(0, tmSbCurve(ycbcr.x) / std.max(1e-5, tmLuma(cin))));
    const tm1 = tmSbCurve3(desatCol);
    return std.mul(std.mix(tm0, tm1, d.vec3f(bt * bt)), 0.97);
});

/**
 * apply the {@link Tonemap} operator `mode` selects. Index 0 — and any unknown index — is Neutral, the
 * zero-config default, so a camera with no `Glaze` (a zeroed config) still gets a display transform.
 * An if-chain rather than a `switch`: TGSL has no `switch` spelling, and the dispatch is over a
 * uniform-valued mode, so the branches fold.
 *
 * @example let display = tonemap(glaze.tonemapMode, radiance);
 */
export const tonemap = tgpu.fn(
    [d.u32, d.vec3f],
    d.vec3f,
)((mode, c) => {
    "use gpu";
    // `std.copy` because typegpu refuses to return a reference to a parameter
    if (mode === 1) return std.copy(c);
    if (mode === 2) return tmAces(c);
    if (mode === 3) return tmReinhard(c);
    if (mode === 4) return tmReinhardLuminance(c);
    if (mode === 5) return tmAgx(c);
    if (mode === 6) return tmSomewhatBoring(c);
    return tmNeutral(c);
});

/** the tonemap operators as one WGSL chunk: `tonemap(mode, color)` dispatching over the {@link Tonemap}
 *  indices, spliced into glaze's composite. A custom composite that wants the same display transforms
 *  splices this and dispatches on its own operator field. */
export const tonemapWgsl = chunk("tonemapWgsl", [tonemap], spliceNs);
