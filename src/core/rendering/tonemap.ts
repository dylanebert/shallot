import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/** Tonemapping operators. */
export const TonemappingMethod = {
    TonyMcMapface: 0,
    /** Bypass grading, exposure and dither; only display encoding applies. */
    None: 1,
    /** Identity curve with color grading, exposure and debanding dither. */
    Linear: 8,
    AcesFitted: 2,
    Reinhard: 3,
    ReinhardLuminance: 4,
    AgX: 5,
    SomewhatBoringDisplayTransform: 6,
    KhronosPbrNeutral: 7,
} as const;

/** Rec. 709 relative luminance. */
export const tmLuma = tgpu.fn(
    [d.vec3f],
    d.f32,
)((v) => {
    "use gpu";
    return std.dot(v, d.vec3f(0.2126, 0.7152, 0.0722));
});

/** Khronos PBR Neutral (three.js `NeutralToneMapping` / modelviewer.dev) — preserves authored color,
 *  desaturating only past the compression knee. The default operator. */
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
    const headroom = 1 - startCompression;
    const newPeak = 1 - (headroom * headroom) / (peak + headroom - startCompression);
    c = std.mul(c, newPeak / peak);
    const g = 1 - 1 / (desaturation * (peak - newPeak) + 1);
    return std.mix(c, d.vec3f(newPeak), g);
});

/** plain Reinhard — oversaturates, the simple baseline. */
export const tmReinhard = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    return std.div(c, std.add(d.vec3f(1), c));
});

/** luminance-space Reinhard — preserves hue (Bevy `tonemapping_reinhard_luminance`). */
export const tmReinhardLuminance = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const lOld = tmLuma(c);
    const lNew = lOld / (1 + lOld);
    return std.mul(c, lNew / std.max(lOld, 1e-5));
});

/** the Stephen Hill RRT+ODT rational fit {@link tmAces} applies between its two ACES matrices. */
export const tmRrtOdtFit = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((v) => {
    "use gpu";
    const a = std.sub(std.mul(v, std.add(v, 0.0245786)), 0.000090537);
    const b = std.add(std.mul(v, std.add(std.mul(0.983729, v), 0.432951)), 0.238081);
    return std.div(a, b);
});

/** ACES Fitted (Stephen Hill RRT+ODT fit) — contrasty, saturation-boosting. */
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

/** Bevy's applyAgXLog, preparing coordinates for its 32³ default-contrast LUT. */
export const tmAgxLog = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((input) => {
    "use gpu";
    const color = std.max(input, d.vec3f(0));
    let prepared = d.vec3f(
        std.dot(color, d.vec3f(0.84247906, 0.0784336, 0.07922375)),
        std.dot(color, d.vec3f(0.04232824, 0.87846864, 0.07916613)),
        std.dot(color, d.vec3f(0.04237565, 0.0784336, 0.87914297)),
    );
    prepared = std.select(
        prepared,
        std.add(prepared, 0.00001525878),
        std.lt(prepared, d.vec3f(0.00003051757)),
    );
    prepared = std.clamp(std.log2(std.div(prepared, 0.18)), d.vec3f(-10), d.vec3f(6.5));
    return std.clamp(std.div(std.add(prepared, 10), 16.5), d.vec3f(0), d.vec3f(1));
});

/** Rec. 709 RGB → YCbCr, the chroma basis {@link tmSomewhatBoring} desaturates in. */
export const tmRgbToYcbcr = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const m = d.mat3x3f(0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5, 0.5, -0.4542, -0.0458);
    return std.mul(c, m);
});

export const tmSbCurve = tgpu.fn(
    [d.f32],
    d.f32,
)((v) => {
    "use gpu";
    return 1 - std.exp(-v);
});

export const tmSbCurve3 = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((v) => {
    "use gpu";
    return d.vec3f(tmSbCurve(v.x), tmSbCurve(v.y), tmSbCurve(v.z));
});

/** SomewhatBoringDisplayTransform (Stachowiak): chroma-aware highlight desaturation. */
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

/** Analytic operators; TonyMcMapface and AgX are sampled by the presentation pass. */
export const tonemap = tgpu.fn(
    [d.u32, d.vec3f],
    d.vec3f,
)((mode, c) => {
    "use gpu";
    if (mode === 8) return std.copy(c);
    if (mode === 2) return tmAces(c);
    if (mode === 3) return tmReinhard(c);
    if (mode === 4) return tmReinhardLuminance(c);
    if (mode === 6) return tmSomewhatBoring(c);
    return tmNeutral(c);
});
