// Display-transform tonemap operators — the curve glaze applies to HDR scene radiance before the
// sRGB encode. Analytic only (no LUT): the Khronos PBR Neutral default plus the Bevy set (Reinhard,
// Reinhard-luminance, ACES Fitted, SomewhatBoring) and the iolite/Filament analytic AgX. Khronos
// Neutral + AgX are ported from three.js (NeutralToneMapping / AgXToneMapping — already column-major,
// so the matrices land in WGSL `mat3x3` verbatim); Reinhard-luminance + SomewhatBoring from Bevy
// (`tonemapping_shared.wgsl`). The `Tonemap` indices are the one source of truth, paired with the
// WGSL `switch` below — index 0 (Neutral) is the zero-config default, so a camera with no `Glaze`
// (a zeroed config → mode 0) tonemaps Neutral.

/**
 * tonemap operator for the {@link Glaze} `tonemap` field. `Neutral` (0) is the zero-config default —
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

/** the tonemap operators as one WGSL chunk — a `tonemap(color, op)` switch over the {@link Tonemap}
 * indices, spliced into glaze's composite. A custom composite that wants the same display transforms
 * splices this and dispatches on its own operator field. */
export const TONEMAP_WGSL = /* wgsl */ `
fn tmLuma(v: vec3<f32>) -> f32 { return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722)); }

// Khronos PBR Neutral (three.js NeutralToneMapping / modelviewer.dev) — preserves authored color,
// desaturating only past the compression knee. The default.
fn tmNeutral(cin: vec3<f32>) -> vec3<f32> {
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;
    var c = cin;
    let x = min(c.r, min(c.g, c.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    c = c - offset;
    let peak = max(c.r, max(c.g, c.b));
    if (peak < startCompression) { return c; }
    let d = 1.0 - startCompression;
    let newPeak = 1.0 - d * d / (peak + d - startCompression);
    c = c * (newPeak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(c, vec3<f32>(newPeak), g);
}

// plain Reinhard — oversaturates, the simple baseline
fn tmReinhard(c: vec3<f32>) -> vec3<f32> { return c / (vec3<f32>(1.0) + c); }

// luminance-space Reinhard — preserves hue (Bevy tonemapping_reinhard_luminance)
fn tmReinhardLuminance(c: vec3<f32>) -> vec3<f32> {
    let lOld = tmLuma(c);
    let lNew = lOld / (1.0 + lOld);
    return c * (lNew / max(lOld, 1e-5));
}

// ACES Fitted (Stephen Hill RRT+ODT fit, via three.js / Godot) — contrasty, saturation-boosting
fn tmRrtOdtFit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
fn tmAces(cin: vec3<f32>) -> vec3<f32> {
    let input = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let output = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    var c = tmRrtOdtFit(input * cin);
    return saturate(output * c);
}

// iolite/Filament analytic AgX (three.js AgXToneMapping) — soft, neutral, gentle desaturation
fn tmAgxContrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn tmAgx(cin: vec3<f32>) -> vec3<f32> {
    let srgbToRec2020 = mat3x3<f32>(
        vec3<f32>(0.6274, 0.0691, 0.0164),
        vec3<f32>(0.3293, 0.9195, 0.0880),
        vec3<f32>(0.0433, 0.0113, 0.8956),
    );
    let rec2020ToSrgb = mat3x3<f32>(
        vec3<f32>(1.6605, -0.1246, -0.0182),
        vec3<f32>(-0.5876, 1.1329, -0.1006),
        vec3<f32>(-0.0728, -0.0083, 1.1187),
    );
    let inset = mat3x3<f32>(
        vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let minEv = -12.47393;
    let maxEv = 4.026069;
    var c = inset * (srgbToRec2020 * cin);
    c = log2(max(c, vec3<f32>(1e-10)));
    c = saturate((c - minEv) / (maxEv - minEv));
    c = tmAgxContrast(c);
    c = pow(max(outset * c, vec3<f32>(0.0)), vec3<f32>(2.2));
    return saturate(rec2020ToSrgb * c);
}

// SomewhatBoringDisplayTransform (Stachowiak, via Bevy) — chroma-aware highlight desaturation
fn tmRgbToYcbcr(c: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        0.2126, 0.7152, 0.0722,
        -0.1146, -0.3854, 0.5,
        0.5, -0.4542, -0.0458,
    );
    return c * m;
}
fn tmSbCurve(v: f32) -> f32 { return 1.0 - exp(-v); }
fn tmSbCurve3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(tmSbCurve(v.x), tmSbCurve(v.y), tmSbCurve(v.z));
}
fn tmSomewhatBoring(cin: vec3<f32>) -> vec3<f32> {
    let ycbcr = tmRgbToYcbcr(cin);
    let bt = tmSbCurve(length(ycbcr.yz) * 2.4);
    var desat = max((bt - 0.7) * 0.8, 0.0);
    desat = desat * desat;
    let desatCol = mix(cin, vec3<f32>(ycbcr.x), desat);
    let tm0 = cin * max(0.0, tmSbCurve(ycbcr.x) / max(1e-5, tmLuma(cin)));
    let tm1 = tmSbCurve3(desatCol);
    return mix(tm0, tm1, vec3<f32>(bt * bt)) * 0.97;
}

// index 0 (and any unknown) = Neutral, the zero-config default
fn tonemap(mode: u32, c: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 1u: { return c; }
        case 2u: { return tmAces(c); }
        case 3u: { return tmReinhard(c); }
        case 4u: { return tmReinhardLuminance(c); }
        case 5u: { return tmAgx(c); }
        case 6u: { return tmSomewhatBoring(c); }
        default: { return tmNeutral(c); }
    }
}`;
