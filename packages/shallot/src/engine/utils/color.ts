/** sRGB → linear for a single 0..1 channel */
export function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

/** linear → sRGB for a single 0..1 channel */
export function linearToSrgb(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

/** unpack a 0xRRGGBB sRGB byte triple to linear-space r/g/b */
export function unpackColor(packed: number): { r: number; g: number; b: number } {
    return {
        r: srgbToLinear(((packed >> 16) & 0xff) / 255),
        g: srgbToLinear(((packed >> 8) & 0xff) / 255),
        b: srgbToLinear((packed & 0xff) / 255),
    };
}

/**
 * pack a 0xRRGGBB sRGB color + a 0..1 opacity into an RGBA8 `u32`, byte 0 = r,
 * byte 3 = a, the `unpack4x8unorm` layout a shader reads. sRGB bytes are kept
 * verbatim (linearize on unpack); opacity clamps to [0, 1] and rounds to a byte
 *
 * @example
 * const word = packColor(0xff8040, 0.5); // 0x80_40_80_ff, half alpha
 */
export function packColor(hex: number, opacity: number): number {
    const r = (hex >> 16) & 0xff;
    const g = (hex >> 8) & 0xff;
    const b = hex & 0xff;
    const a = Math.max(0, Math.min(255, Math.round(opacity * 255)));
    return (r | (g << 8) | (b << 16) | (a << 24)) >>> 0;
}

/**
 * WGSL `linearToOklab(c: vec3<f32>) -> vec3<f32>`: linear sRGB to OkLab
 * (Björn Ottosson's matrices). Splice into a surface preamble for perceptual
 * color work (hue/lightness perturbation around a base color); pair with
 * {@link OKLAB_TO_LINEAR_WGSL} to come back. One source so every shader
 * agrees on the matrices.
 */
export const LINEAR_TO_OKLAB_WGSL = /* wgsl */ `
fn linearToOklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = pow(max(l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}
`;

/**
 * WGSL `oklabToLinear(lab: vec3<f32>) -> vec3<f32>`: OkLab back to linear
 * sRGB (out-of-gamut values are NOT clamped; clamp at the call site if the
 * input can leave gamut). Counterpart of {@link LINEAR_TO_OKLAB_WGSL}; each
 * direction is its own constant so a shader splices only what it calls.
 */
export const OKLAB_TO_LINEAR_WGSL = /* wgsl */ `
fn oklabToLinear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}
`;

/**
 * format an integer as a 6-digit hex string (e.g. `0xff8080`). The `kind`
 * property is a reflection hint; `reflection.isColor` reads it to classify
 * trait `format` entries as color fields.
 */
export const formatHex = Object.assign(
    (n: number) => "0x" + (n >>> 0).toString(16).padStart(6, "0"),
    { kind: "color" as const },
);
