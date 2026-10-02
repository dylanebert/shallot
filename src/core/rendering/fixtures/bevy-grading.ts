// Extracted from Bevy 24661940b42fa8c8a89538a1d3572f7fe6dbd49a.
// MIT OR Apache-2.0, copyright Bevy contributors; see ../bevy-LICENSE-MIT.
// Only the lower crossover assignment is corrected, with its citation below.
export const bevyGrading = `const LEVEL_MARGIN: f32 = 0.1;
const LEVEL_MARGIN_DIV: f32 = 0.5 / LEVEL_MARGIN;
const FRAC_PI_3: f32 = 1.0471975511965976;
fn hsv_to_rgb(hsv: vec3<f32>) -> vec3<f32> {
    let n = vec3(5.0, 3.0, 1.0);
    let k = (n + hsv.x / FRAC_PI_3) % 6.0;
    return hsv.z - hsv.z * hsv.y * max(vec3(0.0), min(k, min(4.0 - k, vec3(1.0))));
}

// Converts RGB to HSV.
//
// Input: R ∈ [0, 1], G ∈ [0, 1], B ∈ [0, 1].
// Output: H ∈ [0, 2π), S ∈ [0, 1], V ∈ [0, 1].
//
// <https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB>
fn rgb_to_hsv(rgb: vec3<f32>) -> vec3<f32> {
    let x_max = max(rgb.r, max(rgb.g, rgb.b));  // i.e. V
    let x_min = min(rgb.r, min(rgb.g, rgb.b));
    let c = x_max - x_min;  // chroma

    var swizzle = vec3<f32>(0.0);
    if (x_max == rgb.r) {
        swizzle = vec3(rgb.gb, 0.0);
    } else if (x_max == rgb.g) {
        swizzle = vec3(rgb.br, 2.0);
    } else {
        swizzle = vec3(rgb.rg, 4.0);
    }

    let h = FRAC_PI_3 * (((swizzle.x - swizzle.y) / c + swizzle.z) % 6.0);

    // Avoid division by zero.
    var s = 0.0;
    if (x_max > 0.0) {
        s = c / x_max;
    }

    return vec3(h, s, x_max);
}

fn powsafe(color: vec3<f32>, power: f32) -> vec3<f32> {
    return pow(abs(color), vec3(power)) * sign(color);
}

fn tonemapping_luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sectional_color_grading(
    in: vec3<f32>,
    color_grading: ptr<function, ColorGrading>,
) -> vec3<f32> {
    var color = in;

    // Determine whether the color is a shadow, midtone, or highlight. Colors
    // close to the edges are considered a mix of both, to avoid sharp
    // discontinuities. The formulas are taken from Blender's compositor.

    let level = (color.r + color.g + color.b) / 3.0;

    // Determine whether this color is a shadow, midtone, or highlight. If close
    // to the cutoff points, blend between the two to avoid sharp color
    // discontinuities.
    var levels = vec3(0.0);
    let midtone_range = (*color_grading).midtone_range;
    if (level < midtone_range.x - LEVEL_MARGIN) {
        levels.x = 1.0;
    } else if (level < midtone_range.x + LEVEL_MARGIN) {
        levels.y = ((level - midtone_range.x) * LEVEL_MARGIN_DIV) + 0.5;
        // Correct Bevy's lower crossover: Blender v3.6.0,
        // source/blender/compositor/operations/COM_ColorCorrectionOperation.cc:67-69
        // sets level_shadows = 1.0f - level_midtones.
        levels.x = 1.0 - levels.y;
    } else if (level < midtone_range.y - LEVEL_MARGIN) {
        levels.y = 1.0;
    } else if (level < midtone_range.y + LEVEL_MARGIN) {
        levels.z = ((level - midtone_range.y) * LEVEL_MARGIN_DIV) + 0.5;
        levels.y = 1.0 - levels.z;
    } else {
        levels.z = 1.0;
    }

    // Calculate contrast/saturation/gamma/gain/lift.
    let contrast = dot(levels, (*color_grading).contrast);
    let saturation = dot(levels, (*color_grading).saturation);
    let gamma = dot(levels, (*color_grading).gamma);
    let gain = dot(levels, (*color_grading).gain);
    let lift = dot(levels, (*color_grading).lift);

    // Adjust saturation and contrast.
    let luma = tonemapping_luminance(color);
    color = luma + saturation * (color - luma);
    color = 0.5 + (color - 0.5) * contrast;

    // The [ASC CDL] formula for color correction. Given *i*, an input color, we
    // have:
    //
    //     out = (i × s + o)ⁿ
    //
    // Following the normal photographic naming convention, *gain* is the *s*
    // factor, *lift* is the *o* term, and the inverse of *gamma* is the *n*
    // exponent.
    //
    // [ASC CDL]: https://en.wikipedia.org/wiki/ASC_CDL#Combined_Function
    color = powsafe(color * gain + lift, 1.0 / gamma);

    // Account for exposure.
    color = color * powsafe(vec3(2.0), (*color_grading).exposure);
    return max(color, vec3(0.0));
}

`;
