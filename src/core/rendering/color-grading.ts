// Bevy 24661940b42fa8c8a89538a1d3572f7fe6dbd49a, bevy_render/view and tonemapping_shared.wgsl.
// MIT OR Apache-2.0, copyright Bevy contributors; see bevy-LICENSE-MIT.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { component, f32, vec2, vec4 } from "../../engine";
import { tmLuma } from "./tonemap";

/** Camera grading in Bevy's model. Exposure is an offset in stops; hue is in degrees.
 * Temperature and tint adjust the D65 CIE xy white point. Section vectors contain
 * shadows, midtones, highlights in xyz; w is unused. All sections default to identity.
 * midtonesRange defaults to [0.2, 0.7], with a 0.1 crossfade on either side.
 */
export const ColorGrading = component(
    "ColorGrading",
    {
        exposure: f32,
        temperature: f32,
        tint: f32,
        /** Hue rotation in degrees. */
        hue: f32,
        postSaturation: f32,
        midtonesRange: vec2,
        saturation: vec4,
        contrast: vec4,
        gamma: vec4,
        gain: vec4,
        lift: vec4,
    },
    { defaults: () => gradingDefaults },
);

export const gradingDefaults = {
    exposure: 0,
    temperature: 0,
    tint: 0,
    hue: 0,
    postSaturation: 1,
    midtonesRange: [0.2, 0.7] as [number, number],
    saturation: [1, 1, 1, 0] as [number, number, number, number],
    contrast: [1, 1, 1, 0] as [number, number, number, number],
    gamma: [1, 1, 1, 0] as [number, number, number, number],
    gain: [1, 1, 1, 0] as [number, number, number, number],
    lift: [0, 0, 0, 0] as [number, number, number, number],
};

export const GradingConfig = d.struct({
    exposure: d.f32,
    temperature: d.f32,
    tint: d.f32,
    hue: d.f32,
    postSaturation: d.f32,
    tonemapMode: d.u32,
    midtonesRange: d.vec2f,
    saturation: d.vec4f,
    contrast: d.vec4f,
    gamma: d.vec4f,
    gain: d.vec4f,
    lift: d.vec4f,
});

/** Section weights, including the corrected lower crossover from Blender 3.6. */
export const gradingWeights = tgpu.fn(
    [d.f32, d.vec2f],
    d.vec3f,
)((level, range) => {
    "use gpu";
    const weights = d.vec3f(0);
    if (level < range.x - 0.1) weights.x = 1;
    else if (level < range.x + 0.1) {
        weights.y = (level - range.x) * 5 + 0.5;
        // Blender v3.6.0 COM_ColorCorrectionOperation.cc:67–69. Bevy's z here is a bug.
        weights.x = 1 - weights.y;
    } else if (level < range.y - 0.1) weights.y = 1;
    else if (level < range.y + 0.1) {
        weights.z = (level - range.y) * 5 + 0.5;
        weights.y = 1 - weights.z;
    } else weights.z = 1;
    return weights;
});

const rotateHue = tgpu.fn(
    [d.vec3f, d.f32],
    d.vec3f,
)(`(rgb: vec3f, hue: f32) -> vec3f {
    let hi = max(rgb.r, max(rgb.g, rgb.b));
    let lo = min(rgb.r, min(rgb.g, rgb.b));
    let c = hi - lo;
    if (c == 0.0) { return rgb; }
    var swizzle = vec3f(0.0);
    if (hi == rgb.r) { swizzle = vec3f(rgb.gb, 0.0); }
    else if (hi == rgb.g) { swizzle = vec3f(rgb.br, 2.0); }
    else { swizzle = vec3f(rgb.rg, 4.0); }
    let h = (1.0471975511965976 * (((swizzle.x - swizzle.y) / c + swizzle.z) % 6.0) + hue) % 6.283185307179586;
    let s = c / hi;
    let k = (vec3f(5.0, 3.0, 1.0) + h / 1.0471975511965976) % 6.0;
    return hi - hi * s * max(vec3f(0.0), min(k, min(4.0 - k, vec3f(1.0))));
}`);

export const grade = tgpu.fn(
    [d.vec3f, GradingConfig],
    d.vec3f,
)((input, g) => {
    "use gpu";
    let color = std.max(input, d.vec3f(0));
    if (g.hue !== 0) color = rotateHue(color, std.radians(g.hue));
    if (g.temperature !== 0 || g.tint !== 0) {
        const xy = d.vec2f(0.31272 - g.temperature, 0.32903 + g.tint);
        const white = std.add(
            d.vec3f(0.701634, 1.15856, -0.904175),
            std.div(
                std.add(
                    d.vec3f(-0.051461, 0.045854, 0.953127),
                    std.mul(d.vec3f(0.452749, -0.296122, -0.955206), xy.x),
                ),
                xy.y,
            ),
        );
        const toLms = d.mat3x3f(
            d.vec3f(0.311692, 0.0905138, 0.00764433),
            d.vec3f(0.652085, 0.901341, 0.0486554),
            d.vec3f(0.0362225, 0.00814478, 0.9437),
        );
        const toRgb = d.mat3x3f(
            d.vec3f(4.06305, -0.40791, -0.0118812),
            d.vec3f(-2.93241, 1.40437, -0.0486532),
            d.vec3f(-0.130646, 0.0035363, 1.0605344),
        );
        color = std.max(
            std.mul(
                toRgb,
                std.mul(std.div(d.vec3f(0.975538, 1.01648, 1.08475), white), std.mul(toLms, color)),
            ),
            d.vec3f(0),
        );
    }
    const weights = gradingWeights((color.x + color.y + color.z) / 3, g.midtonesRange);
    const saturation = std.dot(weights, g.saturation.xyz);
    const contrast = std.dot(weights, g.contrast.xyz);
    const gamma = std.dot(weights, g.gamma.xyz);
    const gain = std.dot(weights, g.gain.xyz);
    const lift = std.dot(weights, g.lift.xyz);
    const luma = tmLuma(color);
    color = std.add(luma, std.mul(saturation, std.sub(color, luma)));
    color = std.add(0.5, std.mul(std.sub(color, 0.5), contrast));
    color = std.add(std.mul(color, gain), lift);
    color = std.mul(std.pow(std.abs(color), d.vec3f(1 / gamma)), std.sign(color));
    return std.max(std.mul(color, std.exp2(g.exposure)), d.vec3f(0));
});
