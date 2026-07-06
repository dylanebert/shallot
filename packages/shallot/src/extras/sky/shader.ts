// The procedural sky fragment, ported from the engine's earlier `SKY_WGSL` (recovered from git history —
// our own validated code). It runs behind sear's backdrop seam: a view-ray → HDR color recipe whose `fs`
// writes `col = sampleSky(dir)`, with the `Sky` uniform (declared by the background's `bindings`) and the
// sear-provided `lighting` uniform in scope. The sun's *direction* comes from `lighting` (the plugin reads
// the sun, writes nothing — a day-night cycle is a separate, deferred plugin that writes the sun); its
// *appearance* comes from the `Sky` config. No moon (it needs a moon-direction source — deferred with the
// day-night cycle).

// the `Sky` uniform — the trimmed descendant of the old struct (no moon, no own sun direction). std140:
// the leading two f32s + two pad floats fill the first 16-byte slot, then every field is a vec4.
//   starParams = (intensity, amount, _, _)   cloudParams = (coverage, density, height, _)
//   sunParams  = (size, _, _, glow)          sunVisualColor = rgb sun tint
const SKY_STRUCT_WGSL = /* wgsl */ `
struct Sky {
    hazeDensity: f32,
    horizonBand: f32,
    _pad0: f32,
    _pad1: f32,
    hazeColor: vec4<f32>,
    skyZenith: vec4<f32>,
    skyHorizon: vec4<f32>,
    starParams: vec4<f32>,
    cloudParams: vec4<f32>,
    cloudColor: vec4<f32>,
    sunParams: vec4<f32>,
    sunVisualColor: vec4<f32>,
}`;

// simplex noise for the cloud FBM (Gustavson 2D simplex)
const NOISE_WGSL = /* wgsl */ `
fn hash2(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn simplex2(p: vec2<f32>) -> f32 {
    let K1 = 0.366025404;
    let K2 = 0.211324865;

    let i = floor(p + (p.x + p.y) * K1);
    let a = p - i + (i.x + i.y) * K2;

    let o = select(vec2(0.0, 1.0), vec2(1.0, 0.0), a.x > a.y);
    let b = a - o + K2;
    let c = a - 1.0 + 2.0 * K2;

    let h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), vec3(0.0));
    let h4 = h * h * h * h;

    let n = vec3(
        dot(a, vec2(hash2(i) * 2.0 - 1.0, hash2(i + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(b, vec2(hash2(i + o) * 2.0 - 1.0, hash2(i + o + vec2(0.0, 1.0)) * 2.0 - 1.0)),
        dot(c, vec2(hash2(i + 1.0) * 2.0 - 1.0, hash2(i + vec2(1.0, 2.0)) * 2.0 - 1.0))
    );

    return dot(h4, n) * 70.0;
}

const FBM2_OCTAVES = 5;

fn fbm2(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < FBM2_OCTAVES; i++) {
        value += amplitude * simplex2(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}`;

// hash-grid stars: a cell grid over (azimuth, elevation), one candidate star per cell with hashed position,
// brightness, twinkle, and color temperature
const STARS_WGSL = /* wgsl */ `
fn hashStar(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash2Star(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3(p.x, p.y, p.x) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

fn sampleStars(dir: vec3<f32>) -> vec3<f32> {
    if (dir.y < 0.0) {
        return vec3(0.0);
    }

    let theta = atan2(dir.z, dir.x);
    let phi = asin(clamp(dir.y, -1.0, 1.0));

    let gridSize = mix(20.0, 100.0, sky.starParams.y);
    let cell = vec2(theta * gridSize / 3.14159, phi * gridSize / 1.5708);
    let cellId = floor(cell);

    var starColor = vec3(0.0);

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let neighbor = cellId + vec2(f32(dx), f32(dy));
            let starHash = hashStar(neighbor);

            if (starHash > sky.starParams.y * 0.7) {
                continue;
            }

            let starPos = hash2Star(neighbor);
            let starCenter = neighbor + starPos;
            let dist = length(cell - starCenter);

            let brightness = hashStar(neighbor + vec2(100.0, 100.0));
            let radius = 0.02 + brightness * 0.03;

            if (dist < radius) {
                let twinkle = 0.8 + 0.2 * sin(brightness * 100.0);
                let intensity = sky.starParams.x * brightness * twinkle;
                let falloff = 1.0 - smoothstep(0.0, radius, dist);

                let temp = hashStar(neighbor + vec2(200.0, 200.0));
                let tint = mix(vec3(1.0, 0.9, 0.8), vec3(0.8, 0.9, 1.0), temp);

                starColor = max(starColor, tint * intensity * falloff);
            }
        }
    }

    return starColor;
}`;

// FBM clouds projected onto a plane above the horizon, fading out toward the zenith and at the horizon
const CLOUDS_WGSL = /* wgsl */ `
fn sampleClouds(dir: vec3<f32>) -> vec4<f32> {
    if (dir.y < 0.01) {
        return vec4(0.0);
    }

    let t = sky.cloudParams.z / max(dir.y, 0.001);
    let uv = dir.xz * t;

    var n = fbm2(uv);

    let coverage = sky.cloudParams.x;
    let density = sky.cloudParams.y;
    n = smoothstep(1.0 - coverage, 1.0, n * 0.5 + 0.5) * density;

    n *= smoothstep(0.0, 0.15, dir.y);

    return vec4(sky.cloudColor.rgb, n);
}`;

// the full preamble spliced into the background's WGSL: the `Sky` struct + the sky math. The fragment is
// `col = sampleSky(dir)`.
export const SKY_WGSL = /* wgsl */ `
${SKY_STRUCT_WGSL}
${NOISE_WGSL}
${STARS_WGSL}
${CLOUDS_WGSL}

fn sampleSky(dir: vec3<f32>) -> vec3<f32> {
    // elevation gradient, softened toward the horizon (pow 0.25), with an optional bright horizon band
    let t = pow(clamp(dir.y, 0.0, 1.0), 0.25);
    var color = mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, t);

    if (sky.horizonBand > 0.0) {
        let horizonBlend = pow(1.0 - abs(dir.y), 32.0) * sky.horizonBand;
        let bandColor = sky.skyHorizon.rgb * 1.5;
        color = mix(color, bandColor, horizonBlend);
    }

    color += sampleStars(dir);

    let clouds = sampleClouds(dir);
    color = mix(color, clouds.rgb, clouds.a);

    // the sun: glow gates on sunGlow, disk on sunSize, so a sun-less sky sets both to 0
    // lighting.sunDirection is the light-travel direction (toward the ground); the sun disk
    // sits opposite it, so negate to point at the sun.
    let sunDir = -lighting.sunDirection.xyz;
    let sunDot = dot(dir, sunDir);

    let sunVisualColor = sky.sunVisualColor.rgb;

    let glowStrength = sky.sunParams.w;
    if (glowStrength > 0.0) {
        // Henyey-Greenstein glow + a tight warm corona
        let g = 0.76;
        let gg = g * g;
        let mie = (1.0 - gg) / pow(1.0 + gg - 2.0 * g * sunDot, 1.5);
        color += sunVisualColor * mie * glowStrength * 0.025;

        let angle = max(0.0, sunDot);
        let corona = pow(angle, 512.0) * 0.4 + pow(angle, 128.0) * 0.06;
        let warmTint = vec3f(1.0, 0.9, 0.7);
        color += warmTint * sunVisualColor * corona * glowStrength;
    }

    // limb-darkened sun disk with a faint chromatic fringe at the edge
    let baseSunSize = 0.9995;
    let sunSizeParam = sky.sunParams.x;
    let sunThreshold = 1.0 - (1.0 - baseSunSize) * sunSizeParam;
    let sunEdgeWidth = (1.0 - sunThreshold) * 0.15;

    let diskBlend = smoothstep(sunThreshold - sunEdgeWidth, sunThreshold + sunEdgeWidth, sunDot);
    if (diskBlend > 0.0) {
        let radial = saturate((sunDot - sunThreshold) / (1.0 - sunThreshold));
        let r = 1.0 - radial;
        let mu = sqrt(1.0 - r * r);
        let limbDarken = 1.0 - 0.6 * (1.0 - mu);
        color += sunVisualColor * limbDarken * diskBlend;

        let edgeDist = 1.0 - smoothstep(0.0, 1.0, radial);
        let fringe = vec3f(
            smoothstep(0.3, 0.7, edgeDist),
            smoothstep(0.5, 0.9, edgeDist),
            smoothstep(0.7, 1.0, edgeDist)
        );
        color += fringe * sunVisualColor * 0.15 * diskBlend * (1.0 - radial);
    }

    if (sky.hazeDensity > 0.0) {
        let horizonFactor = 1.0 - clamp(dir.y, 0.0, 1.0);
        let hazeAmount = pow(horizonFactor, 2.0) * saturate(sky.hazeDensity * 5.0);
        color = mix(color, sky.hazeColor.rgb, hazeAmount);
    }

    return color;
}`;

// the `Sky` uniform's byte size + float count (12 vec4-aligned slots: 2 leading f32s + 2 pad + 10 vec4s).
export const SKY_BYTES = 144;
export const SKY_FLOATS = SKY_BYTES / 4;
