// The fog raymarch, as a single source shared three ways: the WGSL chunks the production `FogSystem` and
// the gym `render` fog probe both splice, and the TS oracle the unit tests + the fog probe-readback assert pin them to.
// S1 integrates **extinction** only (uniform haze + exponential height fog, Beer-Lambert per step); S2 adds
// **in-scatter** (clustered point/spot light shafts) on the same march loop. Keep the WGSL twins and the TS
// functions byte-for-byte equivalent in arithmetic — the validation is GPU-march == TS-march == closed-form.
import { distanceAttenuation } from "../render/core";

/** compute workgroup tile: 8×8 = 64 threads, matching glaze's screen-space composite. */
export const WORKGROUP = 8;

/** the Fog uniform: `color.rgb` linear haze color, `march` = (density, heightBase, heightFalloff, jitter),
 * `extra.x` = step count. f32 throughout; the shader reads `extra.x` as `u32`. 48 bytes. */
export const FOG_BYTES = 48;
export const FOG_FLOATS = FOG_BYTES / 4;

/** the march's constant loop cap. DXC chokes on a fully-dynamic loop bound, so the WGSL loop runs to this
 * compile-time constant and `break`s at the runtime `steps` (gpu.md "DXC: constant upper bound + dynamic
 * break"). `packFog` clamps `Fog.steps` to it, so a clamped step count integrates the full ray at the cap
 * resolution, never a silent under-integration. */
export const FOG_MAX_STEPS = 256;

export const FOG_STRUCT_WGSL = /* wgsl */ `
struct Fog {
    color: vec4<f32>,
    march: vec4<f32>,
    extra: vec4<f32>,
}`;

/**
 * the march primitives, spliced by both the production screen march and the fog probe so there is one GPU
 * source of truth. `fogDensity` is exponential height fog (the S2 in-scatter loop reuses it); `fogTransmittance`
 * is the midpoint Riemann sum of optical depth → Beer-Lambert transmittance (sampleOffset 0.5 = midpoint,
 * per-pixel-jittered on screen); `fogComposite` fades the scene toward the haze color by extinction.
 */
export const FOG_MARCH_WGSL = /* wgsl */ `
fn fogDensity(p: vec3<f32>, density: f32, base: f32, falloff: f32) -> f32 {
    return density * exp(-falloff * (p.y - base));
}

fn fogTransmittance(
    origin: vec3<f32>,
    dir: vec3<f32>,
    dist: f32,
    density: f32,
    base: f32,
    falloff: f32,
    steps: u32,
    sampleOffset: f32,
) -> f32 {
    let ds = dist / f32(steps);
    var tau = 0.0;
    for (var i = 0u; i < ${FOG_MAX_STEPS}u; i = i + 1u) {
        if (i >= steps) { break; }
        let p = origin + dir * ((f32(i) + sampleOffset) * ds);
        tau = tau + fogDensity(p, density, base, falloff) * ds;
    }
    return exp(-tau);
}

fn fogComposite(scene: vec3<f32>, fogColor: vec3<f32>, transmittance: f32) -> vec3<f32> {
    return scene * transmittance + fogColor * (1.0 - transmittance);
}`;

/**
 * the in-scatter primitives (S2 clustered lights + S3 sun), spliced by the production fog shader and the
 * fog probe so there is one GPU source of truth. `henyeyGreenstein` is the single-scatter phase function (g
 * 0 isotropic → 1 forward-peaked, a bright halo toward a light); `inScatterContribution` is one volumetric
 * point/spot light's radiance at a march point (`lightColor · distanceAttenuation · spotFactor · phase`),
 * the same per-light terms sear's lit path uses; `sunInScatter` is the directional sun's `sunColor ·
 * phase`, no falloff/cone. Both shadow-free (the caller multiplies the shadow factor). Splice
 * POINT_LIGHTS_STRUCT_WGSL + OCT_ENCODE_WGSL + LIGHT_EVAL_WGSL (sear/core, for `distanceAttenuation` /
 * `spotFactor`) before it.
 */
export const FOG_INSCATTER_WGSL = /* wgsl */ `
const FOG_PI = 3.14159265359;

fn henyeyGreenstein(g: f32, cosTheta: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
    return (1.0 - g2) / (4.0 * FOG_PI * denom * sqrt(denom));
}

fn inScatterContribution(light: PointLightGpu, p: vec3<f32>, dir: vec3<f32>, g: f32) -> vec3<f32> {
    let toLight = light.posRange.xyz - p;
    let distSq = dot(toLight, toLight);
    let radiusSq = light.params.x * light.params.x;
    let L = toLight * inverseSqrt(max(distSq, 1e-4));
    let atten = distanceAttenuation(distSq, light.posRange.w, radiusSq);
    let phase = henyeyGreenstein(g, dot(dir, L));
    return light.color.rgb * (atten * spotFactor(light, L) * phase);
}

// the sun (directional) in-scatter contribution at a march point: sunColor (intensity baked) weighted by
// the HG phase toward the light. sunDir is the sun's travel direction (lighting.sunDirection.xyz), so
// -sunDir is toward the light; no distance falloff or cone (directional, infinitely far). Shadow-free —
// the caller multiplies the per-step sun shadow. Additive with the clustered cones on the same march
fn sunInScatter(sunColor: vec3<f32>, sunDir: vec3<f32>, dir: vec3<f32>, g: f32) -> vec3<f32> {
    return sunColor * henyeyGreenstein(g, dot(dir, -sunDir));
}`;

// --- TS oracle: the twin the WGSL above is pinned to (the gym readback asserts diff against these). ---

/** exponential-height-fog density at world height `py`: `density · exp(-falloff · (py - base))`. */
export function fogDensity(py: number, density: number, base: number, falloff: number): number {
    return density * Math.exp(-falloff * (py - base));
}

/** transmittance `exp(-τ)` along `[originY, dirY]·dist`, midpoint Riemann sum over `steps`
 * (sampleOffset 0.5 = midpoint). The TS twin of WGSL `fogTransmittance`: density depends only on
 * height, so the y components are all it needs. */
export function fogTransmittance(
    originY: number,
    dirY: number,
    dist: number,
    density: number,
    base: number,
    falloff: number,
    steps: number,
    sampleOffset: number,
): number {
    const ds = dist / steps;
    let tau = 0;
    for (let i = 0; i < steps; i++) {
        const py = originY + dirY * ((i + sampleOffset) * ds);
        tau += fogDensity(py, density, base, falloff) * ds;
    }
    return Math.exp(-tau);
}

/** the closed-form optical depth τ of exponential height fog along `[originY, dirY]·dist`: the analytic
 * integral `∫₀ᴸ density·exp(-falloff·(originY + dirY·t - base)) dt` the midpoint march converges to. The
 * unit test's ground truth (march vs this within the midpoint-rule error bound). */
export function heightOpticalDepth(
    originY: number,
    dirY: number,
    dist: number,
    density: number,
    base: number,
    falloff: number,
): number {
    const a = density * Math.exp(-falloff * (originY - base));
    const k = falloff * dirY;
    if (Math.abs(k) < 1e-9) return a * dist;
    return (a * (1 - Math.exp(-k * dist))) / k;
}

/** composite the marched extinction over the scene color: `scene·T + fogColor·(1-T)`. Twin of WGSL `fogComposite`. */
export function fogComposite(
    scene: readonly [number, number, number],
    fogColor: readonly [number, number, number],
    transmittance: number,
): [number, number, number] {
    const f = 1 - transmittance;
    return [
        scene[0] * transmittance + fogColor[0] * f,
        scene[1] * transmittance + fogColor[1] * f,
        scene[2] * transmittance + fogColor[2] * f,
    ];
}

/** the Henyey-Greenstein single-scatter phase function (the WGSL `henyeyGreenstein` twin). `g` in [-1,1]:
 * 0 isotropic (1/4π), →1 forward-peaked (a bright halo toward a light), →-1 back-scatter. `cosTheta` is the
 * cosine between the view ray and the direction toward the light. */
export function henyeyGreenstein(g: number, cosTheta: number): number {
    const g2 = g * g;
    const denom = Math.max(1 + g2 - 2 * g * cosTheta, 1e-4);
    return (1 - g2) / (4 * Math.PI * denom * Math.sqrt(denom));
}

/** one volumetric light in the decoded terms of the GPU `PointLightGpu` the fog march reads: `pos` world
 * position, `invRangeSq` = 1/range² (posRange.w), `radius` the params.x soft-sphere radius (its magnitude;
 * the sign is the Volumetric flag, squared away here), `color` linear rgb (intensity baked), and the spot
 * cone: `coneAxis` the oct-decoded forward, `coneScale`/`coneOffset` the angular (scale, offset);
 * `coneScale === 0` is a plain point (no cone). */
export interface FogLight {
    pos: readonly [number, number, number];
    invRangeSq: number;
    radius: number;
    color: readonly [number, number, number];
    coneAxis: readonly [number, number, number];
    coneScale: number;
    coneOffset: number;
}

/** the atmosphere knobs the in-scatter integral reads: `density`/`base`/`falloff` the same extinction height
 * fog as {@link fogTransmittance}, `absorption` the absorbed fraction (scattering albedo = 1 − absorption),
 * `gain` the combined scatter intensity (scattering · scatterIntensity), `anisotropy` the HG `g`. */
export interface FogScatter {
    density: number;
    base: number;
    falloff: number;
    absorption: number;
    gain: number;
    anisotropy: number;
}

/** one volumetric light's in-scatter radiance at march point `p` (shadow-free; the caller multiplies the
 * shadow factor). The TS twin of WGSL `inScatterContribution`: `color · distanceAttenuation · spotFactor ·
 * HG-phase`. */
export function inScatterContribution(
    light: FogLight,
    p: readonly [number, number, number],
    dir: readonly [number, number, number],
    g: number,
): [number, number, number] {
    const tx = light.pos[0] - p[0];
    const ty = light.pos[1] - p[1];
    const tz = light.pos[2] - p[2];
    const distSq = tx * tx + ty * ty + tz * tz;
    const radiusSq = light.radius * light.radius;
    const inv = 1 / Math.sqrt(Math.max(distSq, 1e-4));
    const lx = tx * inv;
    const ly = ty * inv;
    const lz = tz * inv;
    const atten = distanceAttenuation(distSq, light.invRangeSq, radiusSq);
    let spot = 1;
    if (light.coneScale !== 0) {
        const cd = -(light.coneAxis[0] * lx + light.coneAxis[1] * ly + light.coneAxis[2] * lz);
        const a = Math.min(Math.max(cd * light.coneScale + light.coneOffset, 0), 1);
        spot = a * a;
    }
    const phase = henyeyGreenstein(g, dir[0] * lx + dir[1] * ly + dir[2] * lz);
    const f = atten * spot * phase;
    return [light.color[0] * f, light.color[1] * f, light.color[2] * f];
}

/** the single-light in-scatter march: the oracle the fog probe's in-scatter readback is pinned to (the
 * production shader sums this over a froxel's volumetric lights; the probe + this run one light). Marches
 * the same midpoint samples as {@link fogTransmittance}, weighting each step's in-scatter by the
 * transmittance to the step start and integrating the source over the step **analytically**
 * (Hillaire/Frostbite energy-conserving form): the in-scatter over a step of optical depth `σ_t·ds` is
 * `albedo·(1−e^{−σ_t·ds})·gain·source`, the within-step extinction folded in, the analytic twin of the
 * haze composite's `(1−T)`, not a `source·ds` rectangle. Makes shaft brightness step-count-stable. */
export function fogInScatter(
    origin: readonly [number, number, number],
    dir: readonly [number, number, number],
    dist: number,
    fog: FogScatter,
    light: FogLight,
    steps: number,
    sampleOffset: number,
): [number, number, number] {
    const ds = dist / steps;
    const albedo = 1 - fog.absorption;
    let trans = 1;
    let r = 0;
    let g = 0;
    let b = 0;
    for (let i = 0; i < steps; i++) {
        const t = (i + sampleOffset) * ds;
        const p: [number, number, number] = [
            origin[0] + dir[0] * t,
            origin[1] + dir[1] * t,
            origin[2] + dir[2] * t,
        ];
        const d = fogDensity(p[1], fog.density, fog.base, fog.falloff);
        const sampleTrans = Math.exp(-d * ds);
        const c = inScatterContribution(light, p, dir, fog.anisotropy);
        const w = trans * albedo * fog.gain * (1 - sampleTrans);
        r += w * c[0];
        g += w * c[1];
        b += w * c[2];
        trans *= sampleTrans;
    }
    return [r, g, b];
}

/** the directional sun in the terms the fog march reads: `direction` the sun's normalized travel direction
 * (lighting.sunDirection.xyz; toward-light is its negation), `color` the linear rgb with intensity baked
 * (lighting.sunColor.rgb). No position/range/cone: directional, infinitely far. */
export interface FogSun {
    direction: readonly [number, number, number];
    color: readonly [number, number, number];
}

/** the sun's in-scatter contribution at a march point (shadow-free; the caller multiplies the shadow):
 * `color · HG(g, dot(dir, -sunDir))`. The TS twin of WGSL `sunInScatter`. */
export function sunInScatter(
    sunColor: readonly [number, number, number],
    sunDir: readonly [number, number, number],
    dir: readonly [number, number, number],
    g: number,
): [number, number, number] {
    const cosTheta = -(dir[0] * sunDir[0] + dir[1] * sunDir[1] + dir[2] * sunDir[2]);
    const phase = henyeyGreenstein(g, cosTheta);
    return [sunColor[0] * phase, sunColor[1] * phase, sunColor[2] * phase];
}

/** the single-sun in-scatter march: the oracle the fog probe's sun-in-scatter readback is pinned to (the
 * production shader adds this to the clustered cones on the same march; the probe + this run the sun alone,
 * shadow-free, the no-occluder analytic). Marches the same midpoint samples as {@link fogTransmittance},
 * weighting each step by the transmittance to its start and integrating the source over the step with the
 * same energy-conserving `albedo·(1−e^{−σ_t·ds})·gain` form as {@link fogInScatter}. */
export function fogSunInScatter(
    origin: readonly [number, number, number],
    dir: readonly [number, number, number],
    dist: number,
    fog: FogScatter,
    sun: FogSun,
    steps: number,
    sampleOffset: number,
): [number, number, number] {
    const ds = dist / steps;
    const albedo = 1 - fog.absorption;
    const c = sunInScatter(sun.color, sun.direction, dir, fog.anisotropy);
    let trans = 1;
    let r = 0;
    let g = 0;
    let b = 0;
    for (let i = 0; i < steps; i++) {
        const py = origin[1] + dir[1] * ((i + sampleOffset) * ds);
        const d = fogDensity(py, fog.density, fog.base, fog.falloff);
        const sampleTrans = Math.exp(-d * ds);
        const w = trans * albedo * fog.gain * (1 - sampleTrans);
        r += w * c[0];
        g += w * c[1];
        b += w * c[2];
        trans *= sampleTrans;
    }
    return [r, g, b];
}

/** reconstruct a fragment's world position from its screen `uv` (0..1, y-down) + ndc `depth` (0..1) using
 * the camera's inverse view-projection (column-major). The TS twin of the WGSL `reconstructWorld` the
 * production march runs each pixel; its round-trip against `viewProj` is the reconstruction unit test. */
export function reconstruct(
    invViewProj: Float32Array,
    u: number,
    v: number,
    depth: number,
): [number, number, number] {
    const nx = u * 2 - 1;
    const ny = 1 - v * 2;
    const m = invViewProj;
    const x = m[0] * nx + m[4] * ny + m[8] * depth + m[12];
    const y = m[1] * nx + m[5] * ny + m[9] * depth + m[13];
    const z = m[2] * nx + m[6] * ny + m[10] * depth + m[14];
    const w = m[3] * nx + m[7] * ny + m[11] * depth + m[15];
    return [x / w, y / w, z / w];
}
