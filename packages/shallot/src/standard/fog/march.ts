// The fog raymarch. Each primitive is ONE TGSL function: the same source resolves to the WGSL the
// production `FogSystem` and the gym `render` fog probe splice, and runs on the CPU as the oracle the unit
// tests + the probe-readback assert pin the GPU to. There is no twin to keep in step.
//
// S1 integrates **extinction** only (uniform haze + exponential height fog, Beer-Lambert per step); S2 adds
// **in-scatter** — the clustered point/spot shafts and the directional sun — on the same march loop.
//
// Two things here are deliberately NOT TGSL. `heightOpticalDepth` is the analytic closed form the march is
// validated against, so it has no shader twin by construction. `fogInScatter` / `fogSunInScatter` are the
// single-light march oracles: the production shader fuses that loop with the per-step light gather + shadow
// lookups, so there is no WGSL function to collapse onto — but they call the TGSL primitives below, never a
// second copy of the arithmetic.
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, octEncodeWgsl, spliceNs } from "../../engine/utils/core";
import { distanceAttenuation, PointLightGpu, pointLightsWgsl, spotFactor } from "../render/core";
import { lightEvalWgsl } from "../sear/core";

/** compute workgroup tile: 8×8 = 64 threads, matching glaze's screen-space composite. */
export const WORKGROUP = 8;

/** the march's constant loop cap. DXC chokes on a fully-dynamic loop bound, so the marched loop runs to this
 * compile-time constant and `break`s at the runtime `steps` (gpu.md "DXC: constant upper bound + dynamic
 * break"); a captured JS constant folds to a literal in the emitted WGSL. `packFog` clamps `Fog.steps` to it,
 * so a clamped step count integrates the full ray at the cap resolution, never a silent under-integration. */
export const FOG_MAX_STEPS = 256;

// a captured constant folds to a literal in the emitted WGSL, so this is the shader's PI too — the
// `FOG_PI` the hand-written in-scatter WGSL declared is gone with it
const PI = Math.PI;

/**
 * the `Fog` uniform's layout, the one source of truth for it: `color.rgb` is the linear haze color, `march`
 * is (density, heightBase, heightFalloff, jitter), `extra` is (steps, anisotropy g, absorption, gain). f32
 * throughout — the shader reads `extra.x` as `u32`. `FOG_BYTES` / `FOG_FLOATS` / `packFog`'s write indices
 * all derive from it, and {@link fogStructWgsl} emits it as the WGSL `Fog` struct.
 */
export const FogGpu = d
    .struct({
        color: d.vec4f,
        march: d.vec4f,
        extra: d.vec4f,
    })
    // the emitted struct is `Fog` (what every splice site declares its binding as); the TS name carries the
    // `Gpu` suffix so it doesn't collide with the `Fog` component, the `PointLight` / `PointLightGpu` shape
    .$name("Fog");

/** the `Fog` uniform's byte size, from {@link FogGpu} — a consumer binding it sizes to match. */
export const FOG_BYTES = d.sizeOf(FogGpu);

/** the `Fog` uniform's f32 count: the length of `packFog`'s staging array. */
export const FOG_FLOATS = FOG_BYTES / 4;

// the f32 index of one field, from the schema — `packFog` fills a Float32Array, so the schema stays the one
// source for where each value lands (a reordered struct moves the writes with it)
const at = <T extends d.BaseData>(schema: T, field: (p: d.Infer<T>) => unknown) =>
    d.memoryLayoutOf(schema, field).offset / 4;

/**
 * the f32 indices `packFog` writes at, derived from {@link FogGpu}.
 * @internal
 */
export const FOG_PARAMS = {
    color: at(FogGpu, (f) => f.color),
    march: at(FogGpu, (f) => f.march),
    extra: at(FogGpu, (f) => f.extra),
} as const;

/** the WGSL `Fog` struct, relocatable so the production march and the fog probe declare the same uniform
 *  {@link packFog} packs. Splice **before** that declaration. */
export const fogStructWgsl = chunk("fogStructWgsl", [FogGpu], spliceNs);

// ---- extinction (S1): the midpoint march + the haze composite + the per-pixel ray setup ----

/** exponential height fog's density at a world point: `density · exp(-falloff · (p.y - base))`. Height-only,
 *  so the x/z lanes are unread — the point is the parameter because the in-scatter loop already has one.
 *  @example let dens = fogDensity(p, density, base, falloff); */
export const fogDensity = tgpu.fn(
    [d.vec3f, d.f32, d.f32, d.f32],
    d.f32,
)((p, density, base, falloff) => {
    "use gpu";
    return density * std.exp(-falloff * (p.y - base));
});

/**
 * transmittance `exp(-τ)` along `origin + dir·[0, dist]`: the midpoint Riemann sum of optical depth through
 * Beer-Lambert (`sampleOffset` 0.5 = midpoint, per-pixel-jittered on screen). The loop runs to the constant
 * {@link FOG_MAX_STEPS} and breaks at the runtime `steps` — the DXC constant-bound shape.
 *
 * @example let t = fogTransmittance(nearWorld, dir, dist, density, base, falloff, steps, offset);
 */
export const fogTransmittance = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32, d.f32, d.f32, d.f32, d.u32, d.f32],
    d.f32,
)((origin, dir, dist, density, base, falloff, steps, sampleOffset) => {
    "use gpu";
    const ds = dist / d.f32(steps);
    let tau = d.f32(0);
    let i = d.u32(0);
    while (i < FOG_MAX_STEPS) {
        if (i >= steps) break;
        const p = std.add(origin, std.mul(dir, (d.f32(i) + sampleOffset) * ds));
        tau = tau + fogDensity(p, density, base, falloff) * ds;
        i = i + 1;
    }
    return std.exp(-tau);
});

/** composite the marched extinction over the scene color: `scene·T + fogColor·(1−T)`.
 *  @example let outc = fogComposite(scn, fog.color.rgb, t); */
export const fogComposite = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32],
    d.vec3f,
)((scene, fogColor, transmittance) => {
    "use gpu";
    return std.add(std.mul(scene, transmittance), std.mul(fogColor, 1 - transmittance));
});

/**
 * reconstruct a fragment's world position from its screen `uv` (0..1, y-down) + ndc `depth` (0..1) through
 * the camera's inverse view-projection. The matrix is a parameter, not a `view` global, so the function
 * stays pure and its round-trip against a `viewProj` is a unit test — the production march calls it with
 * `view.invViewProj`.
 *
 * @example let fragWorld = reconstructWorld(view.invViewProj, uv, depth);
 */
export const reconstructWorld = tgpu.fn(
    [d.mat4x4f, d.vec2f, d.f32],
    d.vec3f,
)((invViewProj, uv, depth) => {
    "use gpu";
    const ndc = d.vec3f(uv.x * 2 - 1, 1 - uv.y * 2, depth);
    const h = std.mul(invViewProj, d.vec4f(ndc, 1));
    return std.div(h.xyz, h.w);
});

/** interleaved gradient noise (Jimenez 2014) — the cheap per-pixel offset that turns march banding into
 *  noise.
 *  @example let offset = mix(0.5, ign(vec2f(gid.xy)), jitter); */
export const ign = tgpu.fn(
    [d.vec2f],
    d.f32,
)((p) => {
    "use gpu";
    return std.fract(52.9829189 * std.fract(std.dot(p, d.vec2f(0.06711056, 0.00583715))));
});

/**
 * the march primitives, spliced by both the production screen march and the fog probe so there is one GPU
 * source of truth: {@link fogDensity} (exponential height fog, reused by the S2 in-scatter loop),
 * {@link fogTransmittance} (the midpoint optical-depth sum → Beer-Lambert transmittance),
 * {@link fogComposite} (fade the scene toward the haze color by extinction), plus the per-pixel ray setup
 * {@link reconstructWorld} + {@link ign}. Needs no bindings — every input is a parameter.
 */
export const fogMarchWgsl = chunk(
    "fogMarchWgsl",
    [fogDensity, fogTransmittance, fogComposite, reconstructWorld, ign],
    spliceNs,
);

// ---- in-scatter (S2 clustered lights + S3 sun) ----

/** the Henyey-Greenstein single-scatter phase function. `g` in [-1,1]: 0 isotropic (1/4π), →1 forward-peaked
 *  (a bright halo toward a light), →-1 back-scatter. `cosTheta` is the cosine between the view ray and the
 *  direction toward the light.
 *  @example let phase = henyeyGreenstein(g, dot(dir, L)); */
export const henyeyGreenstein = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((g, cosTheta) => {
    "use gpu";
    const g2 = g * g;
    const denom = std.max(1 + g2 - 2 * g * cosTheta, 1e-4);
    return (1 - g2) / (4 * PI * denom * std.sqrt(denom));
});

/**
 * one volumetric point/spot light's in-scatter radiance at a march point: `color · distanceAttenuation ·
 * spotFactor · phase`, the same per-light terms sear's lit path uses (it calls the same two functions).
 * Shadow-free — the caller multiplies the shadow factor. `params.x`'s magnitude is the source radius; its
 * sign is the `Volumetric` flag, squared away here.
 *
 * @example lstep += inScatterContribution(light, p, dir, g) * shadow;
 */
export const inScatterContribution = tgpu.fn(
    [PointLightGpu, d.vec3f, d.vec3f, d.f32],
    d.vec3f,
)((light, p, dir, g) => {
    "use gpu";
    const toLight = std.sub(light.posRange.xyz, p);
    const distSq = std.dot(toLight, toLight);
    const radiusSq = light.params.x * light.params.x;
    const L = std.mul(toLight, std.inverseSqrt(std.max(distSq, 1e-4)));
    const atten = distanceAttenuation(distSq, light.posRange.w, radiusSq);
    const phase = henyeyGreenstein(g, std.dot(dir, L));
    return std.mul(light.color.xyz, atten * spotFactor(light, L) * phase);
});

/**
 * the sun's (directional) in-scatter contribution at a march point: `sunColor · phase` toward the light.
 * `sunDir` is the sun's travel direction (`lighting.sunDirection.xyz`), so `-sunDir` is toward the light; no
 * distance falloff or cone (directional, infinitely far). Shadow-free — the caller multiplies the per-step
 * sun shadow. Additive with the clustered cones on the same march.
 *
 * @example lstep += sunInScatter(lighting.sunColor.rgb, lighting.sunDirection.xyz, dir, g) * shadow;
 */
export const sunInScatter = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.f32],
    d.vec3f,
)((sunColor, sunDir, dir, g) => {
    "use gpu";
    return std.mul(sunColor, henyeyGreenstein(g, std.dot(dir, std.neg(sunDir))));
});

/**
 * the in-scatter primitives, spliced by the production fog shader and the fog probe:
 * {@link henyeyGreenstein}, the clustered {@link inScatterContribution}, and the directional
 * {@link sunInScatter}. Splice **after** `pointLightsWgsl()` + `octEncodeWgsl()` + `lightEvalWgsl()`
 * (`sear/core`) — the contribution calls their `distanceAttenuation` / `spotFactor`.
 */
export function fogInScatterWgsl(): string {
    // force the base chunks first, so `PointLightGpu` and the light-eval primitives land in the chunks that
    // own them whatever order a consumer asks in (they are spliced ahead of this one either way)
    pointLightsWgsl();
    octEncodeWgsl();
    lightEvalWgsl();
    return inScatterChunk();
}

const inScatterChunk = chunk(
    "fogInScatterWgsl",
    [henyeyGreenstein, inScatterContribution, sunInScatter],
    spliceNs,
);

// ---- the CPU-only tier: the analytic ground truth + the two single-light march oracles ----

/** the closed-form optical depth τ of exponential height fog along `[originY, dirY]·dist`: the analytic
 * integral `∫₀ᴸ density·exp(-falloff·(originY + dirY·t - base)) dt` the midpoint march converges to. The
 * unit test's ground truth (march vs this within the midpoint-rule error bound), so it has no shader twin. */
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

/** the single-light in-scatter march: the oracle the fog probe's in-scatter readback is pinned to (the
 * production shader sums this over a froxel's volumetric lights; the probe + this run one light). Marches
 * the same midpoint samples as {@link fogTransmittance}, weighting each step's in-scatter by the
 * transmittance to the step start and integrating the source over the step **analytically**
 * (Hillaire/Frostbite energy-conserving form): the in-scatter over a step of optical depth `σ_t·ds` is
 * `albedo·(1−e^{−σ_t·ds})·gain·source`, the within-step extinction folded in, the analytic twin of the
 * haze composite's `(1−T)`, not a `source·ds` rectangle. Makes shaft brightness step-count-stable. */
export function fogInScatter(
    origin: d.v3f,
    dir: d.v3f,
    dist: number,
    fog: FogScatter,
    light: d.Infer<typeof PointLightGpu>,
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
        const p = d.vec3f(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);
        const dens = fogDensity(p, fog.density, fog.base, fog.falloff);
        const sampleTrans = Math.exp(-dens * ds);
        const c = inScatterContribution(light, p, dir, fog.anisotropy);
        const w = trans * albedo * fog.gain * (1 - sampleTrans);
        r += w * c.x;
        g += w * c.y;
        b += w * c.z;
        trans *= sampleTrans;
    }
    return [r, g, b];
}

/** the directional sun in the terms the fog march reads: `direction` the sun's normalized travel direction
 * (lighting.sunDirection.xyz; toward-light is its negation), `color` the linear rgb with intensity baked
 * (lighting.sunColor.rgb). No position/range/cone: directional, infinitely far. */
export interface FogSun {
    direction: d.v3f;
    color: d.v3f;
}

/** the single-sun in-scatter march: the oracle the fog probe's sun-in-scatter readback is pinned to (the
 * production shader adds this to the clustered cones on the same march; the probe + this run the sun alone,
 * shadow-free, the no-occluder analytic). Marches the same midpoint samples as {@link fogTransmittance},
 * weighting each step by the transmittance to its start and integrating the source over the step with the
 * same energy-conserving `albedo·(1−e^{−σ_t·ds})·gain` form as {@link fogInScatter}. */
export function fogSunInScatter(
    origin: d.v3f,
    dir: d.v3f,
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
        const t = (i + sampleOffset) * ds;
        // the full point, not just its y lane: the fused production march samples `origin + dir·t` and
        // fogDensity reads `.y` off that, so the f32 rounding has to happen on the same value
        const p = d.vec3f(origin.x + dir.x * t, origin.y + dir.y * t, origin.z + dir.z * t);
        const dens = fogDensity(p, fog.density, fog.base, fog.falloff);
        const sampleTrans = Math.exp(-dens * ds);
        const w = trans * albedo * fog.gain * (1 - sampleTrans);
        r += w * c.x;
        g += w * c.y;
        b += w * c.z;
        trans *= sampleTrans;
    }
    return [r, g, b];
}
