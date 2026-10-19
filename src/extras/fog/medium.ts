import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { octDecodeNormal } from "../../engine/utils";
import { PointLightGpu, spotFactor } from "../../standard/rendering";

/** Compute workgroup tile: 8×8 = 64 threads. */
export const WORKGROUP = 8;

/** Frustum grid dimensions, capped to the view's width and height; the extra depth texel stores the near boundary. */
export const FOG_SLICES = 64;
export const FOG_WIDTH = 160;
export const FOG_HEIGHT = 90;

// a captured constant folds to a literal in the emitted WGSL, so this is the shader's PI too
const PI = Math.PI;

/**
 * the `Fog` uniform's layout, the one source of truth for it: `color.rgb` is the medium's own light, linear radiance in cd/m², `medium`
 * is (extinction density, heightBase, heightFalloff, viewDistance), `extra` is
 * (maxDistance, scatteringAsymmetry, scattering albedo, radiance gain). f32
 * throughout. `FOG_FLOATS` and `packFog`'s write indices derive
 * from it.
 */
export const FogGpu = d
    .struct({
        color: d.vec4f,
        medium: d.vec4f,
        extra: d.vec4f,
    })
    // Splice sites declare their binding with the emitted name Fog.
    .$name("Fog");

/** the `Fog` uniform's f32 count: the length of `packFog`'s staging array. */
export const FOG_FLOATS = d.sizeOf(FogGpu) / 4;

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
    medium: at(FogGpu, (f) => f.medium),
    extra: at(FogGpu, (f) => f.extra),
} as const;

/** exponential height fog's density at a world point: `density · exp(-falloff · (p.y - base))`. Height-only,
 *  so the x/z lanes are unread — the point is the parameter because the in-scatter loop already has one. */
export const fogDensity = tgpu.fn(
    [d.vec3f, d.f32, d.f32, d.f32],
    d.f32,
)((p, density, base, falloff) => {
    "use gpu";
    return density * std.exp(-falloff * (p.y - base));
});

/**
 * reconstruct a fragment's world position from its screen `uv` (0..1, y-down) + ndc `depth` (0..1) through
 * the camera's inverse view-projection.
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

/** the Henyey-Greenstein single-scatter phase function. `g` in [-1,1]: 0 isotropic (1/4π), →1 forward-peaked
 *  (a bright halo toward a light), →-1 back-scatter. `cosTheta` is the cosine between the view ray and the
 *  direction toward the light. */
export const henyeyGreenstein = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((g, cosTheta) => {
    "use gpu";
    const g2 = g * g;
    const denom = std.max(1 + g2 - 2 * g * cosTheta, 1e-4);
    return (1 - g2) / (4 * PI * denom * std.sqrt(denom));
});

/** One light sample along a froxel's ray segment: where it was taken and the segment's mean radiance. */
export const LightSample = d.struct({ position: d.vec3f, radiance: d.vec3f });

/**
 * The part of `[t0, t1]` along `origin + t·dir` inside a spot light's outer cone, as HDRP's `IntersectRayCone`
 * clips the segment before sampling it; empty when `x > y`. A point light keeps the whole segment.
 */
export const spotSegment = tgpu.fn(
    [PointLightGpu, d.vec3f, d.vec3f, d.f32, d.f32],
    d.vec2f,
)((light, origin, dir, t0, t1) => {
    "use gpu";
    if (light.params.z === 0) return d.vec2f(t0, t1);
    const axis = octDecodeNormal(std.bitcastF32toU32(light.params.y));
    // spotFactor reaches zero where the cone cosine is -offset / scale.
    const cosOuter = -light.params.w / light.params.z;
    const cosSq = cosOuter * cosOuter;
    const w = std.sub(origin, light.posRange.xyz);
    const da = std.dot(dir, axis);
    const wa = std.dot(w, axis);
    // (v·axis)² − cos²|v|² ≥ 0 on the double cone, v = w + t·dir; v·axis ≥ 0 keeps the lit nappe.
    const a = da * da - cosSq;
    const b = da * wa - cosSq * std.dot(dir, w);
    const c = wa * wa - cosSq * std.dot(w, w);
    const disc = b * b - a * c;
    let lo = d.f32(t0);
    let hi = d.f32(t1);
    if (std.abs(a) < 1e-6) {
        // A ray parallel to the cone's side crosses it once.
        if (std.abs(b) > 1e-12) {
            const root = (-c * 0.5) / b;
            if (b > 0) lo = std.max(lo, root);
            else hi = std.min(hi, root);
        } else if (c < 0) {
            hi = lo - 1;
        }
    } else if (disc < 0) {
        // No crossing: wholly inside the double cone when a > 0, wholly outside when a < 0.
        if (a < 0) hi = lo - 1;
    } else {
        const root = std.sqrt(disc);
        const r0 = std.min((-b - root) / a, (-b + root) / a);
        const r1 = std.max((-b - root) / a, (-b + root) / a);
        if (a < 0) {
            lo = std.max(lo, r0);
            hi = std.min(hi, r1);
        } else if (da > 0) {
            // Inside beyond r1 on the lit nappe; before r0 lies the opposite nappe.
            lo = std.max(lo, r1);
        } else {
            hi = std.min(hi, r0);
        }
    }
    // Keep the lit nappe: (w + t·dir)·axis ≥ 0.
    if (std.abs(da) > 1e-8) {
        const apex = -wa / da;
        if (da > 0) lo = std.max(lo, apex);
        else hi = std.min(hi, apex);
    } else if (wa < 0) {
        hi = lo - 1;
    }
    return d.vec2f(lo, hi);
});

/**
 * One volumetric point or spot light's mean in-scatter radiance over the ray segment `[t0, t1]` from `origin`
 * along `dir`, as HDRP samples local lights in `VolumetricLighting.compute`: clip the segment to the spot's cone,
 * then sample it equiangularly at the uniform variate `u` (`ImportanceSamplePunctualLight`, after Kulla and
 * Fajardo 2012). The light's radius is raised to at least `minSqRadius` and, as HDRP's
 * `ModifyDistancesForFillLighting`, softens the falloff to 1 / (d² + r²), which the pdf matches; only the range
 * window, the cone and the caller's shadow vary with `u`, so a lamp inside one froxel no longer depends on where
 * the jitter lands. Shadow- and phase-free: the caller multiplies both. `params.x`'s magnitude is the source
 * radius; its sign is the `VolumetricLight` flag, squared away here.
 */
export const equiangularInScatter = tgpu.fn(
    [PointLightGpu, d.vec3f, d.vec3f, d.f32, d.f32, d.f32, d.f32],
    LightSample,
)((light, origin, dir, t0, t1, u, minSqRadius) => {
    "use gpu";
    const sqRadius = std.max(light.params.x * light.params.x, minSqRadius);
    const length = t1 - t0;
    const lit = spotSegment(light, origin, dir, t0, t1);
    let t = d.f32(t0);
    // falloff / pdf = gamma / D, since d² + r² equals the pdf's D² + τ²; a short segment takes the falloff itself.
    let falloff = d.f32(0);
    if (length <= 1e-4) {
        const toLight = std.sub(light.posRange.xyz, origin);
        falloff = 1 / (std.dot(toLight, toLight) + sqRadius);
    } else if (lit.y > lit.x) {
        const toLight = std.sub(light.posRange.xyz, origin);
        const along = std.dot(toLight, dir);
        const sqD = std.max(std.dot(toLight, toLight) - along * along + sqRadius, 1e-8);
        const rcpD = std.inverseSqrt(sqD);
        const theta0 = std.atan((lit.x - along) * rcpD);
        const gamma = std.atan((lit.y - along) * rcpD) - theta0;
        t = along + sqD * rcpD * std.tan(theta0 + u * gamma);
        falloff = (gamma * rcpD) / length;
    }
    const position = std.add(origin, std.mul(dir, t));
    const toSample = std.sub(light.posRange.xyz, position);
    const distSq = std.dot(toSample, toSample);
    const L = std.mul(toSample, std.inverseSqrt(std.max(distSq, 1e-8)));
    const factor = distSq * light.posRange.w;
    const window = std.saturate(1 - factor * factor);
    return LightSample({
        position,
        radiance: std.mul(light.color.xyz, window * window * spotFactor(light, L) * falloff),
    });
});

/** One volumetric directional light's in-scatter radiance, with no distance falloff or cone. */
export const directionalInScatter = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.f32],
    d.vec3f,
)((lightColor, travelDirection, dir, g) => {
    "use gpu";
    return std.mul(lightColor, henyeyGreenstein(g, std.dot(dir, std.neg(travelDirection))));
});
