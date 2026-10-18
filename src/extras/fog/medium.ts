import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { distanceAttenuation, PointLightGpu, spotFactor } from "../../standard/rendering";

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

/**
 * one volumetric point/spot light's in-scatter radiance at a froxel centre: `color · distanceAttenuation ·
 * spotFactor · phase`, the same per-light terms standard's lit path uses (it calls the same two functions).
 * Shadow-free — the caller multiplies the shadow factor. `params.x`'s magnitude is the source radius; its
 * sign is the `VolumetricLight` flag, squared away here.
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

/** One volumetric directional light's in-scatter radiance, with no distance falloff or cone. */
export const directionalInScatter = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.f32],
    d.vec3f,
)((lightColor, travelDirection, dir, g) => {
    "use gpu";
    return std.mul(lightColor, henyeyGreenstein(g, std.dot(dir, std.neg(travelDirection))));
});
