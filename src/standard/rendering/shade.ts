// StandardRenderer's shading substrate: the schemas that define its shadow-uniform layouts and the TGSL functions
// its color FS shades with — the half of the renderer that is pure math over pure data, sitting beside
// the ECS/system/registry code in `forward.ts` (the kernel-sibling convention).
//
// Two kinds live here, and the split is forced by what a function reaches for:
//
//   - **TGSL.** A function whose inputs are all parameters (the metallic-roughness BRDF set, the cube-face
//     pick, the receiver remap) is one source that runs on the CPU and resolves to the WGSL a shader
//     splices. That is what makes the invariants its comments claim testable — `brdfSphere` reducing to
//     `brdf` at radius 0, `pointReceiver` matching the projection the atlas renders through.
//   - **WGSL-bodied `tgpu.fn`.** The two shadow samplers read their atlas, sampler, and caster uniform as
//     module-scope globals *the consumer declares by name* — that is the relocatable contract standard's color
//     FS and the fog march share, and it has no TGSL spelling.
//     A WGSL body still resolves under strict naming (the emitted name is the authored one) and still
//     takes its struct parameters from the schemas below, so the layout has one source of truth even
//     where the body doesn't.
//
// The one raw-WGSL chunk, `casterWgsl`, is a lazily-resolved thunk (`chunk`) in the engine-wide `spliceNs`.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils";
import { PointLightGpu } from "./lighting";
import { EDGE_TEXELS, MAX_CASCADES, MAX_POINT_CASTERS } from "./shadows";

// ---- the metallic-roughness shading model (glTF 2.0), the `default` / `vertex` / glTF surfaces' lobe ----

/**
 * the shading inputs `litPbr` takes: `dielectric` is the non-metal base reflectance (F0) — the engine
 * StandardMaterial passes 0 for zero specular (the flat Shallot look); glTF passes the spec-standard 0.04.
 */
export const Pbr = d.struct({
    albedo: d.vec3f,
    metallic: d.f32,
    roughness: d.f32,
    occlusion: d.f32,
    dielectric: d.f32,
    diffuseWrap: d.f32,
});

// a captured constant folds to a literal in the emitted WGSL, so this is the shader's PI too
const PI = Math.PI;

/**
 * Valve half-Lambert: remap the diffuse cosine from `[-1,1]` to `[0,1]` and square it, so the gradient
 * spans the whole surface and the terminator softens — the matte, non-plastic happy-path look (Mitton &
 * McTaggart, "Shading in Valve's Source Engine", GDC 2004). The square (not the bare remap) keeps form:
 * the remap alone flattens, squaring restores midtone contrast. Diffuse-only — the specular cosine stays
 * physical, so metals + glTF dielectrics are unchanged. Deliberately not energy-conserving.
 */
export const halfLambert = tgpu.fn(
    [d.f32],
    d.f32,
)((ndl) => {
    "use gpu";
    const h = ndl * 0.5 + 0.5;
    return h * h;
});

/** the GGX / Trowbridge-Reitz normal distribution over the half-vector cosine `ndh` and the squared
 *  roughness `a`. */
export const distributionGGX = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((ndh, a) => {
    "use gpu";
    const a2 = a * a;
    const den = ndh * ndh * (a2 - 1) + 1;
    return a2 / std.max(PI * den * den, 1e-7);
});

/** Smith height-correlated visibility (`G / (4·ndl·ndv)`), Heitz 2014. */
export const visSmithGGX = tgpu.fn(
    [d.f32, d.f32, d.f32],
    d.f32,
)((ndl, ndv, a) => {
    "use gpu";
    const a2 = a * a;
    const lv = ndl * std.sqrt(ndv * ndv * (1 - a2) + a2);
    const ll = ndv * std.sqrt(ndl * ndl * (1 - a2) + a2);
    return 0.5 / std.max(lv + ll, 1e-7);
});

/** Schlick Fresnel with f90 derived from F0 (Frostbite, Lagarde 2014): a true zero-reflectance material
 *  (F0 = 0) gets f90 = 0, so its specular vanishes entirely — grazing included. F0 ≥ ~0.02 saturates to
 *  the standard f90 = 1. This is what makes `dielectric = 0` mean literally no specular. */
export const fresnelSchlick = tgpu.fn(
    [d.f32, d.vec3f],
    d.vec3f,
)((vdh, f0) => {
    "use gpu";
    const f90 = std.saturate(std.dot(f0, d.vec3f(50 / 3)));
    const f = std.pow(std.saturate(1 - vdh), 5);
    return std.add(f0, std.mul(f, std.sub(d.vec3f(f90), f0)));
});

/**
 * one light's Cook-Torrance BRDF, unscaled by photometric radiance / attenuation (the caller scales).
 * Diffuse retains its physical `albedo/PI` normalization; half-Lambert remains the authored soft response.
 */
export const brdf = tgpu.fn(
    [Pbr, d.vec3f, d.vec3f, d.vec3f],
    d.vec3f,
)((s, N, V, L) => {
    "use gpu";
    const dNL = std.dot(N, L);
    const ndl = std.max(dNL, 0);
    const H = std.normalize(std.add(V, L));
    const ndv = std.max(std.dot(N, V), 1e-4);
    const ndh = std.max(std.dot(N, H), 0);
    const vdh = std.max(std.dot(V, H), 0);
    const a = std.max(s.roughness * s.roughness, 1e-3);
    const f0 = std.mix(d.vec3f(s.dielectric), s.albedo, s.metallic);
    const F = fresnelSchlick(vdh, f0);
    // the operand order is the shipped shader's, term for term: reassociating a float product changes
    // the last bits, and this lobe is what the bench's shaded probes compare against
    const spec = std.mul(distributionGGX(ndh, a) * visSmithGGX(ndl, ndv, a), F);
    const kd = std.mul(std.sub(d.vec3f(1), F), 1 - s.metallic);
    const diffuseCosine = std.mix(std.max(dNL, 0), halfLambert(dNL), s.diffuseWrap);
    const diffuse = std.mul(std.div(std.mul(kd, s.albedo), PI), diffuseCosine);
    return std.add(diffuse, std.mul(spec, ndl));
});

/**
 * the sphere-source BRDF for a point light: diffuse on the light CENTER (`Lc`, half-Lambert, identical to
 * {@link brdf}), specular on Karis's representative point (Real Shading in UE4) — the closest point on the
 * source sphere to the mirror ray. A point source gives a pinpoint highlight a rough surface
 * barely catches; a sphere of radius `radius` gives a soft round highlight scaled to the source size. The
 * roughness widens to `aPrime` by the solid angle the sphere subtends and the peak is renormalized by
 * `(a/aPrime)²`, so total specular energy is conserved (the highlight spreads, it doesn't brighten). At
 * radius 0 the representative point is `Lc`, `aPrime = a`, norm = 1, so this reduces to `brdf(s, N, V, Lc)`
 * exactly (pinned by unit test).
 */
export const brdfSphere = tgpu.fn(
    [Pbr, d.vec3f, d.vec3f, d.vec3f, d.f32, d.f32],
    d.vec3f,
)((s, N, V, Lc, dist, radius) => {
    "use gpu";
    const r = std.reflect(std.neg(V), N);
    // the unnormalized light vector (Lc normalized, |Lvec| = dist)
    const Lvec = std.mul(dist, Lc);
    const centerToRay = std.sub(std.mul(std.dot(Lvec, r), r), Lvec);
    const closest = std.add(
        Lvec,
        std.mul(std.saturate(radius / std.max(std.length(centerToRay), 1e-4)), centerToRay),
    );
    const Ls = std.normalize(closest);
    const a = std.max(s.roughness * s.roughness, 1e-3);
    const aPrime = std.saturate(a + radius / (2 * std.max(dist, 1e-4)));
    const norm = (a / aPrime) * (a / aPrime);

    const dC = std.dot(N, Lc);
    const ndl = std.max(std.dot(N, Ls), 0);
    const H = std.normalize(std.add(V, Ls));
    const ndv = std.max(std.dot(N, V), 1e-4);
    const ndh = std.max(std.dot(N, H), 0);
    const vdh = std.max(std.dot(V, H), 0);
    const f0 = std.mix(d.vec3f(s.dielectric), s.albedo, s.metallic);
    const F = fresnelSchlick(vdh, f0);
    const spec = std.mul(
        std.mul(distributionGGX(ndh, aPrime) * visSmithGGX(ndl, ndv, aPrime), F),
        norm,
    );
    const kd = std.mul(std.sub(d.vec3f(1), F), 1 - s.metallic);
    const diffuseCosine = std.mix(std.max(dC, 0), halfLambert(dC), s.diffuseWrap);
    const diffuse = std.mul(std.div(std.mul(kd, s.albedo), PI), diffuseCosine);
    return std.add(diffuse, std.mul(spec, ndl));
});

// ---- point / spot shadows: the caster uniform layout + the receiver math ----

/**
 * one shadowed point/spot caster slot: `pos` = light world position with the source entity id in `w`
 * (`-1` for an empty slot, which matches no light); `nf` = (near, unused, depthBias, normalBias); `spotA/B/C`
 * = the cone's lookAt basis (right / up / fwd) with the widened cone tangent in `spotA.w` (0 for a point
 * caster, which uses cube faces instead).
 */
export const PointCaster = d.struct({
    pos: d.vec4f,
    nf: d.vec4f,
    spotA: d.vec4f,
    spotB: d.vec4f,
    spotC: d.vec4f,
});

/** the `PointCaster` stride in f32 (5 vec4) — the staging mirror's row size, from the schema. */
export const POINT_CASTER_FLOATS = d.sizeOf(PointCaster) / 4;

/** the caster-slot uniform, sized to {@link MAX_POINT_CASTERS}; a slot past the `PointShadows.casters`
 * limit stays empty (`pos.w = -1`), so no light matches it. */
export function pointCastersSchema() {
    // a factory-built schema is anonymous, and an unnamed one resolves to `struct item` — name it, or
    // the raw splice sites that declare `var<uniform> pointShadows: PointCasters` reference nothing
    return d.struct({ casters: d.arrayOf(PointCaster, MAX_POINT_CASTERS) }).$name("PointCasters");
}

/**
 * the per-(caster, face) allocated atlas-UV rects (`[u0, v0, du, dv]`, square), indexed `slot·6 + face`:
 * a point caster's six face tiles, a spot's lone tile at face 0. The receiver samples its matched
 * caster's rect; the importance allocator (`standard/shadows.ts`) sizes + packs them each frame.
 */
export function tileRectsSchema(slots: number) {
    return d.struct({ rects: d.arrayOf(d.vec4f, slots) }).$name("TileRects");
}

/**
 * the atlas VS's per-combo tile-folded view-projections: each
 * combo's viewProj has its atlas tile placement folded in (`tileTransform`), so the VS projects straight
 * into its tile with no manual divide. `slots` is `6 · MAX_POINT_CASTERS` for the point atlas,
 * `MAX_CASCADES` for the cascade atlas — the same sizing {@link tileRectsSchema} takes, so the three
 * schemas always agree on slot count for one atlas pass.
 */
export function faceVPsSchema(slots: number) {
    return d.struct({ m: d.arrayOf(d.mat4x4f, slots) }).$name("FaceVPs");
}

/**
 * the atlas VS's per-combo meta: the (caster slot, face) — or
 * (cascade index, …) for the cascade atlas — each dense combo maps to, which the VS reads to index its
 * tile rect. Same `slots` as {@link faceVPsSchema} / {@link tileRectsSchema}.
 */
export function comboMetaSchema(slots: number) {
    return d.struct({ m: d.arrayOf(d.vec4u, slots) }).$name("ComboMeta");
}

/** the face a light→fragment direction falls in: `face` is the dominant axis (X≥Y≥Z precedence), `stz`
 *  the face-camera coordinates — `s`/`t` along that face's right/up and `z` the forward distance the
 *  projection divides by. */
export const PointFace = d.struct({
    stz: d.vec3f,
    face: d.u32,
});

/**
 * the cube face a light→fragment direction `dir` falls in, plus its face-camera coordinates. One source
 * for the receiver (standard's color FS, the fog march) and the CPU: the six face bases are axis-aligned, so
 * each dot product folds to a signed component pick. Pinned to the `POINT_FACES` table the atlas render's
 * viewProjs come from (`shadows.test.ts`), which is what keeps the two halves of the projection agreeing.
 */
export const pointFaceOf = tgpu.fn(
    [d.vec3f],
    PointFace,
)((dir) => {
    "use gpu";
    const a = std.abs(dir);
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x >= 0) return PointFace({ stz: d.vec3f(dir.z, dir.y, dir.x), face: d.u32(0) });
        return PointFace({ stz: d.vec3f(-dir.z, dir.y, -dir.x), face: d.u32(1) });
    }
    if (a.y >= a.z) {
        if (dir.y >= 0) return PointFace({ stz: d.vec3f(dir.x, dir.z, dir.y), face: d.u32(2) });
        return PointFace({ stz: d.vec3f(dir.x, -dir.z, -dir.y), face: d.u32(3) });
    }
    if (dir.z >= 0) return PointFace({ stz: d.vec3f(-dir.x, dir.y, dir.z), face: d.u32(4) });
    return PointFace({ stz: d.vec3f(dir.x, dir.y, -dir.z), face: d.u32(5) });
});

/** Move a receiver toward its light by a fixed world-space distance. */
export const offsetTowardLight = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.f32],
        d.vec3f,
    )((position, towardLight, distance) => {
        "use gpu";
        return std.add(position, std.mul(std.normalize(towardLight), distance));
    })
    .$name("offsetTowardLight");

/** Infinite reverse-Z perspective depth for a receiver already offset in world space. */
export const pointReceiver = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((z, near) => {
    "use gpu";
    return near / std.max(z, near);
});

/** returns the point/spot caster WGSL: the {@link PointCaster} / `PointCasters` / `TileRects` structs a
 *  consumer declares its group-1 shadow bindings over. Splice **before** those declarations. */
export const casterWgsl = chunk(
    "casterWgsl",
    () => {
        return [pointCastersSchema(), tileRectsSchema(MAX_POINT_CASTERS * 6)];
    },
    spliceNs,
);

// Bevy's fixed 2×2 and Castaño 13-tap map filters, expressed over one atlas tile. `rect` keeps
// comparison taps inside the selected tile; map-space offsets are measured in that tile's pixels.
function mapHardwareWgsl(texture: "shadowMap" | "pointAtlas") {
    return /* wgsl */ `(uv: vec2f, depth: f32, rect: vec4f) -> f32 {
    let atlasSize = vec2<f32>(textureDimensions(${texture}));
    let lo = rect.xy + vec2<f32>(0.5) / atlasSize;
    let hi = rect.xy + rect.zw - vec2<f32>(0.5) / atlasSize;
    return textureSampleCompareLevel(${texture}, shadowSamp, clamp(uv, lo, hi), depth);
}`;
}

const sampleShadowMapHardware = tgpu
    .fn(
        [d.vec2f, d.f32, d.vec4f],
        d.f32,
    )(mapHardwareWgsl("shadowMap"))
    .$name("sample_shadow_map_hardware");
const sampleShadowPointMapHardware = tgpu
    .fn(
        [d.vec2f, d.f32, d.vec4f],
        d.f32,
    )(mapHardwareWgsl("pointAtlas"))
    .$name("sample_shadow_point_map_hardware");

const castanoMapWgsl = /* wgsl */ `(uv: vec2f, depth: f32, rect: vec4f) -> f32 {
    let atlasSize = vec2<f32>(textureDimensions(shadowMap));
    let tileSize = rect.zw * atlasSize;
    let invTileSize = 1.0 / tileSize;
    let tileUv = (uv - rect.xy) / rect.zw;
    let pixelUv = tileUv * tileSize;
    let basePixel = floor(pixelUv + 0.5);
    let s = pixelUv.x + 0.5 - basePixel.x;
    let t = pixelUv.y + 0.5 - basePixel.y;
    let baseUv = (basePixel - 0.5) * invTileSize;

    let uw0 = 4.0 - 3.0 * s;
    let uw1 = 7.0;
    let uw2 = 1.0 + 3.0 * s;
    let u0 = (3.0 - 2.0 * s) / uw0 - 2.0;
    let u1 = (3.0 + s) / uw1;
    let u2 = s / uw2 + 2.0;
    let vw0 = 4.0 - 3.0 * t;
    let vw1 = 7.0;
    let vw2 = 1.0 + 3.0 * t;
    let v0 = (3.0 - 2.0 * t) / vw0 - 2.0;
    let v1 = (3.0 + t) / vw1;
    let v2 = t / vw2 + 2.0;

    var sum = 0.0;
    sum += uw0 * vw0 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u0, v0) * invTileSize) * rect.zw, depth, rect);
    sum += uw1 * vw0 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u1, v0) * invTileSize) * rect.zw, depth, rect);
    sum += uw2 * vw0 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u2, v0) * invTileSize) * rect.zw, depth, rect);
    sum += uw0 * vw1 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u0, v1) * invTileSize) * rect.zw, depth, rect);
    sum += uw1 * vw1 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u1, v1) * invTileSize) * rect.zw, depth, rect);
    sum += uw2 * vw1 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u2, v1) * invTileSize) * rect.zw, depth, rect);
    sum += uw0 * vw2 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u0, v2) * invTileSize) * rect.zw, depth, rect);
    sum += uw1 * vw2 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u1, v2) * invTileSize) * rect.zw, depth, rect);
    sum += uw2 * vw2 * sample_shadow_map_hardware(rect.xy + (baseUv + vec2<f32>(u2, v2) * invTileSize) * rect.zw, depth, rect);
    return sum / 144.0;
}`;

const sampleShadowMapCastanoThirteen = tgpu
    .fn(
        [d.vec2f, d.f32, d.vec4f],
        d.f32,
    )(castanoMapWgsl)
    .$uses({ ["sample_shadow_map_hardware"]: sampleShadowMapHardware })
    .$name("sample_shadow_map_castano_thirteen");
const sampleShadowPointMapCastanoThirteen = tgpu
    .fn(
        [d.vec2f, d.f32, d.vec4f],
        d.f32,
    )(
        castanoMapWgsl
            .replaceAll("shadowMap", "pointAtlas")
            .replaceAll("sample_shadow_map_hardware", "sample_shadow_point_map_hardware"),
    )
    .$uses({ ["sample_shadow_point_map_hardware"]: sampleShadowPointMapHardware })
    .$name("sample_shadow_point_map_castano_thirteen");

// Bevy's cubemap Gaussian pattern projected onto Shallot's face atlas. Each offset resolves its own cube
// face before looking up that face's tile, so samples crossing a face edge stay continuous.
const sampleShadowCubemapHardware = tgpu
    .fn(
        [d.vec3f, d.f32, d.u32, d.f32],
        d.f32,
    )(/* wgsl */ `(lightLocal: vec3f, near: f32, caster: u32, tanHalf: f32) -> f32 {
    let face = pointFaceOf(lightLocal);
    let rect = tileRects.rects[caster * 6u + face.face];
    let ndc = face.stz.xy / (max(face.stz.z, near) * tanHalf);
    let uv = rect.xy + vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * rect.zw;
    let lo = rect.xy + vec2<f32>(0.5 / f32(textureDimensions(pointAtlas).x));
    let hi = rect.xy + rect.zw - vec2<f32>(0.5 / f32(textureDimensions(pointAtlas).x));
    let receiver = near / max(face.stz.z, near);
    return textureSampleCompareLevel(pointAtlas, shadowSamp, clamp(uv, lo, hi), receiver);
}`)
    .$uses({ pointFaceOf })
    .$name("sample_shadow_cubemap_hardware");

const sampleShadowCubemapGaussian = tgpu
    .fn(
        [d.vec3f, d.f32, d.u32, d.f32, d.f32],
        d.f32,
    )(/* wgsl */ `(lightLocal: vec3f, near: f32, caster: u32, tanHalf: f32, distanceToLight: f32) -> f32 {
    let direction = normalize(lightLocal);
    let signZ = select(-1.0, 1.0, direction.z >= 0.0);
    let a = -1.0 / (signZ + direction.z);
    let b = direction.x * direction.y * a;
    let xBasis = vec3<f32>(1.0 + signZ * direction.x * direction.x * a, signZ * b, -signZ * direction.x);
    let yBasis = vec3<f32>(b, signZ + direction.y * direction.y * a, -direction.y);
    let scale = 0.003 * distanceToLight;
    let bx = xBasis * scale;
    let by = yBasis * scale;
    var sum = 0.0;
    sum += 0.157112 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(0.125, -0.375).x * bx + vec2<f32>(0.125, -0.375).y * by, near, caster, tanHalf);
    sum += 0.157112 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(-0.125, 0.375).x * bx + vec2<f32>(-0.125, 0.375).y * by, near, caster, tanHalf);
    sum += 0.138651 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(0.625, 0.125).x * bx + vec2<f32>(0.625, 0.125).y * by, near, caster, tanHalf);
    sum += 0.130251 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(-0.375, -0.625).x * bx + vec2<f32>(-0.375, -0.625).y * by, near, caster, tanHalf);
    sum += 0.114946 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(-0.625, 0.625).x * bx + vec2<f32>(-0.625, 0.625).y * by, near, caster, tanHalf);
    sum += 0.114946 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(-0.875, -0.125).x * bx + vec2<f32>(-0.875, -0.125).y * by, near, caster, tanHalf);
    sum += 0.107982 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(0.375, 0.875).x * bx + vec2<f32>(0.375, 0.875).y * by, near, caster, tanHalf);
    sum += 0.079001 * sample_shadow_cubemap_hardware(lightLocal + vec2<f32>(0.875, -0.875).x * bx + vec2<f32>(0.875, -0.875).y * by, near, caster, tanHalf);
    return sum;
}`)
    .$uses({ ["sample_shadow_cubemap_hardware"]: sampleShadowCubemapHardware })
    .$name("sample_shadow_cubemap_gaussian");

// the receiver — relocatable, so it takes the world position as a parameter and reads `pointAtlas` /
// `shadowSamp` / `pointShadows` / `tileRects` as globals the consumer declares (standard's color group 1, the
// fog march's). Match the light to a caster slot by source entity id (`color.a`, baked by the light
// compact pass; `pos.w` is -1 for an empty slot, so a non-caster never matches), pick the cube face (or
// spot tile) from the light→fragment direction, project into the atlas tile, and filter by camera method.
// It reads the atlas size from the bound texture and loops every slot, so no `PointShadows` setting enters WGSL.
function pointShadowFn() {
    return tgpu
        .fn(
            [PointLightGpu, d.vec3f, d.vec3f, d.u32],
            d.f32,
        )(/* wgsl */ `(light: PointLightGpu, normal: vec3f, fragWorld: vec3f, method: u32) -> f32 {
    let atlas = f32(textureDimensions(pointAtlas).x);
    for (var k = 0u; k < ${MAX_POINT_CASTERS}u; k = k + 1u) {
        let c = pointShadows.casters[k];
        if (c.pos.w != light.color.a) { continue; }
        let toFrag = fragWorld - c.pos.xyz;
        let coneTanHalf = c.spotA.w;
        if (coneTanHalf > 0.0) {
            let rect = tileRects.rects[k * 6u];
            let tilePx = rect.z * atlas;
            let texelWorld = max(length(toFrag), 1e-4) * (2.0 * coneTanHalf / tilePx);
            let biased = offsetTowardLight(toFrag, -toFrag, c.nf.z);
            let dOff = biased + normal * (c.nf.w * 1.4142136 * texelWorld);
            let z = max(dot(dOff, c.spotC.xyz), c.nf.x);
            let receiver = pointReceiver(z, c.nf.x);
            let ndc = vec2<f32>(dot(dOff, c.spotA.xyz), dot(dOff, c.spotB.xyz)) / (z * coneTanHalf);
            let uv = rect.xy + vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * rect.zw;
            if (method == 0u) { return sample_shadow_point_map_hardware(uv, receiver, rect); }
            return sample_shadow_point_map_castano_thirteen(uv, receiver, rect);
        }
        let tilePx = tileRects.rects[k * 6u].z * atlas;
        let tanHalf = 1.0 + ${2 * EDGE_TEXELS}.0 / tilePx;
        let texelWorld = max(length(toFrag), 1e-4) * (2.0 * tanHalf / tilePx);
        let biased = offsetTowardLight(toFrag, -toFrag, c.nf.z);
        let dOff = biased + normal * (c.nf.w * 1.4142136 * texelWorld);
        if (method == 0u) {
            return sample_shadow_cubemap_hardware(dOff, c.nf.x, k, tanHalf);
        }
        // Bevy measures kernel scale before receiver bias, by the dominant axis rather than Euclidean distance.
        let distanceToLight = max(abs(toFrag.x), max(abs(toFrag.y), abs(toFrag.z)));
        return sample_shadow_cubemap_gaussian(dOff, c.nf.x, k, tanHalf, distanceToLight);
    }
    return 1.0;
}`)
        .$uses({
            pointFaceOf,
            pointReceiver,
            offsetTowardLight,
            ["sample_shadow_point_map_hardware"]: sampleShadowPointMapHardware,
            ["sample_shadow_point_map_castano_thirteen"]: sampleShadowPointMapCastanoThirteen,
            ["sample_shadow_cubemap_hardware"]: sampleShadowCubemapHardware,
            ["sample_shadow_cubemap_gaussian"]: sampleShadowCubemapGaussian,
        })
        .$name("pointShadowOf");
}

// the real reference, memoized so every pipeline calling it (standard's color FS, the fog march)
// shares the exact same object.
let _pointShadowOf: ReturnType<typeof pointShadowFn> | undefined;

/** the point/spot shadow receiver as a real callable reference: `pointShadowOf(light, normal, fragWorld, method)`.
 * Its body still reads `pointAtlas` / `shadowSamp` / `pointShadows` / `tileRects` as free names (the
 * relocatable-global law) — a real-reference caller must force those bindings into scope itself (its own
 * layout's binding declarations are invisible to `tgpu.resolve`'s call-graph walk, same as the `hullData`
 * forcing touch), while `pointFaceOf`/`pointReceiver` resolve for free via the `$uses` above. */
export function pointShadowRef() {
    return (_pointShadowOf ??= pointShadowFn());
}

// ---- sun (directional) shadows: the CSM params layout + the cascade sampler ----

/**
 * one CSM cascade's shadow params: its light viewProj, its atlas-UV `rect`, its `far` bound in linear
 * view-z (the receiver selects by these), and `texelWorld` — one shadow texel's world size for this
 * cascade, the normal-offset bias scale. The two trailing pad words the hand-written struct carried are
 * implicit: the `mat4x4f` sets the struct's 16-byte alignment, so its size rounds to 96 either way (and
 * `array<Cascade, N>` strides by that) — `shade.test.ts` pins the numbers the staging writer indexes by.
 */
export const Cascade = d.struct({
    lightViewProj: d.mat4x4f,
    rect: d.vec4f,
    far: d.f32,
    texelWorld: d.f32,
});

/**
 * the sun-shadow params uniform: the per-cascade array plus a globals tail (`count` active cascades,
 * `overlap` the inter-cascade blend-band fraction — Bevy's cascades_overlap_proportion —, the two bias
 * knobs, and `enabled` (0 = no caster, the fully-lit fallback).
 */
export const SunShadow = d.struct({
    cascades: d.arrayOf(Cascade, MAX_CASCADES),
    lightDirection: d.vec4f,
    count: d.f32,
    overlap: d.f32,
    depthBias: d.f32,
    enabled: d.f32,
    normalBias: d.f32,
});

/** byte size of the sun-shadow params uniform, from {@link SunShadow}: a relocatable consumer (the fog
 *  march) sizes its sun-shadow binding to match. */
export const SHADOW_PARAMS_BYTES = d.sizeOf(SunShadow);

/** f32 stride of one {@link Cascade} row in the params staging. */
export const CASCADE_FLOATS = d.sizeOf(Cascade) / 4;

// the f32 index of one field, from the schema — the staging writer indexes a Float32Array, so the
// schema stays the one source for where each value lands (a reordered struct moves the writes with it)
const at = <T extends d.BaseData>(schema: T, field: (p: d.Infer<T>) => unknown) =>
    d.memoryLayoutOf(schema, field).offset / 4;

/**
 * the f32 indices the sun-params staging writes at, derived from {@link Cascade} / {@link SunShadow}: a
 * per-cascade row at `cascade · k` plus the field offsets within it, then the globals tail. The writer
 * fills a `Float32Array`, so this is what keeps it and the WGSL struct one layout.
 * @internal
 */
export const SUN_PARAMS = {
    // within one cascade row (add the row base)
    cascade: {
        viewProj: at(Cascade, (c) => c.lightViewProj),
        rect: at(Cascade, (c) => c.rect),
        far: at(Cascade, (c) => c.far),
        texelWorld: at(Cascade, (c) => c.texelWorld),
    },
    // absolute in the staging (the tail follows the cascade array)
    globals: {
        count: at(SunShadow, (s) => s.count),
        overlap: at(SunShadow, (s) => s.overlap),
        depthBias: at(SunShadow, (s) => s.depthBias),
        enabled: at(SunShadow, (s) => s.enabled),
        normalBias: at(SunShadow, (s) => s.normalBias),
        lightDirection: at(SunShadow, (s) => s.lightDirection),
    },
} as const;

const sampleCascade = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec3f, d.u32],
        d.f32,
    )(/* wgsl */ `(ci: u32, worldPos: vec3f, normal: vec3f, method: u32) -> f32 {
    let c = sunShadow.cascades[ci];
    // Offset toward the light in world space before projection; normal bias remains texel-scaled.
    let biased = offsetTowardLight(worldPos, sunShadow.lightDirection.xyz, sunShadow.depthBias);
    let offset = biased + normalize(normal) * (sunShadow.normalBias * 1.4142136 * c.texelWorld);
    let lc = c.lightViewProj * vec4<f32>(offset, 1.0);
    let l = lc.xyz / lc.w;
    if (l.x < -1.0 || l.x > 1.0 || l.y < -1.0 || l.y > 1.0 || l.z < 0.0 || l.z > 1.0) {
        return 1.0; // outside this cascade box — lit
    }
    // Remap the cascade-NDC into its tile. Both filters clamp their comparison samples to this tile.
    let uv = c.rect.xy + vec2<f32>(l.x * 0.5 + 0.5, 0.5 - l.y * 0.5) * c.rect.zw;
    if (method == 0u) { return sample_shadow_map_hardware(uv, l.z, c.rect); }
    return sample_shadow_map_castano_thirteen(uv, l.z, c.rect);
}`)
    .$uses({
        offsetTowardLight,
        ["sample_shadow_map_hardware"]: sampleShadowMapHardware,
        ["sample_shadow_map_castano_thirteen"]: sampleShadowMapCastanoThirteen,
    })
    .$name("sampleCascade");

/**
 * the camera-forward distance from the eye (forward = `-cross(right, up)`) — the linear view-z the CSM
 * receiver selects a cascade by, Bevy's `get_cascade_index` axis. Pure, so the cascade fit's CPU half and
 * the receiver agree on the axis by construction.
 */
export const viewDepth = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.vec3f],
    d.f32,
)((right, up, eye, worldPos) => {
    "use gpu";
    const fwd = std.neg(std.cross(right, up));
    return std.dot(std.sub(worldPos, eye), fwd);
});

/**
 * the sun-shadow receiver as a real callable reference: `sampleSunShadow(worldPos, normal)` selects a
 * cascade by linear view-z (via {@link viewDepth}, reading `view` as a free name), PCF-samples it (via
 * {@link sampleCascade}, reading `shadowMap` / `shadowSamp` / `sunShadow`), and blends across the overlap
 * band. A real-reference caller forces `view` / `shadowMap` / `shadowSamp` / `sunShadow` into scope itself
 * (the {@link pointShadowRef} law); `viewDepth` / `sampleCascade` resolve for free via `$uses`.
 */
export const sampleSunShadow = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.u32],
        d.f32,
    )(/* wgsl */ `(worldPos: vec3f, normal: vec3f, method: u32) -> f32 {
    if (sunShadow.enabled == 0.0) { return 1.0; }
    let count = u32(sunShadow.count);
    let viewZ = viewDepth(view.right.xyz, view.up.xyz, view.eye.xyz, worldPos);
    // the first cascade whose far-bound the fragment is within
    var ci = count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (viewZ < sunShadow.cascades[i].far) { ci = i; break; }
    }
    if (ci >= count) { return 1.0; } // beyond the last cascade — lit
    var shadow = sampleCascade(ci, worldPos, normal, method);
    // blend into the next cascade across the overlap band ((1−overlap)·far … far) so the boundary has no seam
    let next = ci + 1u;
    if (next < count) {
        let thisFar = sunShadow.cascades[ci].far;
        let nextNear = (1.0 - sunShadow.overlap) * thisFar;
        if (viewZ >= nextNear) {
            let t = clamp((viewZ - nextNear) / max(thisFar - nextNear, 1e-5), 0.0, 1.0);
            shadow = mix(shadow, sampleCascade(next, worldPos, normal, method), t);
        }
    }
    return shadow;
}`)
    // its function dependencies (portable across any consumer); `sunShadow` stays a free name — a
    // real-reference caller forces it into scope itself, same as the point-shadow path above
    .$uses({ viewDepth, sampleCascade })
    .$name("sampleSunShadow");
