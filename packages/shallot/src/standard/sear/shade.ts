// Sear's shading substrate: the schemas that define its shadow-uniform layouts and the TGSL functions
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
//     module-scope globals *the consumer declares by name* — that is the relocatable contract sear's color
//     FS and the fog march share, and it has no TGSL spelling until the surface contract itself is typed.
//     A WGSL body still resolves under strict naming (the emitted name is the authored one) and still
//     takes its struct parameters from the schemas below, so the layout has one source of truth even
//     where the body doesn't.
//
// The chunks are lazily-resolved thunks (`chunk`) sharing the engine-wide `spliceNs`, each forcing the
// base chunks it depends on first so a shared dependency always lands in the lower chunk.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { chunk, spliceNs } from "../../engine/utils/core";
import { PointLightGpu, pointLightsWgsl } from "../render/core";
import { EDGE_TEXELS, MAX_CASCADES, pointAtlasSize, pointCasters } from "./shadows";

// ---- the metallic-roughness shading model (glTF 2.0), the `default` / `vertex` / glTF surfaces' lobe ----

/**
 * the shading inputs `litPbr` takes: `dielectric` is the non-metal base reflectance (F0) — the engine
 * `Material` default passes 0 for zero specular (the flat shallot look, reducing `litPbr` to `lit`
 * exactly at metallic 0 / roughness 1), glTF passes the spec-standard 0.04.
 */
export const Pbr = d.struct({
    albedo: d.vec3f,
    metallic: d.f32,
    roughness: d.f32,
    occlusion: d.f32,
    dielectric: d.f32,
});

// a captured constant folds to a literal in the emitted WGSL, so this is the shader's PI too
const PI = Math.PI;

/**
 * Valve half-Lambert: remap the diffuse cosine from `[-1,1]` to `[0,1]` and square it, so the gradient
 * spans the whole surface and the terminator softens — the matte, non-plastic happy-path look (Mitton &
 * McTaggart, "Shading in Valve's Source Engine", GDC 2004). The square (not the bare remap) keeps form:
 * the remap alone flattens, squaring restores midtone contrast. Diffuse-only — the specular cosine stays
 * physical, so metals + glTF dielectrics are unchanged. Deliberately not energy-conserving.
 *
 * @example let diffuse = halfLambert(dot(normal, L));
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
 *  roughness `a`.
 *  @example let D = distributionGGX(ndh, a); */
export const distributionGGX = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((ndh, a) => {
    "use gpu";
    const a2 = a * a;
    const den = ndh * ndh * (a2 - 1) + 1;
    return a2 / std.max(PI * den * den, 1e-7);
});

/** Smith height-correlated visibility (`G / (4·ndl·ndv)`), Heitz 2014.
 *  @example let V = visSmithGGX(ndl, ndv, a); */
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
 *  the standard f90 = 1. This is what makes `dielectric = 0` mean literally no specular.
 *  @example let F = fresnelSchlick(vdh, f0); */
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
 * one light's Cook-Torrance radiance, unscaled by light color / attenuation (the caller scales). The
 * trailing `* PI` folds physical diffuse (`albedo/PI`) back to shallot's no-PI light convention so the
 * diffuse term matches `lit` exactly. The diffuse cosine is {@link halfLambert} (the soft default); the
 * specular keeps the physical clamped cosine, so it vanishes on back faces and metals / glTF dielectrics
 * stay correct.
 *
 * @example let radiance = brdf(surface, N, V, L);
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
    const diffuse = std.mul(std.div(std.mul(kd, s.albedo), PI), halfLambert(dNL));
    return std.mul(std.add(diffuse, std.mul(spec, ndl)), PI);
});

/**
 * the sphere-source BRDF for a point light: diffuse on the light CENTER (`Lc`, half-Lambert, identical to
 * {@link brdf}), specular on Karis's representative point (Real Shading in UE4) — the closest point on the
 * source sphere to the mirror reflection ray. A point source gives a pinpoint highlight a rough surface
 * barely catches; a sphere of radius `radius` gives a soft round highlight scaled to the source size. The
 * roughness widens to `aPrime` by the solid angle the sphere subtends and the peak is renormalized by
 * `(a/aPrime)²`, so total specular energy is conserved (the highlight spreads, it doesn't brighten). At
 * radius 0 the representative point is `Lc`, `aPrime = a`, norm = 1, so this reduces to `brdf(s, N, V, Lc)`
 * exactly (pinned by unit test).
 *
 * @example let radiance = brdfSphere(surface, N, V, L, dist, light.params.x);
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
    const diffuse = std.mul(std.div(std.mul(kd, s.albedo), PI), halfLambert(dC));
    return std.mul(std.add(diffuse, std.mul(spec, ndl)), PI);
});

/** the metallic-roughness lobe sear's `lit` / `litPbr` helpers are built from: the {@link Pbr} struct,
 *  {@link halfLambert}, the GGX / Smith / Schlick terms, and the punctual + sphere-source BRDFs. Spliced
 *  into every surface module; a surface's own preamble must not redefine any of them. */
export const pbrWgsl = chunk(
    "pbrWgsl",
    [halfLambert, distributionGGX, visSmithGGX, fresnelSchlick, brdf, brdfSphere],
    spliceNs,
);

// ---- point / spot shadows: the caster uniform layout + the receiver math ----

/**
 * one shadowed point/spot caster slot: `pos` = light world position with the source entity id in `w`
 * (`-1` for an empty slot, which matches no light); `nf` = (near, far, depthBias, normalBias); `spotA/B/C`
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

/** the caster-slot uniform, sized by the `PointShadows.casters` cap (fixed before `build()`). */
export function pointCastersSchema() {
    // a factory-built schema is anonymous, and an unnamed one resolves to `struct item` — name it, or
    // the raw splice sites that declare `var<uniform> pointShadows: PointCasters` reference nothing
    return d.struct({ casters: d.arrayOf(PointCaster, pointCasters()) }).$name("PointCasters");
}

// the `PointShadows` config the caster chunks folded into their WGSL, captured at first resolve. The
// chunks are memoized process-wide (a resolved schema can't be re-emitted under the same name — a second
// resolve suffixes it `PointCasters_1`, which the raw splice site couldn't reference), while the uniforms
// are re-sized from the live config at every warm. So a config change between builds would bind a
// re-sized buffer against a stale struct: `checkShadowConfig` turns that into a named throw at warm,
// which is what "fixed before build(), like capacity" (render.md) means.
let _folded: { casters: number; atlas: number } | null = null;

/**
 * assert the live `PointShadows` config still matches what the resolved shadow WGSL folded in, throwing a
 * named error when it doesn't. Called at warm, after the chunks a build compiled: the config is fixed
 * before `build()`, so a live host that mutates it between builds fails loud rather than binding a
 * re-sized uniform against a stale struct.
 * @internal
 */
export function checkShadowConfig(): void {
    if (!_folded) return;
    const live = { casters: pointCasters(), atlas: pointAtlasSize() };
    if (_folded.casters === live.casters && _folded.atlas === live.atlas) return;
    throw new Error(
        `PointShadows changed after the shadow shaders were compiled (casters ${_folded.casters} → ` +
            `${live.casters}, atlas ${_folded.atlas} → ${live.atlas}). Both are fixed before build(), ` +
            "like capacity — set them once at startup, before the first build.",
    );
}

/**
 * the per-(caster, face) allocated atlas-UV rects (`[u0, v0, du, dv]`, square), indexed `slot·6 + face`:
 * a point caster's six face tiles, a spot's lone tile at face 0. The receiver samples its matched
 * caster's rect; the importance allocator (`sear/shadows.ts`) sizes + packs them each frame.
 */
export function tileRectsSchema(slots: number) {
    return d.struct({ rects: d.arrayOf(d.vec4f, slots) }).$name("TileRects");
}

/**
 * the atlas VS's per-combo tile-folded view-projections (`codegen.ts`'s `pointShadowCode`, group 1): each
 * combo's viewProj has its atlas tile placement folded in (`tileTransform`), so the VS projects straight
 * into its tile with no manual divide. `slots` is `6 · casters` for the point atlas, `MAX_CASCADES` for
 * the cascade atlas — the same config-folded sizing {@link tileRectsSchema} takes, so the three schemas
 * always agree on slot count for one atlas pass.
 */
export function faceVPsSchema(slots: number) {
    return d.struct({ m: d.arrayOf(d.mat4x4f, slots) }).$name("FaceVPs");
}

/**
 * the atlas VS's per-combo meta (`codegen.ts`'s `pointShadowCode`, group 1): the (caster slot, face) — or
 * (cascade index, …) for the cascade atlas — each dense combo maps to, which the VS reads to index its
 * tile rect. Same config-folded `slots` as {@link faceVPsSchema} / {@link tileRectsSchema}.
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
 * for the receiver (sear's color FS, the fog march) and the CPU: the six face bases are axis-aligned, so
 * each dot product folds to a signed component pick. Pinned to the `POINT_FACES` table the atlas render's
 * viewProjs come from (`shadows.test.ts`), which is what keeps the two halves of the projection agreeing.
 *
 * @example let f = pointFaceOf(fragWorld - casterPos);
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

/**
 * the depth a point/spot shadow receiver compares against, biased toward the light. `z` is the receiver's
 * view-space forward distance (already normal-offset), `near`/`far` the caster's clip planes, `depthBias`
 * the residual lift. The bias applies in **linear** depth: `z` is pulled toward the light by
 * `depthBias·(far−near)` world units *before* the perspective remap, so the world-space lift is constant
 * across distance. A fixed offset in the hyperbolic NDC depth (what an orthographic sun gets for free, its
 * depth being linear) instead grows with z² and detaches far contact shadows (peter-panning). The remap is
 * reverse-Z (near→1, far→0), matching the `perspective` the atlas renders through — pinned to it by unit
 * test, so the hardware depth the atlas wrote compares exactly.
 *
 * @example let receiver = pointReceiver(z, near, far, depthBias);
 */
export const pointReceiver = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((z, near, far, depthBias) => {
    "use gpu";
    const zb = std.max(z - depthBias * (far - near), near);
    return (near * (far - zb)) / (zb * (far - near));
});

/** returns the point/spot caster WGSL: the {@link PointCaster} / `PointCasters` / `TileRects` structs a
 *  consumer declares its group-1 shadow bindings over. Splice **before** those declarations, and
 *  {@link pointShadowWgsl} after them. */
export const casterWgsl = chunk(
    "casterWgsl",
    () => {
        _folded = { casters: pointCasters(), atlas: pointAtlasSize() };
        return [pointCastersSchema(), tileRectsSchema(pointCasters() * 6)];
    },
    spliceNs,
);

// the receiver — relocatable, so it takes the world position as a parameter and reads `pointAtlas` /
// `shadowSamp` / `pointShadows` / `tileRects` as globals the consumer declares (sear's color group 1, the
// fog march's). Match the light to a caster slot by source entity id (`color.a`, baked by the light
// compact pass; `pos.w` is -1 for an empty slot, so a non-caster never matches), pick the cube face (or
// spot tile) from the light→fragment direction, project into the atlas tile, and 3×3 PCF-compare.
// built lazily: its body folds the `PointShadows` config (the atlas size + the caster cap), which is only
// final after this module loads — the same reason the chunk itself is a thunk
function pointShadowFn() {
    return (
        tgpu
            .fn(
                [PointLightGpu, d.vec3f, d.vec3f],
                d.f32,
            )(/* wgsl */ `(light: PointLightGpu, normal: vec3f, fragWorld: vec3f) -> f32 {
    let atlas = ${pointAtlasSize()}.0;
    let texel = 1.0 / atlas; // one atlas pixel in uv — tile-size-independent
    for (var k = 0u; k < ${pointCasters()}u; k = k + 1u) {
        let c = pointShadows.casters[k];
        if (c.pos.w != light.color.a) { continue; }
        let toFrag = fragWorld - c.pos.xyz;
        let coneTanHalf = c.spotA.w;
        var uv: vec2<f32>;
        var receiver: f32;
        var rect: vec4<f32>;
        if (coneTanHalf > 0.0) {
            // spot caster: its single tile (face 0 of the slot). Project the normal-offset fragment onto the
            // cone's lookAt basis (right/up/fwd, c.spotA/B/C.xyz) — the texel world size (from the tile's own
            // pixel count), receiver depth, and ndc are the same forms as a cube face, just with the cone basis
            rect = tileRects.rects[k * 6u];
            let tilePx = rect.z * atlas;
            let texelWorld = max(length(toFrag), 1e-4) * (2.0 * coneTanHalf / tilePx);
            let dOff = toFrag + normal * (c.nf.w * 1.4142136 * texelWorld);
            let z = max(dot(dOff, c.spotC.xyz), c.nf.x);
            receiver = pointReceiver(z, c.nf.x, c.nf.y, c.nf.z);
            let ndc = vec2<f32>(dot(dOff, c.spotA.xyz), dot(dOff, c.spotB.xyz)) / (z * coneTanHalf);
            uv = rect.xy + vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * rect.zw;
        } else {
            // point caster: all six faces share a tile size, so the widened tangent + texel come from face 0
            // (no need to know the fragment's face yet); the offset then picks the actual face
            let tilePx = tileRects.rects[k * 6u].z * atlas;
            let tanHalf = 1.0 + ${2 * EDGE_TEXELS}.0 / tilePx;
            let texelWorld = max(length(toFrag), 1e-4) * (2.0 * tanHalf / tilePx);
            let dOff = toFrag + normal * (c.nf.w * 1.4142136 * texelWorld);
            let f = pointFaceOf(dOff);
            rect = tileRects.rects[k * 6u + f.face];
            let z = max(f.stz.z, c.nf.x);
            receiver = pointReceiver(z, c.nf.x, c.nf.y, c.nf.z);
            let ndc = f.stz.xy / (z * tanHalf);
            uv = rect.xy + vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * rect.zw;
        }
        // clamp the 3×3 PCF taps to the tile interior (half a texel in from each edge) so a grazing/wide
        // sample never bleeds into a neighbour tile — the leak fix the scissor margin alone can't give
        let lo = rect.xy + vec2<f32>(0.5 * texel);
        let hi = rect.xy + rect.zw - vec2<f32>(0.5 * texel);
        var sum = 0.0;
        for (var oy = -1; oy <= 1; oy = oy + 1) {
            for (var ox = -1; ox <= 1; ox = ox + 1) {
                let o = vec2<f32>(f32(ox), f32(oy)) * texel;
                sum = sum + textureSampleCompareLevel(pointAtlas, shadowSamp, clamp(uv + o, lo, hi), receiver);
            }
        }
        return sum / 9.0;
    }
    return 1.0;
}`)
            // `pointFaceOf`/`pointReceiver` are its only *function* dependencies (portable across any consumer,
            // unlike the atlas/sampler/caster/rect bindings below, which differ by consumer layout and stay
            // free names for the caller to declare) — naming them here is what lets a real-reference caller
            // (a typed pipeline, not just the raw-splice chunk below) pull them in via `tgpu.resolve`'s call
            // graph without also re-listing them by hand
            .$uses({ pointFaceOf, pointReceiver })
            .$name("pointShadowOf")
    );
}

// the real reference, memoized so a raw-splice consumer (`pointShadowChunk` below) and a real-reference
// consumer (a typed pipeline's kernel calling it as a function, not splicing text) share the exact same
// object — `pointShadowFn()` is a factory only because its body folds the `PointShadows` config (the atlas
// size + caster cap), which must stay fixed to one instance regardless of caller.
let _pointShadowOf: ReturnType<typeof pointShadowFn> | undefined;

/** the point/spot shadow receiver as a real callable reference: `pointShadowOf(light, normal, fragWorld)`.
 * Its body still reads `pointAtlas` / `shadowSamp` / `pointShadows` / `tileRects` as free names (the
 * relocatable-global law) — a real-reference caller must force those bindings into scope itself (its own
 * layout's binding declarations are invisible to `tgpu.resolve`'s call-graph walk, same as the `hullData`
 * forcing touch), while `pointFaceOf`/`pointReceiver` resolve for free via the `$uses` above. */
export function pointShadowRef() {
    return (_pointShadowOf ??= pointShadowFn());
}

const pointShadowChunk = chunk(
    "pointShadowWgsl",
    () => [pointFaceOf, pointReceiver, pointShadowRef()],
    spliceNs,
);

/** returns the point/spot shadow WGSL: `pointShadowOf(light, normal, fragWorld)` (world pos a parameter,
 *  atlas / sampler / casters / tile-rects referenced by name), the per-light shadow factor sear's clustered
 *  loop and a relocatable consumer both call, plus the {@link pointFaceOf} / {@link pointReceiver} math it
 *  routes through. Splice **after** {@link casterWgsl} + the group-1 declarations. */
export function pointShadowWgsl(): string {
    // force the base chunks first, so `PointLightGpu` + the caster structs land in the chunks that own
    // them whatever order a consumer asks in (they are spliced ahead of this one either way)
    pointLightsWgsl();
    casterWgsl();
    return pointShadowChunk();
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
 * knobs, `enabled` (0 = no caster, the fully-lit fallback), and `texel`, one atlas pixel in uv).
 */
export const SunShadow = d.struct({
    cascades: d.arrayOf(Cascade, MAX_CASCADES),
    count: d.f32,
    overlap: d.f32,
    depthBias: d.f32,
    enabled: d.f32,
    normalBias: d.f32,
    texel: d.f32,
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
        texel: at(SunShadow, (s) => s.texel),
    },
} as const;

/** the WGSL {@link SunShadow} + {@link Cascade} structs, relocatable so a screen-space consumer declares
 *  the same binding sear's color FS reads. Splice **before** that declaration, and
 *  {@link sunShadowWgsl} after it. */
export const sunStructWgsl = chunk("sunStructWgsl", [SunShadow], spliceNs);

const sampleCascade = tgpu
    .fn(
        [d.u32, d.vec3f, d.vec3f],
        d.f32,
    )(/* wgsl */ `(ci: u32, worldPos: vec3f, normal: vec3f) -> f32 {
    let c = sunShadow.cascades[ci];
    // normal-offset bias (the primary acne fix, matching Bevy): shift the receiver along its world normal by
    // normalBias shadow texels of world size before projecting. 1.41 is SQRT_2 (worst-case diagonal); the
    // texel world size is per-cascade, so a near cascade's finer texels don't over-offset
    let offset = worldPos + normalize(normal) * (sunShadow.normalBias * 1.4142136 * c.texelWorld);
    let lc = c.lightViewProj * vec4<f32>(offset, 1.0);
    let l = lc.xyz / lc.w;
    if (l.x < -1.0 || l.x > 1.0 || l.y < -1.0 || l.y > 1.0 || l.z < 0.0 || l.z > 1.0) {
        return 1.0; // outside this cascade box — lit
    }
    // remap the cascade-NDC into its atlas tile, then clamp the 3×3 PCF taps to the tile interior so a
    // grazing sample never bleeds into a neighbour cascade's tile (the point atlas's seam-clamp)
    let uv = c.rect.xy + vec2<f32>(l.x * 0.5 + 0.5, 0.5 - l.y * 0.5) * c.rect.zw;
    // a small residual constant lift toward the light (reverse-Z: the light is at greater depth, so it adds)
    let receiver = l.z + sunShadow.depthBias;
    let lo = c.rect.xy + vec2<f32>(0.5 * sunShadow.texel);
    let hi = c.rect.xy + c.rect.zw - vec2<f32>(0.5 * sunShadow.texel);
    var sum = 0.0;
    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let o = vec2<f32>(f32(ox), f32(oy)) * sunShadow.texel;
            sum = sum + textureSampleCompareLevel(shadowMap, shadowSamp, clamp(uv + o, lo, hi), receiver);
        }
    }
    return sum / 9.0;
}`)
    .$name("sampleCascade");

/**
 * the camera-forward distance from the eye (forward = `-cross(right, up)`) — the linear view-z the CSM
 * receiver selects a cascade by, Bevy's `get_cascade_index` axis. Pure, so the cascade fit's CPU half and
 * the receiver agree on the axis by construction.
 *
 * @example let viewZ = viewDepth(view.right.xyz, view.up.xyz, view.eye.xyz, worldPos);
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
        [d.vec3f, d.vec3f],
        d.f32,
    )(/* wgsl */ `(worldPos: vec3f, normal: vec3f) -> f32 {
    if (sunShadow.enabled == 0.0) { return 1.0; }
    let count = u32(sunShadow.count);
    let viewZ = viewDepth(view.right.xyz, view.up.xyz, view.eye.xyz, worldPos);
    // the first cascade whose far-bound the fragment is within
    var ci = count;
    for (var i = 0u; i < count; i = i + 1u) {
        if (viewZ < sunShadow.cascades[i].far) { ci = i; break; }
    }
    if (ci >= count) { return 1.0; } // beyond the last cascade — lit
    var shadow = sampleCascade(ci, worldPos, normal);
    // blend into the next cascade across the overlap band ((1−overlap)·far … far) so the boundary has no seam
    let next = ci + 1u;
    if (next < count) {
        let thisFar = sunShadow.cascades[ci].far;
        let nextNear = (1.0 - sunShadow.overlap) * thisFar;
        if (viewZ >= nextNear) {
            let t = clamp((viewZ - nextNear) / max(thisFar - nextNear, 1e-5), 0.0, 1.0);
            shadow = mix(shadow, sampleCascade(next, worldPos, normal), t);
        }
    }
    return shadow;
}`)
    // its function dependencies (portable across any consumer); `sunShadow` stays a free name — a
    // real-reference caller forces it into scope itself, same as the point-shadow path above
    .$uses({ viewDepth, sampleCascade })
    .$name("sampleSunShadow");

const sunShadowChunk = chunk(
    "sunShadowWgsl",
    [viewDepth, sampleCascade, sampleSunShadow],
    spliceNs,
);

/** returns the sun-shadow sampler WGSL: `sampleSunShadow(worldPos, normal)` selects a cascade by linear
 *  view-z, PCF-samples its atlas tile, and blends across the overlap band; the `enabled: 0` fallback
 *  returns fully lit. `shadowMap` / `shadowSamp` / `sunShadow` / `view` are referenced by name. Splice
 *  **after** {@link sunStructWgsl} + the group-1 declarations. */
export function sunShadowWgsl(): string {
    sunStructWgsl();
    return sunShadowChunk();
}
