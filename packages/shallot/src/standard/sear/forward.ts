// Sear — the one kitchen renderer. A GPU-driven raster *forward* pass (Aaltonen-Haar / niagara
// submission spine, primary visibility only) with sun shadows sampled inline in the FS, matching Bevy's
// clustered-forward shape. One renderer, one plugin (`SearPlugin`), no layers behind seams: one color
// pass (opaque draws then `blend` draws composited over them in a single `beginRenderPass`), an
// opt-in single-sample **prepass** emitting per-camera lanes (the `Tag` / `Depth` markers, Bevy's
// `DepthPrepass` / `NormalPrepass` shape), and sun shadows (the `Shadow` component on a directional
// light) are all sear-internal features, gated by data the way Bevy gates a shadow map on light data —
// not composed plugins coordinating through a singleton.
//
// Sun shadows: the CPU/ECS half (the off-screen light camera + placement) lives in ./shadows; this file
// owns the GPU half (the shadow map, its render through sear's own depth pipelines, and the group-1
// binding the FS samples). Sear renders its own map and reads its own `_sun` state directly — nothing
// publishes into it. Add a `Shadow` to the sun to cast; omit it for the fully-lit bare path (no map
// allocated), exactly like a camera without a lane marker runs no prepass.

import type { Plugin, State, System } from "../../engine";
import {
    Compute,
    capacity,
    f16x4,
    laneAlias,
    Registry,
    sparse,
    u32,
    unpackColor,
} from "../../engine";
import {
    LDR_COLOR_UNPACK_WGSL,
    OCT_ENCODE_WGSL,
    POS_QUANT_WGSL,
    XFORM_WGSL,
} from "../../engine/utils/core";
import { GlazeSystem } from "../glaze";
import { Camera, RenderPlugin } from "../render";
import type { Binding, Draw, Surface, View } from "../render/core";
import {
    BeginFrameSystem,
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Draws,
    FRAME_STRUCT_WGSL,
    Frame,
    LIGHTING_STRUCT_WGSL,
    LightCull,
    Lighting,
    Meshes,
    POINT_LIGHTS_STRUCT_WGSL,
    Render,
    Surfaces,
    VIEW_BYTES,
    VIEW_STRIDE,
    VIEW_STRUCT_WGSL,
    Views,
} from "../render/core";
import { SlabPlugin, slab } from "../slab";
import {
    COMBO_SHIFT,
    createRegather,
    EID_MASK,
    prepareRegather,
    SHADOW_ARG_STRIDE,
} from "./regather";
import {
    cascadeAtlasSize,
    cascadeComboEids,
    cascadeCount,
    cascadeCovers,
    cascadeFaceVP,
    cascadeFars,
    cascadeMeta,
    cascadeRecvVP,
    cascadeTileRects,
    destroyCascades,
    destroyPointShadows,
    EDGE_TEXELS,
    MAX_CASCADES,
    POINT_FACE_WGSL,
    POINT_RECEIVER_WGSL,
    type PointShadowFrame,
    pointAtlasSize,
    pointCasters,
    pointComboCount,
    pointComboEids,
    pointComboMeta,
    pointFaceVP,
    pointTileRects,
    resetCascades,
    resetPointShadows,
    SHADOW_DEFAULTS,
    Shadow,
    SunShadows,
    sunBias,
    sunCascades,
    sunResolution,
    updateCascades,
    updatePointShadows,
} from "./shadows";

const VS_FS = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;

// the depth format shared by the color pass's own 4× MSAA depth, the 1× prepass depth, and the shadow
// map. depth32float is sampleable and the reverse-Z precision win needs a float buffer (an integer depth
// gains nothing from reverse-Z); one source of truth for every depth-stencil state — and the shadow map
// renders through sear's compiled prepass depth pipeline, so one format keeps them sharing it
/** the depth-stencil format for every sear depth target (color pass, prepass, shadow atlases): one format so they share the compiled prepass depth pipeline */
export const DEPTH_FORMAT: GPUTextureFormat = "depth32float";

// the geometric-AA sample count when a camera's `Camera.antialias` is on (the default); off renders
// single-sample straight into the offscreen, no resolve. Per-camera + runtime-toggleable, sample-based
// only (no post-process blur). The prepass + shadow map stay single-sample regardless — an id can't
// MSAA-resolve and a render pass can't mix counts, so they own their own 1× depth, never varying by this
const SAMPLE_COUNT = 4;

// the id lane's screen-space target, published per camera as `view.tag`. r32uint holds the front-most
// fragment's tag per pixel, filled by the single-sample prepass (not the color MRT — see PrepassSystem)
// for cameras carrying the `Tag` marker. The tag is surface-authored — a mutable fs local defaulting to
// the entity's eid for an instanced surface and TAG_NONE otherwise, which a surface's fs overrides
// (terrain → `capacity + cell`). A consumer reads `view.tag` to know which surface owns each pixel
// (hover, outline, debug)
/** the id-lane texture format (`r32uint`): an integer id can't MSAA-resolve, which forces the single-sample prepass */
export const TAG_FORMAT: GPUTextureFormat = "r32uint";

// the reserved tag sentinel: the default for a non-instanced surface that authors no tag, and the
// value the tag target clears to (the background). eids are bounded by `capacity`, so 0xffffffff
// never collides with a real one — a reader takes any other value as a literal surface tag. Exported
// so a consumer interprets the readback without re-deriving the sentinel
/** the reserved id-lane sentinel: a pixel no surface owns. A consumer decoding `view.tag` reads any other value as a literal surface tag */
export const TAG_NONE = 0xffffffff;

/**
 * marker selecting Sear as the active renderer on a Camera entity. A camera carrying it renders through
 * sear's color pass, plus the opt-in prepass lanes its {@link Tag} / {@link Depth} markers request.
 * Lives in the renderer impl with the systems that query it; the thin `sear` barrel re-exports it to the
 * game author, `sear/core` re-exports the systems to an extender.
 *
 * @example
 * ```
 * <a camera sear transform />
 * ```
 */
export const Sear = {};

/**
 * opt a Sear camera into the **id lane**: the `view.tag` target {@link PrepassSystem} fills. Unreal's
 * `CustomStencil` generalized from an 8-bit stencil to a u32 lane; Bevy's prepass carries no id (it
 * CPU-raycasts), so the id rides this single-sample pass because it's the same rasterization. A marker
 * in the spirit of Bevy's `DepthPrepass` / `NormalPrepass`: add it to enable one extra camera output;
 * omit it and the lane is absent (no target allocated). Per-camera, so a minimap opts out while the main
 * view opts in. A consumer that reads `view.tag` (hover, outline, picking) wants this on its camera; the
 * engine itself stays tag-agnostic.
 *
 * @example
 * ```
 * <a camera sear tag transform />
 * ```
 */
export const Tag = {};

/**
 * opt a Sear camera into the **depth lane**: the prepass *stores* its single-sample depth and publishes
 * it as `view.depth` (without this marker the prepass depth is discarded, never reaching main memory).
 * Requestable on its own (a depth-only consumer needs no id) or alongside {@link Tag} (one prepass writes
 * both). Bevy's `DepthPrepass`. A screen-space consumer (AO, fog, volumetrics) adds it to read
 * `view.depth`.
 *
 * @example
 * ```
 * <a camera sear depth transform />
 * ```
 */
export const Depth = {};

/**
 * per-entity PBR material the `default` / `vertex` surfaces read (alongside `Color`, the base albedo).
 * One slab `Quad` published as `"material"`, lanes `(metallic, roughness, emissive, occlusion)`:
 * `metallic` and `roughness` are the metallic-roughness knobs ([0,1]); `emissive` is a glow **strength**
 * tinting the base color (`Color.rgb * emissive`); `occlusion` dims ambient ([0,1]). Defaults are flat
 * (metallic 0, roughness 1, emissive 0, occlusion 1), so a Part without it shades exactly like the
 * pre-PBR diffuse `lit`. Independent (non-tinted) emissive + texture-driven maps are the glTF importer's
 * job (it drives sear's `litPbr` from its own per-material palette).
 *
 * @example
 * ```
 * <a part material="metallic: 1; roughness: 0.2" transform />
 * ```
 */
export const Material = {
    /** the four PBR lanes `(metallic, roughness, emissive, occlusion)`, authored named via the `material` attribute (`material="metallic: 1; roughness: 0.2"`). */
    params: slab(f16x4, "material"),
};

const MATERIAL_FLAT: [number, number, number, number] = [0, 1, 0, 1];

const MaterialTraits = {
    defaults: () => ({ params: MATERIAL_FLAT }),
    aliases: { params: laneAlias("params", ["metallic", "roughness", "emissive", "occlusion"]) },
};

// base every slot to the flat material so a Part lacking the Material component shades like the pre-PBR
// diffuse default (an entity with Material overwrites its slot via the trait default on add). Mirrors
// Part.initPart's magenta Color base; the pack gates each slot on membership, so stale slots never draw.
function initMaterial(): void {
    for (let i = 0; i < capacity; i++) Material.params.set(i, ...MATERIAL_FLAT);
}

/**
 * a registered background: a renderer-agnostic *view-ray → HDR color* recipe sear draws as a fullscreen
 * backdrop on the un-rendered pixels (the standard infinite-skybox technique). `fs` is a WGSL chunk that
 * writes the HDR color into `col: vec3<f32>` from a normalized world-space view ray `dir` (sear
 * reconstructs it per-pixel from `view.invViewProj`), with read access to `view`, `lighting`, `frame`, and
 * any declared `bindings`. `preamble` is an optional module-scope WGSL chunk (helpers / structs /
 * constants the `fs` calls). Modeled on {@link Surface}, but backdrop-only: no mesh, instancing,
 * interpolators, or blend modes; the engine names no sky concept, a plugin owns its own sky math.
 */
export interface Background {
    name: string;
    bindings?: Record<string, Binding>;
    preamble?: string;
    fs: string;
}

/** every registered background, keyed by name with a stable numeric ID; cleared on `SearPlugin.initialize` */
export const Backgrounds: Registry<Background> = new Registry<Background>();

/**
 * select a Sear camera's backdrop: the {@link Backgrounds} recipe drawn behind the scene as a fullscreen
 * view-ray → color fill on the un-rendered pixels. Without it the camera shows the flat `Camera.clearColor`
 * (the opt-in fallback). The recipe is registered in code (`Backgrounds.register`); this picks one per
 * camera by name.
 *
 * @example
 * ```
 * <a camera sear backdrop="name: gradient" transform />
 * ```
 */
export const Backdrop = {
    /** the registered background drawn behind the scene (selected by name) */
    name: sparse(u32),
};

// name ↔ Backgrounds-id at scene parse / format, the PartTraits surface pattern (id stored, name authored)
const BackdropTraits = {
    parse: { name: (value: string) => Backgrounds.id(value) },
    format: { name: (value: number) => Backgrounds.name(value) },
};

// ---- prepass lanes: opt-in screen-space outputs, a closed engine-owned union (not a consumer registry).
// Each lane is gated by a camera marker (Bevy's DepthPrepass / NormalPrepass shape). Two ship: the `depth`
// lane (the depth-stencil itself, marker `Depth`, published as `view.depth`) and the id lane (a color
// attachment, marker `Tag`, published as `view.tag`). normal / motion are the future rows — adding one is
// a COLOR_LANES entry + a `View.*` field, and the subset codegen in `surfaceCode` + the prepass already
// iterate it; never a new pass. `depth` isn't a color attachment (it's the depth-stencil), so it's stored
// or discarded by the `Depth` marker, separate from COLOR_LANES (the color attachments the prepass MRTs)
interface ColorLane {
    // the `view.*` field + lane identity ("tag"); `set` publishes the rendered texture onto that field
    name: string;
    marker: object;
    format: GPUTextureFormat;
    usage: number;
    clear: GPUColor;
    // the mutable fs local a surface authors (symmetric with `col`), its WGSL type, and its per-surface
    // default (the fs chunk may override it)
    local: string;
    type: string;
    init(instanced: boolean): string;
    set(view: View, texture: GPUTexture): void;
}

// the id lane: the front-most opaque / `clip` surface's tag per pixel. r32uint (an integer id can't
// MSAA-resolve — what forces the prepass single-sample), COPY_SRC for a hover readback + TEXTURE_BINDING
// for an outline sample, cleared to TAG_NONE (the "no surface" background). The tag local defaults to the
// instance's eid (instanced) or TAG_NONE (a world-space producer like terrain authors its own)
const COLOR_LANES: ColorLane[] = [
    {
        name: "tag",
        marker: Tag,
        format: TAG_FORMAT,
        usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
        clear: { r: TAG_NONE, g: 0, b: 0, a: 0 },
        local: "tag",
        type: "u32",
        init: (instanced) => (instanced ? "eid" : `${TAG_NONE}u`),
        set: (view, texture) => {
            view.tag = texture;
        },
    },
];

// every subset of the color lanes (the requestable MRT lane-sets — each lane has its own marker, so any
// combination is valid), smallest first: [] then [tag]. The empty set is the position-only depth prepass
// (the shadow map reuses it); [tag] is the id lane. `surfaceCode` emits one prepass fragment per subset;
// `prepareSear` compiles one pipeline per subset — bounded, since the lane set is engine-closed
function laneSubsets(): ColorLane[][] {
    return COLOR_LANES.reduce<ColorLane[][]>(
        (acc, lane) => acc.concat(acc.map((s) => [...s, lane])),
        [[]],
    );
}

// a lane-set's stable key (the prepass pipeline-map key): "" for the empty depth-only set, "tag" for the
// id lane, "tag-normal" when normal lands
function laneKey(lanes: ColorLane[]): string {
    return lanes.map((l) => l.name).join("-");
}

// a lane-set's WGSL fragment entry point: fsPrepass (empty depth-only set), fsPrepassTag (id lane)
function prepassEntry(lanes: ColorLane[]): string {
    return `fsPrepass${lanes.map((l) => l.name[0].toUpperCase() + l.name.slice(1)).join("")}`;
}

// frame/view/lighting + vertex pull + the clustered point-light bindings (0..7), then the
// surface's own bindings
const FRAME = 0;
const VIEW = 1;
const LIGHTING = 2;
// slot 3 is the vertex stream: the color pipelines bind the 16 B main stream (`array<vec4<u32>>`), the
// prepass/shadow pipelines bind the 8 B position-only stream (`array<vec2<u32>>`) — same slot, distinct
// buffers via the two bind groups (the layout entry is `read-only-storage` either way). `meshQuant` is the
// one net-new shared binding: it pushes the heaviest surfaces (skin / glTF-textured) to exactly the
// 10-storage-per-stage ceiling (gpu.md), no headroom — reuse a cols-buffer lane before adding another
const VERTICES = 3;
const POINT_LIGHTS = 4;
const LIGHT_GRID = 5;
const LIGHT_INDICES = 6;
const MESH_QUANT = 7;
const SURFACE_BASE = 8;

/** storage bindings every sear color pass shares (indices `VERTICES..SURFACE_BASE`): vertices, pointLights, lightGrid, lightIndices, meshQuant. A surface's own storage plus this must fit the 10-per-stage ceiling (gpu.md), so a surface has `10 − SHARED_STORAGE_COUNT` for its own. Derived from the binding indices so a sixth shared binding (which bumps `SURFACE_BASE`) updates it automatically — the gltf storage-ceiling audit (`extras/gltf/live.test.ts`) imports it. */
export const SHARED_STORAGE_COUNT = SURFACE_BASE - VERTICES;

// a background's own bindings start here — after frame/view/lighting (0/1/2). A backdrop needs none of
// the surface group's vertex-pull / light-grid bindings, so its group 0 is just the three uniforms + these
const BG_BASE = 3;

// the clustered point/spot evaluation primitives, relocatable (no view / fragCoord globals): both sear's
// color FS and the fog volumetric march splice them. `distanceAttenuation` is Bevy's getDistanceAttenuation
// (the TS oracle is `distanceAttenuation` in render/lighting.ts); `spotFactor` is Frostbite getAngleAtt over
// the cone params (params.zw); `clusterCell` maps a pixel fraction + view depth to the slot-major froxel the
// light cull binned into. Spliced after POINT_LIGHTS_STRUCT_WGSL + OCT_ENCODE_WGSL (it reads PointLightGpu +
// octDecodeNormal)
/** relocatable clustered-light WGSL (`distanceAttenuation` / `spotFactor` / `clusterCell`) so a screen-space consumer evaluates the same froxel lights sear's color FS does */
export const LIGHT_EVAL_WGSL = /* wgsl */ `
fn distanceAttenuation(distSq: f32, invRangeSq: f32, radiusSq: f32) -> f32 {
    let factor = distSq * invRangeSq;
    let smoothFactor = saturate(1.0 - factor * factor);
    return smoothFactor * smoothFactor / max(distSq, radiusSq);
}

// the spot cone's angular attenuation (Frostbite getAngleAtt). A point light's params.zw carry the cone's
// angular (scale, offset) — (0, 1) for a plain point light, so the early-out returns 1 and the multiply is
// a no-op. For a spot, cd is the cosine between the cone axis (params.y, the oct-packed forward) and the
// light→fragment direction (-L, since L points fragment→light); saturate(cd·scale + offset)² is 1 inside
// the inner cone and smoothly 0 at the outer (the spotParams oracle is the (scale, offset) twin)
fn spotFactor(light: PointLightGpu, L: vec3<f32>) -> f32 {
    if (light.params.z == 0.0) { return 1.0; }
    let axis = octDecodeNormal(bitcast<u32>(light.params.y));
    let cd = -dot(axis, L);
    let a = saturate(cd * light.params.z + light.params.w);
    return a * a;
}

// the slot-major cluster-grid index for a pixel (fx, fy in [0,1], y-down) at view depth viewZ. Tile (0,0)
// is NDC (-1,-1) — bottom-left — so the y tile flips from the top-down screen y; the z slice is log over
// [near, far]. sear passes fragCoord-derived args; the fog march passes its pixel + the per-step view depth
// (tile-xy fixed along the ray, z-slice per step)
fn clusterCell(fx: f32, fy: f32, viewZ: f32, near: f32, far: f32, slot: u32) -> u32 {
    let zs = clamp(i32(log(viewZ / near) / log(far / near) * ${CLUSTER_Z}.0), 0, ${CLUSTER_Z - 1});
    let tx = min(u32(fx * ${CLUSTER_X}.0), ${CLUSTER_X - 1}u);
    let tyTop = min(u32(fy * ${CLUSTER_Y}.0), ${CLUSTER_Y - 1}u);
    let ty = ${CLUSTER_Y - 1}u - tyTop;
    let cluster = (ty * ${CLUSTER_X}u + tx) * ${CLUSTER_Z}u + u32(zs);
    return slot * ${CLUSTER_COUNT}u + cluster;
}`;

// the lighting helpers sear exposes to surface chunks: `lightFactor(normal)`
// is ambient + sun·ndl·shadow + the point-light sum (callable in the vs for per-vertex
// shading), `lit(base, normal)` applies it to a base color (per-fragment). A surface
// shades by calling them, or writes `col` directly to stay unlit.
//
// `sunVisibility` is the sun-shadow seam: the visibility of the sun at this fragment,
// multiplied into the sun term only (ambient stays unshadowed, matching niagara's
// ndotl·shadow·sun + ambient). The fragment scaffold sets it by projecting the fragment's
// world position into the shadow map inline (`sampleSunShadow`, see SHADOW_WGSL) before
// splicing the surface chunk; it defaults to 1.0, so the vertex stage (no world fragment)
// and shadowless frames are fully lit — the no-op fallback.
//
// `fragWorld` + `fragCoord` + `pointScale` are the point-light seam, the same scaffold-filled
// shape: the color FS sets the fragment's world position + screen coords and enables the
// cluster loop; the defaults (pointScale 0) make the vs and the prepass entries a no-op. So
// per-vertex shading (lightFactor in the vs) receives NEITHER sun shadows NOR point lights;
// per-pixel lit() gets both. The loop reads only the fragment's froxel cluster's lights —
// `clusterOf` maps (fragCoord.xy, view depth) to the grid cell the light cull binned into
// (view.cluster carries near/far/perspective/slot). The falloff is Bevy's
// getDistanceAttenuation — inverse-square windowed smoothly to exactly zero at the range;
// the TS oracle is `distanceAttenuation` in render/lighting.ts (pinned by its unit tests)
const LIGHT_WGSL = /* wgsl */ `
var<private> sunVisibility: f32 = 1.0;
var<private> fragWorld: vec3<f32> = vec3<f32>(0.0);
var<private> fragCoord: vec4<f32> = vec4<f32>(0.0);
var<private> pointScale: f32 = 0.0;

// the fragment's slot-major cluster index (clusterCell, LIGHT_EVAL_WGSL). View depth recovers from the
// position builtin: perspective clip.w is the view depth (fragCoord.w = 1/clip.w); orthographic depth is
// linear in fragCoord.z
fn clusterOf() -> u32 {
    let near = view.cluster.x;
    let far = view.cluster.y;
    var viewZ = 1.0 / fragCoord.w;
    if (view.cluster.z < 0.5) { viewZ = near + fragCoord.z * (far - near); }
    return clusterCell(
        fragCoord.x / view.resolution.x, fragCoord.y / view.resolution.y, viewZ, near, far, u32(view.cluster.w));
}

// Valve half-Lambert: remap the diffuse cosine from [-1,1] to [0,1] and square it, so the gradient
// spans the whole surface and the terminator softens — the matte, non-plastic happy-path look
// (Mitton & McTaggart, "Shading in Valve's Source Engine", GDC 2004). The square (not the bare remap)
// keeps form: the remap alone flattens, squaring restores midtone contrast. Diffuse-only — the
// specular cosine stays physical, so metals + glTF dielectrics are unchanged. Deliberately not
// energy-conserving; a stylized default, not physical Lambert.
fn halfLambert(ndl: f32) -> f32 {
    let h = ndl * 0.5 + 0.5;
    return h * h;
}

fn pointFactor(normal: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    if (pointScale == 0.0) { return sum; }
    let entry = lightGrid[clusterOf()];
    for (var i = 0u; i < entry.y; i = i + 1u) {
        let light = pointLights.lights[lightIndices[entry.x + i]];
        let toLight = light.posRange.xyz - fragWorld;
        let distSq = dot(toLight, toLight);
        let radiusSq = light.params.x * light.params.x;
        let L = toLight * inverseSqrt(max(distSq, 0.0001));
        let diff = halfLambert(dot(normal, L));
        sum += light.color.rgb
            * (distanceAttenuation(distSq, light.posRange.w, radiusSq) * diff * spotFactor(light, L) * pointShadowOf(light, normal, fragWorld));
    }
    return sum;
}

fn lightFactor(normal: vec3<f32>) -> vec3<f32> {
    let L = -lighting.sunDirection.xyz;
    let sun = halfLambert(dot(normal, L));
    return lighting.ambientColor.rgb * lighting.ambientColor.a
        + lighting.sunColor.rgb * sun * sunVisibility
        + pointFactor(normal);
}

fn lit(baseColor: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return baseColor * lightFactor(normal);
}

// metallic-roughness PBR (glTF 2.0 model). \`dielectric\` is the non-metal base reflectance (F0): the
// engine default material passes 0 so a non-metal has zero specular (the flat shallot look), glTF passes
// the spec-standard 0.04. \`F0 = mix(dielectric, albedo, metallic)\`.
const PI = 3.14159265359;

struct Pbr {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    dielectric: f32,
}

fn distributionGGX(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility (G / (4·ndl·ndv)), Heitz 2014
fn visSmithGGX(ndl: f32, ndv: f32, a: f32) -> f32 {
    let a2 = a * a;
    let lv = ndl * sqrt(ndv * ndv * (1.0 - a2) + a2);
    let ll = ndv * sqrt(ndl * ndl * (1.0 - a2) + a2);
    return 0.5 / max(lv + ll, 1e-7);
}

// Schlick Fresnel with f90 derived from F0 (Frostbite, Lagarde 2014): a true zero-reflectance material
// (F0 = 0) gets f90 = 0, so its specular vanishes entirely — grazing included. F0 >= ~0.02 saturates to
// the standard f90 = 1. This is what makes \`dielectric = 0\` mean literally no specular.
fn fresnelSchlick(vdh: f32, f0: vec3<f32>) -> vec3<f32> {
    let f90 = saturate(dot(f0, vec3<f32>(50.0 / 3.0)));
    let f = pow(saturate(1.0 - vdh), 5.0);
    return f0 + (vec3<f32>(f90) - f0) * f;
}

// one light's Cook-Torrance radiance, unscaled by light color / attenuation (the caller scales). The
// trailing * PI folds physical diffuse (albedo/PI) back to shallot's no-PI light convention so the diffuse
// term matches \`lit\` exactly. The diffuse cosine is half-Lambert (the soft default); the specular keeps
// the physical clamped cosine, so it vanishes on back faces and metals/glTF dielectrics stay correct.
fn brdf(s: Pbr, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> vec3<f32> {
    let d = dot(N, L);
    let ndl = max(d, 0.0);
    let H = normalize(V + L);
    let ndv = max(dot(N, V), 1e-4);
    let ndh = max(dot(N, H), 0.0);
    let vdh = max(dot(V, H), 0.0);
    let a = max(s.roughness * s.roughness, 1e-3);
    let f0 = mix(vec3<f32>(s.dielectric), s.albedo, s.metallic);
    let F = fresnelSchlick(vdh, f0);
    let spec = distributionGGX(ndh, a) * visSmithGGX(ndl, ndv, a) * F;
    let kd = (vec3<f32>(1.0) - F) * (1.0 - s.metallic);
    return (kd * s.albedo / PI * halfLambert(d) + spec * ndl) * PI;
}

// the sphere-source BRDF for a point light: diffuse on the light CENTER (\`Lc\`, half-Lambert, identical to
// \`brdf\`), specular on Karis's representative point (Real Shading in UE4) — the closest point on the source
// sphere to the mirror reflection ray. A point source gives a pinpoint highlight that a rough surface
// barely catches; a sphere of radius \`radius\` (in \`params.x\`) gives a soft round highlight scaled to the
// source size. The roughness is widened to \`aPrime\` by the solid angle the sphere subtends and the peak is
// renormalized by (a/aPrime)² so total specular energy is conserved (the highlight spreads, doesn't brighten).
// At radius 0 the representative point is \`Lc\`, \`aPrime = a\`, norm = 1, so this reduces to \`brdf(s,N,V,Lc)\` exactly.
fn brdfSphere(s: Pbr, N: vec3<f32>, V: vec3<f32>, Lc: vec3<f32>, dist: f32, radius: f32) -> vec3<f32> {
    let r = reflect(-V, N);
    let Lvec = Lc * dist; // the unnormalized light vector (Lc normalized, |Lvec| = dist)
    let centerToRay = dot(Lvec, r) * r - Lvec;
    let closest = Lvec + centerToRay * saturate(radius / max(length(centerToRay), 1e-4));
    let Ls = normalize(closest);
    let a = max(s.roughness * s.roughness, 1e-3);
    let aPrime = saturate(a + radius / (2.0 * max(dist, 1e-4)));
    let norm = (a / aPrime) * (a / aPrime);

    let dC = dot(N, Lc);
    let ndl = max(dot(N, Ls), 0.0);
    let H = normalize(V + Ls);
    let ndv = max(dot(N, V), 1e-4);
    let ndh = max(dot(N, H), 0.0);
    let vdh = max(dot(V, H), 0.0);
    let f0 = mix(vec3<f32>(s.dielectric), s.albedo, s.metallic);
    let F = fresnelSchlick(vdh, f0);
    let spec = distributionGGX(ndh, aPrime) * visSmithGGX(ndl, ndv, aPrime) * F * norm;
    let kd = (vec3<f32>(1.0) - F) * (1.0 - s.metallic);
    return (kd * s.albedo / PI * halfLambert(dC) + spec * ndl) * PI;
}

// per-pixel (or per-vertex) metallic-roughness shading. Shares the sun-shadow / point-cluster seam with
// \`lightFactor\`: \`sunVisibility\` and \`pointScale\`/\`fragWorld\` are the same fs-scaffold privates, so a
// vs-side call (pointScale 0, sunVisibility 1) gets neither point lights nor sun shadows, same as the
// diffuse path. At metallic 0 / roughness 1 / dielectric 0 this reduces to \`lit(albedo, normal)\` exactly.
fn litPbr(s: Pbr, normal: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let V = normalize(view.eye.xyz - world);
    var radiance = lighting.ambientColor.rgb * lighting.ambientColor.a * s.albedo * s.occlusion;
    radiance += lighting.sunColor.rgb * sunVisibility * brdf(s, normal, V, -lighting.sunDirection.xyz);
    if (pointScale != 0.0) {
        let entry = lightGrid[clusterOf()];
        for (var i = 0u; i < entry.y; i = i + 1u) {
            let light = pointLights.lights[lightIndices[entry.x + i]];
            let toLight = light.posRange.xyz - fragWorld;
            let distSq = dot(toLight, toLight);
            let radiusSq = light.params.x * light.params.x;
            let dist = sqrt(max(distSq, 1e-8));
            let L = toLight / dist;
            let f = distanceAttenuation(distSq, light.posRange.w, radiusSq) * spotFactor(light, L) * pointShadowOf(light, normal, fragWorld);
            radiance += light.color.rgb * f * brdfSphere(s, normal, V, L, dist, light.params.x);
        }
    }
    return radiance;
}
`;

// the shared group-0 bindings. Slot 3 is pass-specific: the color module reads the 16 B main stream
// (`vertices`, `vec4<u32>` — pos + meshId / oct normal / uv), the prepass + shadow module reads the
// 8 B position-only stream (`position`, `vec2<u32>` — pos + meshId), bound at the same slot by the two
// per-draw bind groups. `meshQuant` (MESH_QUANT_WGSL) is spliced after POS_QUANT_WGSL defines MeshQuant
const uniformWgsl = (pass: "color" | "prepass") => /* wgsl */ `
@group(0) @binding(${FRAME}) var<uniform> frame: Frame;
@group(0) @binding(${VIEW}) var<uniform> view: View;
@group(0) @binding(${LIGHTING}) var<uniform> lighting: Lighting;
${
    pass === "color"
        ? `@group(0) @binding(${VERTICES}) var<storage, read> vertices: array<vec4<u32>>;`
        : `@group(0) @binding(${VERTICES}) var<storage, read> position: array<vec2<u32>>;`
}
@group(0) @binding(${POINT_LIGHTS}) var<storage, read> pointLights: PointLights;
@group(0) @binding(${LIGHT_GRID}) var<storage, read> lightGrid: array<vec2<u32>>;
@group(0) @binding(${LIGHT_INDICES}) var<storage, read> lightIndices: array<u32>;`;

// the per-mesh dequant table — spliced after POS_QUANT_WGSL (it references the MeshQuant struct it defines)
const MESH_QUANT_WGSL = /* wgsl */ `@group(0) @binding(${MESH_QUANT}) var<storage, read> meshQuant: array<MeshQuant>;`;

// the byte size of the SunShadow params uniform: MAX_CASCADES Cascade structs (96 B each — mat4 lightViewProj
// + rect + far + texelWorld + 2 pad) + the globals tail (count / overlap / depthBias / enabled / normalBias /
// texel + 2 pad, 32 B). Sear owns the layout, the fallback, and the group-1 bindings; ./shadows owns the
// cascade cameras that drive the values sear writes here each shadowed frame. Exported so a relocatable
// consumer (the fog march) sizes its sun-shadow uniform binding to match
/** byte size of the sun-shadow params uniform: a relocatable consumer (the fog march) sizes its sun-shadow binding to match */
export const SHADOW_PARAMS_BYTES = MAX_CASCADES * 96 + 32;
// f32 strides into the params staging: one Cascade is 24 floats (mat4 16 + rect 4 + far 1 + texelWorld 1 + 2
// pad), the globals tail starts after the cascade array
const CASCADE_FLOATS = 24;
const SUN_GLOBALS_OFFSET = MAX_CASCADES * CASCADE_FLOATS;

// the relocatable shadow chunks below are spliced into sear's color FS (via `shadowWgsl`) AND a
// screen-space consumer's pass (the fog volumetric march) — one source of truth, each consumer declaring
// the group-1 bindings the chunks reference by name. The color pass's opaque + transparent pipelines call
// `sampleSunShadow` / `pointShadowOf`; the tag + depth pipelines omit group 1, so their fragments never
// reference these and stay valid. The sun half (`SUN_SHADOW_STRUCT_WGSL` + `SAMPLE_SUN_SHADOW_WGSL`) is
// plain consts (no config); the point/spot half (`casterWgsl` / `pointShadowWgsl`) are `() =>` functions
// because they interpolate the `PointShadows` config (caster count + atlas size), fixed before build but
// after this module loads. `enabled: 0` / an empty caster slot is the no-cast fallback → fully lit.
// the point/spot caster uniform structs — relocatable, spliced before the consumer's `pointShadows`
// binding decl (sear's color group 1, the fog march's group 1). `pointCasters()` is the config cap, fixed
// before build
/** returns the point/spot caster WGSL: the `PointCaster` struct + the group-1 binding decl a consumer declares to reach sear's shadow atlas */
export const casterWgsl = () => /* wgsl */ `
struct PointCaster {
    pos: vec4<f32>,
    nf: vec4<f32>,
    spotA: vec4<f32>,
    spotB: vec4<f32>,
    spotC: vec4<f32>,
}
struct PointCasters {
    casters: array<PointCaster, ${pointCasters()}>,
}
// the per-(caster, face) allocated atlas-UV rects (\`[u0, v0, du, dv]\`, square), indexed \`slot·6 + face\`:
// a point caster's six face tiles, a spot's lone tile at face 0. The receiver samples its matched caster's
// rect; the importance allocator (sear/shadows.ts) sizes + packs them into the square atlas each frame
struct TileRects {
    rects: array<vec4<f32>, ${pointCasters() * 6}>,
}`;

// the point-light shadow factor for one compacted light — relocatable: it takes the world position as a
// param and references `pointAtlas` / `shadowSamp` / `pointShadows` by name (the consumer declares them at
// its own binding, sear's color group 1 or the fog march's). Match the light to a caster slot by source
// entity id (color.a, baked by the light compact pass; pos.w is -1 for an empty slot, so a non-caster
// never matches), pick the cube face (or spot tile) from the light→fragment direction, project into the
// atlas tile, and 3×3 PCF-compare. The receiver depth reproduces the face projection's perspective depth
// analytically from the forward distance (same [near, far] = nf.xy), so the hardware depth the atlas render
// wrote compares exactly. The receiver is offset along `normal` by normalBias face texels (a volumetric
// caller with no surface normal passes vec3(0) — zero offset), plus the nf.z depth bias applied toward
// the light in linear depth (`pointReceiver`) so its world-space lift doesn't blow up with distance
/** returns the point/spot shadow WGSL: `pointShadowOf(light, normal, fragWorld)` (world pos a param, atlas/sampler/casters by name), the per-light shadow factor sear's clustered loop and a relocatable consumer both call */
export const pointShadowWgsl = () => /* wgsl */ `
${POINT_FACE_WGSL}
${POINT_RECEIVER_WGSL}

fn pointShadowOf(light: PointLightGpu, normal: vec3<f32>, fragWorld: vec3<f32>) -> f32 {
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
            let d = toFrag + normal * (c.nf.w * 1.4142136 * texelWorld);
            let z = max(dot(d, c.spotC.xyz), c.nf.x);
            receiver = pointReceiver(z, c.nf.x, c.nf.y, c.nf.z);
            let ndc = vec2<f32>(dot(d, c.spotA.xyz), dot(d, c.spotB.xyz)) / (z * coneTanHalf);
            uv = rect.xy + vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * rect.zw;
        } else {
            // point caster: all six faces share a tile size, so the widened tangent + texel come from face 0
            // (no need to know the fragment's face yet); the offset then picks the actual face
            let tilePx = tileRects.rects[k * 6u].z * atlas;
            let tanHalf = 1.0 + ${2 * EDGE_TEXELS}.0 / tilePx;
            let texelWorld = max(length(toFrag), 1e-4) * (2.0 * tanHalf / tilePx);
            let d = toFrag + normal * (c.nf.w * 1.4142136 * texelWorld);
            let f = pointFaceOf(d);
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
}`;

// the SunShadow params struct — relocatable, spliced before the consumer's group-1 sun-shadow bindings
// (sear's color group 1, the fog march's). CSM: an array of per-cascade { lightViewProj, atlas-UV `rect`,
// `far`-bound (linear view-z), per-cascade `texelWorld` } + a globals tail. Sear owns the layout, the
// fallback, and the values it writes each shadowed frame
/** the WGSL `SunShadow` uniform struct (per-cascade viewProj + atlas rect + far bound + texel size, plus a globals tail), relocatable so a screen-space consumer declares the same binding sear's color FS reads */
export const SUN_SHADOW_STRUCT_WGSL = /* wgsl */ `
struct Cascade {
    lightViewProj: mat4x4<f32>,
    rect: vec4<f32>,       // the cascade's atlas-UV tile [u0, v0, du, dv]
    far: f32,              // the cascade's far-bound in linear view-z (the receiver selects by these)
    texelWorld: f32,       // one shadow texel's world size for this cascade (the normal-offset bias scale)
    _p0: f32,
    _p1: f32,
};
struct SunShadow {
    cascades: array<Cascade, ${MAX_CASCADES}>,
    count: f32,            // active cascade count
    overlap: f32,          // inter-cascade blend-band fraction (Bevy's cascades_overlap_proportion)
    depthBias: f32,
    enabled: f32,
    normalBias: f32,
    texel: f32,            // one atlas pixel in uv (the PCF tap step) — tile-size-independent
    _p0: f32,
    _p1: f32,
};`;

// the sun (directional) shadow factor — relocatable, the sun twin of `pointShadowWgsl`: world pos + normal
// are params, and `shadowMap` (the cascade atlas) / `shadowSamp` / `sunShadow` / `view` are referenced by
// name (the consumer declares them — sear's color group 1, or the fog march's; both bind `view`). CSM
// (Bevy `get_cascade_index` + `fetch_directional_shadow`): pick the cascade by the fragment's linear view-z,
// sample its atlas tile, and blend into the next cascade across the overlap band. `enabled: 0` (the no-caster
// fallback) short-circuits to fully lit, so a volumetric march reading the fallback scatters the sun unshadowed
/** relocatable WGSL: `sampleSunShadow(worldPos, normal)` selects a cascade, PCF-samples its atlas tile, and blends across the overlap band; the `enabled: 0` fallback returns fully lit */
export const SAMPLE_SUN_SHADOW_WGSL = /* wgsl */ `
fn sampleCascade(ci: u32, worldPos: vec3<f32>, normal: vec3<f32>) -> f32 {
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
}

fn sampleSunShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> f32 {
    if (sunShadow.enabled == 0.0) { return 1.0; }
    let count = u32(sunShadow.count);
    // linear view-z: the camera-forward distance from the eye (forward = -cross(right, up)) — the cleanest
    // select space, Bevy's get_cascade_index axis
    let fwd = -cross(view.right.xyz, view.up.xyz);
    let viewZ = dot(worldPos - view.eye.xyz, fwd);
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
}`;

const shadowWgsl = () => /* wgsl */ `
${SUN_SHADOW_STRUCT_WGSL}
@group(1) @binding(0) var shadowMap: texture_depth_2d;
@group(1) @binding(1) var shadowSamp: sampler_comparison;
@group(1) @binding(2) var<uniform> sunShadow: SunShadow;
${SAMPLE_SUN_SHADOW_WGSL}

${casterWgsl()}
@group(1) @binding(3) var pointAtlas: texture_depth_2d;
@group(1) @binding(4) var<uniform> pointShadows: PointCasters;
@group(1) @binding(5) var<uniform> tileRects: TileRects;
${pointShadowWgsl()}`;

// the prepass module's shadow seam: stubs with the color module's signatures, no group-1
// declarations. The prepass entries splice the same surface chunk as the color fs, and a chunk's
// `lit()` reaches `pointFactor` → `pointShadowOf` — a real atlas read there would statically pull
// group 1 into the prepass pipelines (which bind group 0 alone) and make the atlas render's own
// `clip` discard pass sample the very texture it's writing. The stubs keep the prepass module
// group-1-free; at runtime the chunk's lighting result is discarded anyway (pointScale stays 0)
const SHADOW_STUB_WGSL = /* wgsl */ `
fn sampleSunShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> f32 { return 1.0; }
fn pointShadowOf(light: PointLightGpu, normal: vec3<f32>, fragWorld: vec3<f32>) -> f32 { return 1.0; }`;

// the standard instance transform. A surface declaring `eids` + `transforms` storage
// bindings is instanced: sear reads this instance's entity id and applies its world
// transform to position + normal, before splicing the surface's own vs chunk. The normal
// uses the inverse-transpose (`xformNormal`), correct under non-uniform scale. A surface
// without those bindings (a producer whose geometry is already world-space) stays identity.
// This is the instancing convention — nothing here is Part-specific; Part is one producer
// that publishes `eids`, and any producer publishing them gets the transform
const INSTANCE_VS = /* wgsl */ `eid = eids[iid];
    let xf = transforms[eid];
    world = vec4<f32>(xformPoint(xf, world.xyz), world.w);
    worldNormal = xformNormal(xf, worldNormal);`;

// a Binding maps to a WGSL declaration and a matching layout entry; both number
// the slot identically so they stay in lockstep
function bindingDecl(name: string, b: Binding, i: number): string {
    switch (b.type) {
        case "uniform":
            return `@group(0) @binding(${i}) var<uniform> ${name}: ${b.struct};`;
        case "storage": {
            const access = b.access === "read_write" ? "read_write" : "read";
            return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<${b.element}>;`;
        }
        case "texture-2d":
            return `@group(0) @binding(${i}) var ${name}: texture_2d<f32>;`;
        case "texture-2d-array":
            return `@group(0) @binding(${i}) var ${name}: texture_2d_array<f32>;`;
        case "texture-depth-2d":
            return `@group(0) @binding(${i}) var ${name}: texture_depth_2d;`;
        case "sampler":
            return `@group(0) @binding(${i}) var ${name}: sampler;`;
        case "sampler-comparison":
            return `@group(0) @binding(${i}) var ${name}: sampler_comparison;`;
    }
}

function bindingEntry(b: Binding, i: number): GPUBindGroupLayoutEntry {
    switch (b.type) {
        case "uniform":
            return { binding: i, visibility: VS_FS, buffer: { type: "uniform" } };
        case "storage": {
            const type = b.access === "read_write" ? "storage" : "read-only-storage";
            return { binding: i, visibility: VS_FS, buffer: { type } };
        }
        case "texture-2d":
            return { binding: i, visibility: VS_FS, texture: { sampleType: "float" } };
        case "texture-2d-array":
            return {
                binding: i,
                visibility: VS_FS,
                texture: { sampleType: "float", viewDimension: "2d-array" },
            };
        case "texture-depth-2d":
            return { binding: i, visibility: VS_FS, texture: { sampleType: "depth" } };
        case "sampler":
            return { binding: i, visibility: VS_FS, sampler: { type: "filtering" } };
        case "sampler-comparison":
            return { binding: i, visibility: VS_FS, sampler: { type: "comparison" } };
    }
}

const UNIFORM_LAYOUT: GPUBindGroupLayoutEntry[] = [
    { binding: FRAME, visibility: VS_FS, buffer: { type: "uniform" } },
    {
        binding: VIEW,
        visibility: VS_FS,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: VIEW_BYTES },
    },
    { binding: LIGHTING, visibility: VS_FS, buffer: { type: "uniform" } },
    // slot 3 (the vertex stream) — same layout entry for both bind groups; the color group binds the
    // 16 B main buffer here, the prepass group the 8 B position buffer (the element type is shader-side)
    { binding: VERTICES, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: POINT_LIGHTS, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: LIGHT_GRID, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: LIGHT_INDICES, visibility: VS_FS, buffer: { type: "read-only-storage" } },
    { binding: MESH_QUANT, visibility: VS_FS, buffer: { type: "read-only-storage" } },
];

// built-in inter-stage varyings lead the struct; custom interpolators follow, capped at 16 total. The
// set is per-surface, not fixed: `worldNormal` (a plain vec3, renormalized in the fs — oct-encoding a
// normal for interpolation breaks across the octahedral seam, and a vec2 fills a full @location slot
// anyway, so oct bought nothing here; see `builtinFields`) + `eid` + `world` always cross (the color
// scaffold's sampleSunShadow / fragWorld / tag default reach them), but `uv` + `localPos` cross only when
// the surface's fs reads them — an unread interpolator is per-fragment varying bandwidth, pruned on desktop
const MAX_INTERSTAGE = 16;

const RESERVED = new Set([
    "clip",
    "clipPos",
    "worldNormal",
    "uv",
    "localPos",
    "localNormal",
    "eid",
    "world",
    "vidx",
    "iid",
    "v",
    "out",
    "fin",
    "col",
    "tag",
    "lit",
    "lightFactor",
    "sunVisibility",
    "fragWorld",
    "fragCoord",
    "pointScale",
    "pointFactor",
    "pointLights",
    "lightGrid",
    "lightIndices",
    "clusterOf",
    "distanceAttenuation",
    "sampleSunShadow",
    "shadowMap",
    "shadowSamp",
    "sunShadow",
    "pointShadowOf",
    "pointFaceOf",
    "pointAtlas",
    "pointShadows",
    "tileRects",
    "position",
    "meshQuant",
    "octEncodeNormal",
    "octDecodeNormal",
    "decodePos",
    "decodeUv",
    "meshIdOf",
]);

// lower a surface's custom interpolators into the WGSL fragments the scaffold splices, so the vs chunk
// and the fs see the same varyings. `base` is the @location the customs start at — the builtin count,
// which varies per surface now that uv/localPos prune (see builtinFields)
function interp(record: Record<string, string>, name: string, base: number) {
    const fields = Object.entries(record);
    if (base + fields.length > MAX_INTERSTAGE) {
        throw new Error(
            `sear: surface "${name}" declares ${fields.length} interpolators; max ${MAX_INTERSTAGE - base} (inter-stage limit)`,
        );
    }
    for (const [n] of fields) {
        if (RESERVED.has(n)) {
            throw new Error(
                `sear: surface "${name}" interpolator "${n}" collides with a reserved name`,
            );
        }
    }
    return {
        // integer varyings must be flat — the rasterizer can't interpolate them
        out: fields
            .map(
                ([n, t], i) =>
                    `    @location(${base + i}) ${/\b(u32|i32)\b/.test(t) ? "@interpolate(flat) " : ""}${n}: ${t},`,
            )
            .join("\n"),
        decls: fields.map(([n, t]) => `    var ${n}: ${t};`).join("\n"),
        toOut: fields.map(([n]) => `    out.${n} = ${n};`).join("\n"),
        fromOut: fields.map(([n]) => `    let ${n} = fin.${n};`).join("\n"),
    };
}

// the built-in interstage fields a surface carries, in struct order. `worldNormal`, `eid`, and `world`
// always cross — the color scaffold (sampleSunShadow / fragWorld) and the tag default reach them regardless
// of the chunk. `uv` / `localPos` cross only when the fs reads them (the fs rebinds `fin.uv` / `fin.localPos`;
// an unread one is pure per-fragment varying waste). Returns the field list + the rebind fragments the vs
// and fs splice.
//
// The normal crosses as a **plain vec3, renormalized in the fs** — NOT oct-encoded. Oct-encoding a normal
// for *interpolation* is invalid across the octahedral seam: a triangle whose world normals straddle the
// z=0 equator interpolates between the inner diamond (z>0) and the outer corners (z<0) of the oct square,
// passing through coordinates that decode to garbage normals (the symptom: jagged normal zigzag on the faces
// of curved/draped geometry oriented to straddle the seam, e.g. one wall's banners but not the other's).
// A `vec2` oct field fills a full `@location` slot anyway, so the `vec3` costs no extra interpolator — oct
// here only bought the seam bug. (Oct stays correct for per-vertex *storage* `OCT_ENCODE_WGSL`, which decodes
// once per vertex without interpolating — the same seam hazard is why the VAT normal texture is a plain vec3.)
function builtinFields(fs: string) {
    const needsUv = /\buv\b/.test(fs);
    const needsLocal = /\blocalPos\b/.test(fs);
    const out = [`    @location(0) worldNormal: vec3<f32>,`];
    let loc = 1;
    if (needsUv) out.push(`    @location(${loc++}) uv: vec2<f32>,`);
    if (needsLocal) out.push(`    @location(${loc++}) localPos: vec3<f32>,`);
    out.push(`    @location(${loc++}) @interpolate(flat) eid: u32,`);
    out.push(`    @location(${loc++}) world: vec3<f32>,`);
    return {
        count: loc,
        struct: out.join("\n"),
        toOut: [
            `    out.worldNormal = normalize(worldNormal);`,
            ...(needsUv ? [`    out.uv = uv;`] : []),
            ...(needsLocal ? [`    out.localPos = localPos;`] : []),
            `    out.eid = eid;`,
            `    out.world = world.xyz;`,
        ].join("\n"),
        // fs rebinds the present fields as locals (the chunk + scaffold read these names). renormalize the
        // normal — linear interpolation of a unit vector across the triangle denormalizes it.
        fromOut: [
            `    let eid = fin.eid;`,
            `    let world = fin.world;`,
            `    let worldNormal = normalize(fin.worldNormal);`,
            ...(needsUv ? [`    let uv = fin.uv;`] : []),
            ...(needsLocal ? [`    let localPos = fin.localPos;`] : []),
        ].join("\n"),
    };
}

/**
 * the WGSL module for a surface: the shared frame/view/lighting + vertex pull, then the surface's bindings,
 * vs chunk, and fs chunk (which writes `col`). Two `pass` modules: `"color"` carries the real group-1 shadow
 * bindings + the color fs; `"prepass"` carries the prepass entry points with shadow **stubs** (see
 * SHADOW_STUB_WGSL: a surface chunk's `lit()` statically reaches the shadow samplers, and the prepass
 * pipelines bind group 0 alone). `variant` is the material map-set a specializing surface compiles a pipeline
 * per (the glTF importer; `surface.specialize(variant)` overrides the preamble/fs); a non-specializing surface
 * ignores it. Pure: exported for structural tests
 */
export function surfaceCode(
    surface: Surface,
    pass: "color" | "prepass" = "color",
    variant = 0,
): string {
    // a specializing surface splices the variant's preamble / fs (Bevy's on-demand `specialize`); its own
    // `preamble` / `fs` are the fallback. The glTF importer returns only a `preamble` (the map-set helpers
    // for `variant`), so the surface's own `fs` stands; a surface with no `specialize` ignores `variant`.
    const spec = surface.specialize?.(variant);
    const preamble = spec?.preamble ?? surface.preamble ?? "";
    const fsChunk = spec?.fs ?? surface.fs ?? "";
    const built = builtinFields(fsChunk);
    const i = interp(surface.interpolators ?? {}, surface.name, built.count);
    const binds = surface.bindings ?? {};
    // the instancing convention: declaring both bindings opts the surface into sear's
    // standard per-instance transform (see INSTANCE_VS)
    const instanced = !!(binds.eids && binds.transforms);
    // a screen-space surface (lines) projects its own endpoints: the vs chunk writes `clipPos` and
    // sear emits `out.clip = clipPos`. A world-space surface (the default) gets `view.viewProj * world`
    // computed after the chunk, so a chunk that displaces `world` projects correctly
    const screen = surface.screen === true;
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + SURFACE_BASE))
        .join("\n");
    // a surface that binds an f16 storage element (the `material` vec4<f16>) needs the directive, which
    // must lead the module. shader-f16 is on the platform floor, so enabling it is always valid.
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${FRAME_STRUCT_WGSL}
${VIEW_STRUCT_WGSL}
${LIGHTING_STRUCT_WGSL}
${POINT_LIGHTS_STRUCT_WGSL}
${LIGHT_EVAL_WGSL}
${LIGHT_WGSL}
${uniformWgsl(pass)}
${OCT_ENCODE_WGSL}
${POS_QUANT_WGSL}
${MESH_QUANT_WGSL}
${XFORM_WGSL}
${LDR_COLOR_UNPACK_WGSL}
${pass === "color" ? shadowWgsl() : SHADOW_STUB_WGSL}
${decls}
${preamble}
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
${built.struct}
${i.out}
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iid: u32) -> VertexOut {
${
    // color decodes the full quantized vertex (pos + oct normal + uv) from the 16 B main stream; the
    // prepass decodes only position from the 8 B stream (depth needs no normal / uv — defaulted), both
    // dequantizing against the meshId-selected MeshQuant. A surface vs chunk may still override any of them
    pass === "color"
        ? `    let v = vertices[vidx];
    let mq = meshQuant[meshIdOf(v.y)];
    var localPos = decodePos(v.x, v.y, mq);
    var localNormal = octDecodeNormal(v.z);
    var uv = decodeUv(v.w, mq);`
        : `    let v = position[vidx];
    var localPos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]);
    var localNormal = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec2<f32>(0.0);`
}
    var eid: u32 = 0u;
    var world = vec4<f32>(localPos, 1.0);
    var worldNormal = localNormal;
${screen ? "    var clipPos = vec4<f32>(0.0);" : ""}
${i.decls}
${instanced ? `    { ${INSTANCE_VS} }` : ""}
${surface.vs ? `    { ${surface.vs} }` : ""}
    var out: VertexOut;
    out.clip = ${screen ? "clipPos" : "view.viewProj * world"};
${built.toOut}
${i.toOut}
    return out;
}

${pass === "color" ? colorFragment(fsChunk, built.fromOut, i.fromOut, instanced) : ""}
${
    // alpha writes no prepass lanes (no id, no depth-write, no shadow cast); every other mode emits one
    // prepass fragment per color-lane subset (the empty subset is depth-only — a fragment only for `clip`)
    pass === "color" || surface.blend === "alpha"
        ? ""
        : laneSubsets()
              .map((s) =>
                  prepassFragment(
                      fsChunk,
                      built.fromOut,
                      i.fromOut,
                      instanced,
                      surface.blend === "clip",
                      s,
                  ),
              )
              .join("\n")
}
`;
}

/**
 * the WGSL module for a background: frame/view/lighting uniforms + the background's own bindings, a
 * fullscreen-triangle VS at the reverse-Z far plane, and an fs that reconstructs the world-space view ray
 * `dir` per-pixel then splices the chunk (which writes the HDR `col: vec3<f32>`). `dir` comes from
 * `@builtin(position)` + `view.invViewProj` (the fog reconstruct), **not** an interstage interpolator:
 * the view ray is derivable from the pixel, so crossing it would waste a varying slot (gpu.md rule 9). Pure:
 * exported for structural tests.
 */
export function backgroundCode(bg: Background): string {
    const binds = bg.bindings ?? {};
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + BG_BASE))
        .join("\n");
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${FRAME_STRUCT_WGSL}
${VIEW_STRUCT_WGSL}
${LIGHTING_STRUCT_WGSL}
@group(0) @binding(${FRAME}) var<uniform> frame: Frame;
@group(0) @binding(${VIEW}) var<uniform> view: View;
@group(0) @binding(${LIGHTING}) var<uniform> lighting: Lighting;
${decls}
${bg.preamble ?? ""}
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
}

// a fullscreen triangle (the three corners are the vertex index — no vertex pull), emitted at the
// reverse-Z far plane (clip z = 0) so the depth-equal test admits only un-rendered (background) pixels
@vertex
fn vs(@builtin(vertex_index) vidx: u32) -> VertexOut {
    let c = vec2<f32>(f32((vidx << 1u) & 2u), f32(vidx & 2u));
    var out: VertexOut;
    out.clip = vec4<f32>(c * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs(fin: VertexOut) -> @location(0) vec4<f32> {
    // reconstruct the world-space view ray from the pixel + the inverse view-projection at the far plane
    // (reverse-Z far = 0) — projection-agnostic, and derived from @builtin(position) (gpu.md rule 9)
    let uv = fin.clip.xy / view.resolution;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0);
    let far = view.invViewProj * vec4<f32>(ndc, 1.0);
    let dir = normalize(far.xyz / far.w - view.eye.xyz);
    var col = vec3<f32>(0.0);
    { ${bg.fs} }
    return vec4<f32>(col, 1.0);
}
`;
}

// the shared fragment interior every entry point splices: rebind varyings as locals, declare `col` (the
// shaded color) + every color lane's mutable local (`tag`, …), then splice the surface chunk which writes
// `col` and may override any lane local (symmetric with `col` — `tag` defaults to the instance's `eid`
// for an instanced surface and TAG_NONE otherwise, which terrain overrides to `capacity + cell`).
// `shaded` entries also sample the sun shadow inline (lit()/lightFactor multiply the sun term by it); the
// prepass entries discard without lighting and bind no shadow map, so they omit the sample. `col` is
// unused in the prepass entries, the lane locals in the color entry — the chunk is one source, spliced
// into each entry that returns a slice of its output
function fragmentBody(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    shaded: boolean,
    instanced: boolean,
): string {
    // project the fragment's world position into the shadow map and PCF-compare (1.0 when no light
    // casts — see sampleSunShadow); the sun term in lit()/lightFactor reads it
    const sun = shaded
        ? `    sunVisibility = sampleSunShadow(world, worldNormal);\n    fragWorld = world;\n    fragCoord = fin.clip;\n    pointScale = 1.0;\n`
        : "";
    const lanes = COLOR_LANES.map(
        (l) => `    var ${l.local}: ${l.type} = ${l.init(instanced)};`,
    ).join("\n");
    return `${builtinFrom}
${fromOut}
${sun}    var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);
${lanes}
${fs ? `    { ${fs} }` : ""}`;
}

// the color fragment — one single-target shape for opaque, `clip`, and `alpha` alike (they differ
// only in pipeline blend + depth policy, never in fragment output). No forced lighting: surfaces
// shade via `lit` / `lightFactor`, or write a flat `col`. The lane locals (tag, …) are NOT returned
// here — they ride the prepass (see prepassFragment / PrepassSystem), so the color pass is the
// engine's one blendable, postfx-fed lane
function colorFragment(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    instanced: boolean,
): string {
    return /* wgsl */ `@fragment
fn fs(fin: VertexOut) -> @location(0) vec4<f32> {
${fragmentBody(fs, builtinFrom, fromOut, true, instanced)}
    return col;
}`;
}

// the prepass fragment for one color-lane subset (opaque + `clip` surfaces), spliced per subset by
// surfaceCode. Runs the surface chunk for its authored lane locals (and a `clip` chunk's `discard`),
// then returns the subset's lanes. The empty subset is position-only depth: a fragment is emitted only
// for a `clip` surface (to discard) — a plain opaque surface's empty-subset pipeline has no fragment
// stage. One lane returns a scalar at @location(0); two or more an MRT output struct. Binds group 0
// only — no shadow read (the prepass carries no lighting), so the group-1 bindings stay unreferenced
function prepassFragment(
    fs: string,
    builtinFrom: string,
    fromOut: string,
    instanced: boolean,
    clip: boolean,
    lanes: ColorLane[],
): string {
    const body = fragmentBody(fs, builtinFrom, fromOut, false, instanced);
    if (lanes.length === 0) {
        if (!clip) return ""; // position-only depth — no fragment stage
        return /* wgsl */ `@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) {
${body}
}`;
    }
    if (lanes.length === 1) {
        const l = lanes[0];
        return /* wgsl */ `@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) -> @location(0) ${l.type} {
${body}
    return ${l.local};
}`;
    }
    const out = `PrepassOut${lanes.map((l) => l.name[0].toUpperCase() + l.name.slice(1)).join("")}`;
    return /* wgsl */ `struct ${out} {
${lanes.map((l, k) => `    @location(${k}) ${l.local}: ${l.type},`).join("\n")}
}
@fragment
fn ${prepassEntry(lanes)}(fin: VertexOut) -> ${out} {
${body}
    var out: ${out};
${lanes.map((l) => `    out.${l.local} = ${l.local};`).join("\n")}
    return out;
}`;
}

/**
 * the point-shadow atlas module for a surface (depth-only): one indirect draw per casting mesh, the VS
 * reading the **re-gathered** instance list: each combo (cube face / spot cone) culled independently
 * through the Part pack (its own depth-only view slot), then concatenated mesh-major into one contiguous
 * run + a per-instance combo index (`renderPointShadows`). The list is packed `(combo << COMBO_SHIFT) | eid`
 * and bound at the surface's `eids` lane (so no new storage binding past the 10-ceiling, gpu.md). The VS
 * reads each instance's (caster, face) from `comboMeta[combo]`, transforms by that combo's CPU-computed
 * viewProj (`faceVP[combo]`), and remaps clip XY into the face's atlas tile (the tile placement folded into
 * the viewProj, so the hardware does the divide + near-plane clip and the depth matches `pointShadowOf`'s
 * analytic receiver). The FS discards fragments outside the tile rect (seam bleed): depth-only, so a kept
 * fragment writes its face's depth into exactly its tile. Reuses the prepass group-0 (position-only stream +
 * the instance bindings, `eids` → the re-gathered list, SHADOW_STUB so no group-1 sampler), adds a
 * point-only group 1 (face viewProjs + combo meta + the caster rects, all uniforms). `screen` surfaces have
 * no atlas placement and `alpha` casts nothing: the caller compiles neither.
 */
function pointShadowCode(surface: Surface, variant: number, cascade = false): string {
    const spec = surface.specialize?.(variant);
    const preamble = spec?.preamble ?? surface.preamble ?? "";
    const fsChunk = spec?.fs ?? surface.fs ?? "";
    const clip = surface.blend === "clip";
    // a clip cutout runs its chunk's discard so it casts a holed shadow — cross the varyings the chunk
    // reads (matching the prepass; the position-only decode means uv is 0, the same prepass limitation).
    // An opaque surface is the tile discard alone — no chunk, no builtin varyings
    const built = clip ? builtinFields(fsChunk) : { count: 0, struct: "", toOut: "", fromOut: "" };
    const customs = surface.interpolators ?? {};
    const i = interp(customs, surface.name, built.count);
    const tileLoc = built.count + Object.keys(customs).length;
    const binds = surface.bindings ?? {};
    const instanced = !!(binds.eids && binds.transforms);
    const decls = Object.entries(binds)
        .map(([n, b], k) => bindingDecl(n, b, k + SURFACE_BASE))
        .join("\n");
    const enableF16 = Object.values(binds).some(
        (b) => "element" in b && (b.element ?? "").includes("f16"),
    );
    // point: one tile per (caster, face) — 6·casters slots, rect indexed slot·6 + face. cascade: one tile per
    // cascade — MAX_CASCADES slots, rect indexed by the cascade index (meta.x). Same VS, the count + index differ
    const slots = cascade ? MAX_CASCADES : 6 * pointCasters();
    const atlas = cascade ? cascadeAtlasSize(sunResolution(), sunCascades()) : pointAtlasSize();
    const rectExpr = cascade ? "m.x" : "m.x * 6u + m.y";
    return /* wgsl */ `${enableF16 ? "enable f16;\n" : ""}
${FRAME_STRUCT_WGSL}
${VIEW_STRUCT_WGSL}
${LIGHTING_STRUCT_WGSL}
${POINT_LIGHTS_STRUCT_WGSL}
${LIGHT_EVAL_WGSL}
${LIGHT_WGSL}
${uniformWgsl("prepass")}
${OCT_ENCODE_WGSL}
${POS_QUANT_WGSL}
${MESH_QUANT_WGSL}
${XFORM_WGSL}
${LDR_COLOR_UNPACK_WGSL}
${SHADOW_STUB_WGSL}
${decls}
${preamble}
// each combo viewProj has its atlas tile placement folded in (tileTransform), so the VS needs no manual
// divide — it reads faceVP for the clip position and tileRects (indexed by comboMeta's caster·6+face) only
// for the tile-discard bounds. The receiver samples the same tileRects on the color pass's group 1
struct FaceVPs { m: array<mat4x4<f32>, ${slots}> }
// the per-combo meta the VS reads to index its tile rect — dense, one per active combo (point: (casterSlot,
// face), 6 per caster / 1 per spot; cascade: (cascadeIndex, …), one per cascade)
struct ComboMeta { m: array<vec4<u32>, ${slots}> }
// the allocated atlas-UV rects (point: per (caster, face), slot·6 + face; cascade: per cascade index)
struct TileRects { rects: array<vec4<f32>, ${slots}> }
@group(1) @binding(0) var<uniform> faceVP: FaceVPs;
@group(1) @binding(1) var<uniform> comboMeta: ComboMeta;
@group(1) @binding(2) var<uniform> tileRects: TileRects;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
${built.struct}
${i.out}
    // the tile's pixel rect (origin.xy, size, _) for the seam discard — the tile sizes vary per caster now
    @location(${tileLoc}) @interpolate(flat) tileBox: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iid: u32) -> VertexOut {
    let v = position[vidx];
    var localPos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]);
    var localNormal = vec3<f32>(0.0, 0.0, 1.0);
    var uv = vec2<f32>(0.0);
    // the re-gathered instance list (bound at the eids slot): one entry per (combo, surviving member),
    // packed eid in the low ${COMBO_SHIFT} bits + the dense combo index above. instance_index starts at the
    // record's firstInstance (indirect-first-instance, base floor), so it indexes the mesh's run directly.
    // comboMeta maps the combo to its (caster slot, face); faceVP[combo] carries the tile-folded projection
    let packed = eids[iid];
    var eid: u32 = packed & ${EID_MASK}u;
    let combo = packed >> ${COMBO_SHIFT}u;
    let xf = transforms[eid];
    var world = vec4<f32>(xformPoint(xf, localPos), 1.0);
    var worldNormal = xformNormal(xf, localNormal);
${i.decls}
${surface.vs ? `    { ${surface.vs} }` : ""}
    let m = comboMeta.m[combo];
    let rect = tileRects.rects[${rectExpr}]; // the allocated atlas-UV rect (point: caster·6+face; cascade: index)
    var out: VertexOut;
    // the combo viewProj has its atlas tile placement folded in (tileTransform), so this projects straight
    // into the tile — the hardware does the perspective divide AND the near-plane clip. A manual fc.xy/fc.w
    // here can't clip: a triangle behind the face near plane (fc.w ≤ 0) divides to garbage and lands in-tile
    out.clip = faceVP.m[combo] * world;
    out.tileBox = vec4<f32>(rect.xy * ${atlas}.0, rect.z * ${atlas}.0, 0.0);
${built.toOut}
${i.toOut}
    return out;
}

@fragment
fn fsPoint(fin: VertexOut) {
    // tile-seam discard: a triangle outside its face frustum remaps outside the tile (but inside the
    // atlas, so the hardware doesn't clip it) — discard it so each tile gets exactly its face's depth
    let p = fin.clip.xy;
    let mn = fin.tileBox.xy;
    let sz = fin.tileBox.z;
    if (p.x < mn.x || p.x >= mn.x + sz || p.y < mn.y || p.y >= mn.y + sz) { discard; }
${clip ? fragmentBody(fsChunk, built.fromOut, i.fromOut, false, instanced) : ""}
}
`;
}

interface Compiled {
    // a surface compiles to one shape, by render mode. Opaque + `clip` cutout: `color` (the
    // single-target framebuffer) + `prepass` (one pipeline per color-lane subset — `""` is the
    // position-only depth pipeline the shadow map renders casters through, `"tag"` writes the id lane;
    // a `clip` surface's `""` runs the fragment to discard so it casts a holed shadow). `alpha`:
    // `transparent` alone — one blended color target, no prepass lanes (a transparent pixel has no
    // single owner, writes no prepass depth, casts nothing). The unused slots are null / an empty map,
    // so the shadow-map, prepass, and color passes each pick the pipeline that's theirs (the color pass
    // draws both `color` opaque and `transparent` blended, in that order, within one render pass)
    color: GPURenderPipeline | null;
    transparent: GPURenderPipeline | null;
    // the single-sample (AA-off) twin, compiled lazily by `ensureSingle` the first time a camera with
    // `Camera.antialias` off renders — null until then. An all-AA-on scene (the default) never touches it.
    // Differs from `color`/`transparent` only in `multisample.count` (1 vs SAMPLE_COUNT); `lazy` carries
    // the shared inputs to compile it
    single: { color: GPURenderPipeline | null; transparent: GPURenderPipeline | null } | null;
    singlePending: boolean;
    colorArgs: ColorArgs;
    prepass: Map<string, GPURenderPipeline>;
    // the point-shadow atlas pipeline (depth-only): one indirect draw per casting mesh, the VS reading the
    // re-gathered packed instance list (per-combo culled, concatenated mesh-major) at the eids lane and
    // remapping clip XY into each combo's atlas tile. null for `alpha` (a transparent pixel casts nothing)
    // and `screen` surfaces (2D overlays have no atlas placement)
    point: GPURenderPipeline | null;
    // the CSM cascade atlas pipeline (depth-only): the point pipeline's twin, the VS reading the cascade
    // re-gathered list and remapping clip XY into each cascade's atlas tile. Same gating as `point` (null for
    // `alpha`/`screen`/non-instanced); they differ only in the per-cascade vs per-(caster, face) tile index
    cascade: GPURenderPipeline | null;
    layout: GPUBindGroupLayout;
    slots: { name: string; type: Binding["type"] }[];
}

// straight (non-premultiplied) alpha. The swapchain is sRGB, so the blend unit linearizes the
// stored color before compositing — `src·α + dst·(1−α)` is gamma-correct with the fs writing
// linear `col`. The alpha channel keeps the framebuffer's coverage sensible for a later read
const ALPHA_BLEND: GPUBlendState = {
    color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
    alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
};

type BindResource = GPUBuffer | GPUTexture | GPUSampler;
type GroupCache = Map<
    string,
    {
        group: GPUBindGroup;
        prepassGroup: GPUBindGroup;
        // the point-shadow pass's group 0 — the prepass group with the `eids` lane swapped to the point
        // re-gathered packed instance list (`_pointRegather.eids()`); null for a non-casting surface, or
        // until the re-gather buffer is allocated (first casting frame)
        pointGroup: GPUBindGroup | null;
        // the cascade-atlas pass's group 0 — the same swap, to the cascade re-gathered list
        cascadeGroup: GPUBindGroup | null;
        resources: BindResource[];
    }
>;

// pipelines keyed `${surface}#${variant}` — one entry per (surface, material map-set) a scene actually
// draws (Bevy's on-demand specialization). A non-specializing surface is always variant 0 (compiled eagerly
// at warm); a specializing surface (the glTF importer) compiles each map-set variant lazily on first draw
const _compiled = new Map<string, Compiled>();
// the variant keys whose async compile is in flight (or permanently failed), so `record` triggers each
// compile once — the draw skips (returns null) until the pipeline lands, the same shape as an unpublished
// texture. A failed key stays in the set: a variant's WGSL is fixed for the State's life, so retrying would
// recompile + re-warn every frame; a fresh build clears the set (prepareSear)
const _compiling = new Set<string>();
const variantKey = (surface: string, variant: number) => `${surface}#${variant}`;
const _groups: GroupCache = new Map();

// a compiled background: the 4× MSAA + single-sample backdrop pipelines (one camera picks one by AA mode),
// the shared group-0 layout, and its binding slots. `group` builds lazily on first use in `renderColor`
// and holds for the State's life — the frame/view/lighting + bg-binding buffers are stable post-warm, and a
// rebuild recompiles from scratch (`_backgrounds.clear()` in prepareSear), so no in-build invalidation
type CompiledBg = {
    name: string;
    color: GPURenderPipeline;
    single: GPURenderPipeline;
    layout: GPUBindGroupLayout;
    slots: { name: string; type: Binding["type"] }[];
    group: GPUBindGroup | null;
};
const _backgrounds = new Map<string, CompiledBg>();

// the frame's draw list, resolved once by PrepassSystem (the first geometry pass) and shared across the
// prepass, shadow atlases, and color pass — they all draw the same resolved records, so resolving per-pass
// (the old 3×) was wasted work
let _frameDraws: { draw: Draw; r: Recorded }[] = [];

// ---- sun shadows: the GPU half — the CSM cascade atlas (the CPU/ECS half — cascade cameras + fit — is in
// ./shadows). The single directional map is gone: the sun renders through the cascade atlas like the point
// atlas, and the receiver selects a cascade by view-z ----

// the sun-shadow seam, sear-internal: the cascade atlas view + the per-cascade SunShadow params the color
// pass's opaque + transparent draws sample inline via group 1. Set by `renderCascades` after it renders the
// caster depth, or `null` when no sun casts (fallback → fully lit). Sear owns the atlas and reads its own
// state directly — no cross-module seam
let _sun: { map: GPUTextureView; params: GPUBuffer } | null = null;

// the no-shadow fallback bound when no light casts: a 1×1 depth texture (never sampled — `enabled: 0`
// in the all-zero params short-circuits `sampleSunShadow`) + that params buffer. Sear owns the
// comparison sampler too — one config, shared by the fallback and the real map
let _shadowSampler: GPUSampler | null = null;
let _fallbackDepth: GPUTexture | null = null;
let _fallbackView: GPUTextureView | null = null;
let _fallbackParams: GPUBuffer | null = null;

// group 1 (sun shadow) — one layout shared across surfaces: the map depth texture + the comparison
// sampler + the params uniform. The color + transparent pipelines bind it; the tag + depth pipelines
// omit it. One global bind group (one sun), cached on the bound map + params identity
let _shadowBgl: GPUBindGroupLayout | null = null;
let _shadowGroup: {
    map: GPUTextureView;
    params: GPUBuffer;
    atlas: GPUTextureView;
    group: GPUBindGroup;
} | null = null;

// the real SunShadow params sear writes each casting frame (created at warm); `_shadowReady` gates the
// render until warm has run
let _shadowReady = false;
let _sunParams: GPUBuffer | null = null;

// the SunShadow params staging: MAX_CASCADES Cascade structs (CASCADE_FLOATS each) then the globals tail
// (count / overlap / depthBias / enabled / normalBias / texel) at SUN_GLOBALS_OFFSET
const _paramsBuf = new ArrayBuffer(SHADOW_PARAMS_BYTES);
const _paramsF32 = new Float32Array(_paramsBuf);

// ---- point-light shadows: the GPU half (face viewProjs + tile math in ./shadows) ----
//
// One fixed-size depth atlas shared by every shadowed point light (cube faces as tiles, the
// PlayCanvas model), allocated lazily on the first casting frame. `_pointParams` is the PointCaster
// uniform array the FS matches compacted lights against — always bound on group 1 (an empty slot's
// pos.w = -1 never matches a real eid, so the no-caster path reads the fallback atlas never).
// Published as "pointShadows" so the gym Mirror can pin the metadata to the TS oracle.
// `_pointAtlasView` doubles as the seam: non-null once the atlas exists.
//
// The atlas renders in one pass, one indirect draw per casting mesh (the re-gather concatenates each mesh's
// per-combo culled members into one run — gpu.md "WebGPU-specific traps"). `_faceVP` is the combo-major
// face viewProj uniform the VS projects by; the re-gather state (`_shadowEids` etc) is below
let _pointAtlas: GPUTexture | null = null;
let _pointAtlasView: GPUTextureView | null = null;
let _pointParams: GPUBuffer | null = null;
// the per-(caster, face) allocated atlas-UV rects, indexed slot·6 + face — the receiver samples it (color
// group 1) and the atlas VS reads it for the tile-discard bounds (point group 1). Published "pointTileRects"
// so the gym Mirror can pin the allocation; (re)sized at warm when the PointShadows config is final
let _pointTileRects: GPUBuffer | null = null;
let _pointFrames: PointShadowFrame[] = [];
// pos + nf + spotA/B/C vec4s per caster — (re)sized at warm, when the PointShadows config is final
let _pointBuf = new ArrayBuffer(0);
let _pointF32 = new Float32Array(_pointBuf);

/** the point-shadow atlas depth view a screen-space consumer (the fog volumetric march) binds to sample
 * the casters' shadows: the real atlas once a point/spot light casts, else the 1×1 fallback (whose empty
 * caster slots never match a light, so the march reads it as fully lit). Pairs with {@link shadowSampler}
 * + the published `"pointShadows"` caster uniform. */
export function pointAtlasView(): GPUTextureView | null {
    return _pointAtlasView ?? _fallbackView;
}

/** the shared shadow comparison sampler (less-equal + linear PCF): a screen-space consumer binds it to
 * comparison-sample {@link pointAtlasView} or {@link sunShadowView}. */
export function shadowSampler(): GPUSampler | null {
    return _shadowSampler;
}

/** the sun (directional) shadow map depth view a screen-space consumer (the fog volumetric march) binds
 * to sample shadowed sun shafts: the real map once the sun casts (a `Shadow` on the directional light),
 * else the 1×1 fallback (whose `enabled: 0` params make {@link SAMPLE_SUN_SHADOW_WGSL} return 1.0, so the
 * march scatters the sun unshadowed). Pairs with {@link shadowSampler} + {@link sunShadowParams}. */
export function sunShadowView(): GPUTextureView | null {
    return _sun?.map ?? _fallbackView;
}

/** the {@link SUN_SHADOW_STRUCT_WGSL} params uniform a screen-space consumer binds: the real
 * light viewProj + bias when the sun casts, else the all-zero `enabled: 0` fallback. Pairs with
 * {@link sunShadowView}. */
export function sunShadowParams(): GPUBuffer | null {
    return _sun?.params ?? _fallbackParams;
}

// the point pipeline's group 1: the combo tile-viewProjs + the per-combo (caster, face) meta + the
// per-(caster, face) tile rects (shared with the color group, read for the VS's tile-discard bounds), all
// uniforms. The tile placement is folded into the viewProjs, so the VS's rect read is only for the seam
// discard; the per-instance (eid, combo) rides the re-gathered list at the surface's `eids` lane
let _pointBgl: GPUBindGroupLayout | null = null;
let _faceVP: GPUBuffer | null = null; // dense per-combo viewProjs (≤ 6 · casters mat4), uploaded per frame
let _comboMeta: GPUBuffer | null = null; // dense per-combo (caster slot, face) the VS tile-decodes
let _pointGroup1: {
    faceVP: GPUBuffer;
    combo: GPUBuffer;
    rects: GPUBuffer;
    group: GPUBindGroup;
} | null = null;

// the point atlas's re-gather instance: concatenates each casting mesh's per-combo culled members (the Part
// pack output) into one contiguous run + a per-instance combo index, so the atlas renders in one indirect
// draw per mesh. Its packed list (`_pointRegather.eids()`) binds at the point pass's `eids` lane. The CSM
// cascade atlas owns a second instance (`regather.ts`); both share the singleton A/B pipelines.
const _pointRegather = createRegather("point");
// the casting draws this frame (those whose surface compiled a point pipeline), filled in renderPointShadows
const _castDraws: { draw: Draw; r: Recorded }[] = [];
// per-frame re-gather meta scratch (no per-frame alloc): the view slot each dense combo culled into, and the
// (surface,mesh) pair each casting draw owns — `Regather.run` reads these. Shared across the point + cascade
// renders (each fills then consumes them in turn within ShadowMapSystem)
const _comboSlots: number[] = [];
const _drawPairs: number[] = [];

// ---- CSM cascade atlas: the GPU half (the cascade combo cameras + fit are in ./shadows) ----
//
// A dedicated depth atlas (separate from the point atlas — Bevy's directional/point split), N cascade tiles
// in a fixed grid. Each cascade is its own frustum-culled depth view (the per-cascade cull, ./shadows poses
// the cameras); the cascade `Regather` instance concatenates each casting mesh's per-cascade culled members
// into one indirect draw per mesh, the cascade pipeline's VS projecting each into its tile. Mirrors the point
// atlas exactly, the per-cascade vs per-(caster, face) tile index the only difference.
let _cascadeAtlas: GPUTexture | null = null;
let _cascadeAtlasView: GPUTextureView | null = null;
// the cascade pipeline's group 1 (same 3-uniform shape as the point group 1, reuses `_pointBgl`): the dense
// per-cascade folded tile viewProjs the VS projects by, the per-cascade meta (tile index), and the tile rects
let _cascadeVPBuf: GPUBuffer | null = null;
let _cascadeMetaBuf: GPUBuffer | null = null;
let _cascadeRectsBuf: GPUBuffer | null = null;
let _cascadeGroup1: {
    faceVP: GPUBuffer;
    combo: GPUBuffer;
    rects: GPUBuffer;
    group: GPUBindGroup;
} | null = null;
const _cascadeRegather = createRegather("cascade");
// the casting draws this frame whose surface compiled a cascade pipeline, filled in renderCascades
const _cascadeCastDraws: { draw: Draw; r: Recorded }[] = [];

// the PointCaster stride in f32: pos + nf + spotA/B/C (the spot basis), 5 vec4. The tile rects live in the
// separate "pointTileRects" uniform (6 per caster), so the FS receiver matches a light by eid then reads
// its face rect there — keeping the per-caster struct small and the rects sharable with the atlas VS
const POINT_CASTER_FLOATS = 20;

// every slot empty: pos.w = -1 (eids are non-negative, so nothing matches)
function clearPointParams(): void {
    _pointF32.fill(0);
    for (let k = 0; k < pointCasters(); k++) _pointF32[k * POINT_CASTER_FLOATS + 3] = -1;
}

function shadowGroup(): GPUBindGroup {
    const map = _sun?.map ?? _fallbackView!;
    const params = _sun?.params ?? _fallbackParams!;
    const atlas = _pointAtlasView ?? _fallbackView!;
    if (
        _shadowGroup &&
        _shadowGroup.map === map &&
        _shadowGroup.params === params &&
        _shadowGroup.atlas === atlas
    ) {
        return _shadowGroup.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-shadow",
        layout: _shadowBgl!,
        entries: [
            { binding: 0, resource: map },
            { binding: 1, resource: _shadowSampler! },
            { binding: 2, resource: { buffer: params } },
            { binding: 3, resource: atlas },
            { binding: 4, resource: { buffer: _pointParams! } },
            { binding: 5, resource: { buffer: _pointTileRects! } },
        ],
    });
    _shadowGroup = { map, params, atlas, group };
    return group;
}

/**
 * compile the forward pipelines for every registered surface, sharing one shader module: a 4× MSAA
 * single-target color pipeline (its own depth, `less` + write) that writes shaded color resolved into
 * the offscreen framebuffer, a 1× tag pipeline (its own single-sample depth, `less` + write, single
 * `r32uint` target) that stamps the front-most fragment's surface tag into `view.tag`, and a 1× depth
 * pipeline (position-only, the shadow map renders through it). Color is one camera-independent shape
 * across opaque / `clip` / `alpha`: no MRT; the tag is its own single-sample lane. Color samples the
 * sun shadow inline (group 1 = the map + comparison sampler + light params); the tag + depth pipelines
 * omit group 1. Sear declares the vertex-pull bindings itself; each draw selects its mesh via
 * `Draw.mesh`. Uniform across surfaces: no "Part-shaped" detection. Also (re)creates the sun-shadow
 * GPU resources sear owns (the comparison sampler, the 1×1 fallback, the group-1 layout, and the real
 * params buffer), surviving HMR re-warms
 */
async function prepareSear(device: GPUDevice): Promise<void> {
    _compiled.clear();
    _compiling.clear();
    _groups.clear();
    _backgrounds.clear();
    _shadowGroup = null;
    _warned.clear();
    // drop any seam a prior State left behind (module-level survives HMR)
    _sun = null;
    // the comparison sampler — `greater-equal` (reverse-Z: a lit receiver is at or in front of the
    // stored occluder, i.e. ≥ its depth) + linear filtering, so each `textureSampleCompareLevel` tap is
    // a 2×2 hardware PCF. Shared by the fallback and a real shadow map
    _shadowSampler = device.createSampler({
        label: "sear-shadow-cmp",
        compare: "greater-equal",
        magFilter: "linear",
        minFilter: "linear",
    });
    // the 1×1 fallback depth + all-zero params (enabled: 0) bound when no light casts. The map is never
    // sampled (the enabled gate short-circuits), so its undefined contents don't matter
    _fallbackDepth?.destroy();
    _fallbackDepth = device.createTexture({
        label: "sear-shadow-fallback",
        size: { width: 1, height: 1 },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _fallbackView = _fallbackDepth.createView();
    _fallbackParams?.destroy();
    _fallbackParams = device.createBuffer({
        label: "sear-shadow-fallback-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(_fallbackParams, 0, new Float32Array(SHADOW_PARAMS_BYTES / 4));
    // the real params buffer sear writes each shadowed frame (viewProj + texel + depth/normal bias)
    _sunParams?.destroy();
    _sunParams = device.createBuffer({
        label: "sear-shadow-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // the point-shadow atlas also allocates lazily; the params buffer always exists (always bound on
    // group 1, cleared to empty slots). COPY_SRC + published by name for the gym's metadata Mirror
    _pointAtlas?.destroy();
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointFrames = [];
    _pointParams?.destroy();
    _pointBuf = new ArrayBuffer(pointCasters() * POINT_CASTER_FLOATS * 4);
    _pointF32 = new Float32Array(_pointBuf);
    _pointParams = device.createBuffer({
        label: "sear-point-shadow-params",
        size: _pointBuf.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    clearPointParams();
    device.queue.writeBuffer(_pointParams, 0, _pointBuf);
    Compute.buffers.set("pointShadows", _pointParams);
    // the per-(caster, face) tile rects — bound on both the color shadow group (the receiver) and the point
    // group (the atlas VS's discard bounds). Always exists (cleared to zero), COPY_SRC + published for the
    // gym Mirror. 6 vec4 per caster
    _pointTileRects?.destroy();
    _pointTileRects = device.createBuffer({
        label: "sear-point-tilerects",
        size: pointCasters() * 6 * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(_pointTileRects, 0, new Float32Array(pointCasters() * 6 * 4));
    Compute.buffers.set("pointTileRects", _pointTileRects);
    _shadowBgl = device.createBindGroupLayout({
        label: "sear-shadow",
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
            // the sampler + point-shadow bindings are vertex-visible too: a per-vertex surface's vs
            // chunk calls lightFactor → pointFactor → pointShadowOf, so they're statically reachable
            // from the vertex stage (it early-outs on pointScale == 0 at runtime, and
            // textureSampleCompareLevel is vertex-legal — the reason the FS uses the Level variant).
            // The sun map + params (0, 2) stay fragment-only: sampleSunShadow is scaffold-called,
            // never reachable from a vs chunk. The tile rects (5) are part of pointShadowOf, so vertex-visible
            { binding: 1, visibility: VS_FS, sampler: { type: "comparison" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 3, visibility: VS_FS, texture: { sampleType: "depth" } },
            { binding: 4, visibility: VS_FS, buffer: { type: "uniform" } },
            { binding: 5, visibility: VS_FS, buffer: { type: "uniform" } },
        ],
    });

    // the point pipeline's group 1 buffers: the combo-major face viewProjs + the per-combo meta (the tile
    // rects bind alongside, all uniforms the point VS reads)
    _faceVP?.destroy();
    _faceVP = device.createBuffer({
        label: "sear-point-facevp",
        size: pointCasters() * 6 * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _comboMeta?.destroy();
    _comboMeta = device.createBuffer({
        label: "sear-point-combometa",
        size: pointCasters() * 6 * 16, // vec4<u32> per combo
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _pointBgl = device.createBindGroupLayout({
        label: "sear-point",
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        ],
    });
    _pointGroup1 = null;
    // the cascade pipeline's group 1 buffers (same shape as the point group 1, reuses `_pointBgl`): the dense
    // per-cascade folded tile viewProjs, the per-cascade meta, and the tile rects — all MAX_CASCADES-sized
    _cascadeVPBuf?.destroy();
    _cascadeVPBuf = device.createBuffer({
        label: "sear-cascade-vp",
        size: MAX_CASCADES * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeMetaBuf?.destroy();
    _cascadeMetaBuf = device.createBuffer({
        label: "sear-cascade-meta",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeRectsBuf?.destroy();
    _cascadeRectsBuf = device.createBuffer({
        label: "sear-cascade-rects",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeGroup1 = null;
    // both atlases' re-gather: (re)create the per-instance buffers + clear caches. The lazily-allocated packed
    // list binds at each atlas pipeline's `eids` lane, so allocating it clears `_groups` to rebuild the bind
    // groups with it
    _pointRegather.reset(() => _groups.clear());
    _cascadeRegather.reset(() => _groups.clear());

    _shadowReady = true;
    // eager-compile variant 0 for every non-specializing surface (the Part materials, lines, sprite, …) so
    // the bare happy path renders on the first frame, plus the shared re-gather A/B pipelines (idempotent —
    // the point + cascade atlases share them). A specializing surface (the glTF importer) defers — its draws
    // lazily compile their material map-set variant in `record`, known only once meshes load
    await Promise.all([
        prepareRegather(device),
        ...Array.from(Surfaces, (surface) =>
            surface.specialize ? null : compileVariant(device, surface, 0),
        ),
        // backdrops are registered in code at initialize (like surfaces), so the set is final at warm
        ...Array.from(Backgrounds, (bg) => compileBackground(device, bg)),
    ]);
}

// the inputs to build a surface's color/transparent pipelines at any sample count. Fixed per variant
// (only the count varies), so `compileVariant` stashes them on `Compiled` for `ensureSingle` to compile
// the single-sample twin without re-deriving the shader module (the expensive part)
type ColorArgs = {
    name: string;
    variant: number;
    module: GPUShaderModule;
    colorLayout: GPUPipelineLayout;
    primitive: GPUPrimitiveState;
    blend: Surface["blend"];
};

// build the color pass's pipelines at a given sample count — the opaque `color` (a `clip` surface is
// opaque too) or the blended `transparent` (`alpha`), whichever the surface's blend mode selects; the
// other stays null. `multisample.count` is the only thing that varies with AA mode, so the same shader
// module + pipeline layout produce both the 4× (`compileVariant`) and 1× (`ensureSingle`) twins
async function colorPipelines(
    device: GPUDevice,
    args: ColorArgs,
    samples: number,
): Promise<{ color: GPURenderPipeline | null; transparent: GPURenderPipeline | null }> {
    const { name, variant, module, colorLayout, primitive, blend } = args;
    const suffix = samples === 1 ? "-1x" : "";
    if (blend === "alpha") {
        const transparent = await device.createRenderPipelineAsync({
            label: `sear-transparent-${name}#${variant}${suffix}`,
            layout: colorLayout,
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: Render.format, blend: ALPHA_BLEND }],
            },
            primitive,
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: false,
                depthCompare: "greater-equal",
            },
            multisample: { count: samples },
        });
        return { color: null, transparent };
    }
    const color = await device.createRenderPipelineAsync({
        label: `sear-${name}#${variant}${suffix}`,
        layout: colorLayout,
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: Render.format }] },
        primitive,
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "greater" },
        multisample: { count: samples },
    });
    return { color, transparent: null };
}

/**
 * compile one background's backdrop pipelines into `_backgrounds`: the 4× MSAA + single-sample twins
 * (a camera binds whichever its `Camera.antialias` selects), sharing one shader module + group-0 layout
 * (frame / view-with-dynamic-offset / lighting + the background's own bindings at {@link BG_BASE}). The
 * pipeline draws the fullscreen triangle at the reverse-Z far plane with `depthCompare: "greater-equal"`
 * and **no depth write**: at clip z = 0 an un-rendered pixel (cleared depth 0) passes `0 >= 0`, a
 * geometry pixel (depth > 0) fails, so the backdrop fills only background pixels with no readback. Under
 * MSAA the per-sample test resolves the sky↔geometry silhouette antialiased. Both twins compile eagerly
 * (backgrounds are few; the camera's AA mode is known only at draw time).
 */
async function compileBackground(device: GPUDevice, bg: Background): Promise<void> {
    const entries = Object.entries(bg.bindings ?? {});
    const slots = entries.map(([n, b]) => ({ name: n, type: b.type }));
    const layout = device.createBindGroupLayout({
        label: `sear-bg-${bg.name}`,
        entries: [
            { binding: FRAME, visibility: VS_FS, buffer: { type: "uniform" } },
            {
                binding: VIEW,
                visibility: VS_FS,
                buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: VIEW_BYTES },
            },
            { binding: LIGHTING, visibility: VS_FS, buffer: { type: "uniform" } },
            ...entries.map(([, b], k) => bindingEntry(b, k + BG_BASE)),
        ],
    });
    const module = device.createShaderModule({
        label: `sear-bg-${bg.name}`,
        code: backgroundCode(bg),
    });
    // group 1 is the shadow seam — the background shader never references it, but the layout declares it
    // (unused) so the bg pipeline has the same two-group shape every color pipeline does. That keeps the
    // shadow group (bound once per pass) alive across the opaque → backdrop → blend pipeline switches: a
    // bg pipeline with only group 0 would be a group-count mismatch that drops group 1 for the blend draws
    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [layout, _shadowBgl!],
    });
    const pipe = (samples: number) =>
        device.createRenderPipelineAsync({
            label: `sear-bg-${bg.name}${samples === 1 ? "-1x" : ""}`,
            layout: pipelineLayout,
            vertex: { module, entryPoint: "vs" },
            fragment: { module, entryPoint: "fs", targets: [{ format: Render.format }] },
            // a fullscreen triangle has no consistent winding — disable back-face culling
            primitive: { topology: "triangle-list", cullMode: "none" },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: false,
                depthCompare: "greater-equal",
            },
            multisample: { count: samples },
        });
    const [color, single] = await Promise.all([pipe(SAMPLE_COUNT), pipe(1)]);
    _backgrounds.set(bg.name, { name: bg.name, color, single, layout, slots, group: null });
}

/**
 * compile one surface's pipelines for a material map-set `variant` (the color + transparent + per-lane-set
 * prepass pipelines + the shared bind-group layout), keyed `${surface}#${variant}` in `_compiled`. A
 * non-specializing surface only ever uses variant 0 (compiled eagerly at warm); a specializing surface (the
 * glTF importer) gets one entry per map-set a scene draws (Bevy's on-demand specialization). The bindings are
 * variant-invariant, so every variant shares the same layout shape and `record`'s one cached bind group.
 * The color pipelines compile at 4× (the AA-on default); the single-sample twin compiles lazily in `ensureSingle`.
 */
async function compileVariant(device: GPUDevice, surface: Surface, variant: number): Promise<void> {
    const name = surface.name;
    const entries = Object.entries(surface.bindings ?? {});
    const slots = entries.map(([n, b]) => ({ name: n, type: b.type }));

    const layout = device.createBindGroupLayout({
        label: `sear-${name}`,
        entries: [
            ...UNIFORM_LAYOUT,
            ...entries.map(([, b], k) => bindingEntry(b, k + SURFACE_BASE)),
        ],
    });
    // color binds group 0 (per-draw, camera-independent) + group 1 (the sun shadow map +
    // sampler + params); the prepass pipelines bind group 0 alone, since their fragments never
    // reference the group-1 shadow bindings, so the missing group 1 is valid
    const colorLayout = device.createPipelineLayout({
        bindGroupLayouts: [layout, _shadowBgl!],
    });
    const prepassLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    // two modules: the color entries reference the real group-1 shadow bindings; the
    // prepass entries compile against stubs so their group-0-only layout stays valid (and
    // the atlas render never samples the texture it's writing). `alpha` has no prepass
    // entries, so it compiles the color module alone
    const module = device.createShaderModule({
        label: `sear-${name}#${variant}`,
        code: surfaceCode(surface, "color", variant),
    });
    // a screen-space surface builds its own quads in clip space (lines), so their winding
    // flips with segment direction — back-face culling would drop half of them. World-space
    // surfaces keep back-face culling (the overdraw win + correct cutout/shadow facing)
    const primitive: GPUPrimitiveState = {
        topology: "triangle-list",
        cullMode: surface.screen ? "none" : "back",
        frontFace: "ccw",
    };
    const colorArgs: ColorArgs = {
        name,
        variant,
        module,
        colorLayout,
        primitive,
        blend: surface.blend,
    };

    // a `blend` surface is one non-opaque pipeline: a single blended color target, depth-*tested*
    // (`less-equal`) against the color pass's depth so nearer opaque geometry occludes it, but never
    // depth-*written* so it occludes nothing itself. No prepass lanes (a transparent pixel has no
    // single owner, writes no prepass depth, casts nothing) — `color` stays null, `prepass` an empty map
    if (surface.blend === "alpha") {
        const { color, transparent } = await colorPipelines(device, colorArgs, SAMPLE_COUNT);
        _compiled.set(variantKey(name, variant), {
            color,
            transparent,
            single: null,
            singlePending: false,
            colorArgs,
            prepass: new Map(),
            point: null, // a transparent pixel casts nothing
            cascade: null,
            layout,
            slots,
        });
        return;
    }

    // opaque and masked-opaque cutout (`clip`) share the color pipeline + the per-lane-set prepass
    // pipelines; they differ only in the empty-set prepass fragment stage (a `clip` surface runs
    // `fsPrepass` to discard, an opaque one is position-only)
    const clip = surface.blend === "clip";
    const prepassModule = device.createShaderModule({
        label: `sear-prepass-${name}#${variant}`,
        code: surfaceCode(surface, "prepass", variant),
    });
    // the prepass depth-stencil (reverse-Z `greater` + write, its own single-sample depth cleared each
    // frame). The color pass's depth lives inside `colorPipelines`; both are `greater` + write, never
    // cross-compared
    const depthStencil: GPUDepthStencilState = {
        format: DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "greater",
    };
    // the point-shadow atlas pipeline (depth-only): one indirect draw per casting mesh into the shared
    // atlas, the VS reading the re-gathered per-combo culled instances + remapping clip XY to the tile.
    // cullMode "back" (as the depth pass): the tile remap applies two y-flips (face-NDC → atlas-uv →
    // atlas-NDC) that cancel, so the net winding is unchanged and back-face cull drops the same faces. It
    // must — a light sitting inside a caster (a lamp fixture sphere, a light marker) sees only that mesh's
    // back faces, so culling them is what stops the fixture from occluding its own light in every
    // direction (receiver-side bias carries the acne). Only an **instanced** surface casts (the re-gathered
    // list keys on the per-instance `eids` + `transforms`); a non-instanced producer or a `screen` overlay
    // has no per-instance member list, so it gets no point pipeline
    const instanced = !!(surface.bindings?.eids && surface.bindings?.transforms);
    const castable = !surface.screen && instanced;
    const pointModule = castable
        ? device.createShaderModule({
              label: `sear-point-${name}#${variant}`,
              code: pointShadowCode(surface, variant),
          })
        : null;
    // the cascade atlas pipeline is the point pipeline's twin (same depth-only shape + group-1 layout, the
    // per-cascade tile index the only difference), so it gates + compiles the same way
    const cascadeModule = castable
        ? device.createShaderModule({
              label: `sear-cascade-${name}#${variant}`,
              code: pointShadowCode(surface, variant, true),
          })
        : null;
    const castLayout = castable
        ? device.createPipelineLayout({ bindGroupLayouts: [layout, _pointBgl!] })
        : null;

    const prepass = new Map<string, GPURenderPipeline>();
    const [{ color, transparent }, point, cascade] = await Promise.all([
        // single-target color at the AA-on sample count, resolved into the offscreen framebuffer. Owns
        // its own depth (`less` + write); the single-sample twin compiles lazily in `ensureSingle`
        colorPipelines(device, colorArgs, SAMPLE_COUNT),
        pointModule && castLayout
            ? device.createRenderPipelineAsync({
                  label: `sear-point-${name}#${variant}`,
                  layout: castLayout,
                  vertex: { module: pointModule, entryPoint: "vs" },
                  fragment: { module: pointModule, entryPoint: "fsPoint", targets: [] },
                  primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
                  depthStencil,
                  multisample: { count: 1 },
              })
            : Promise.resolve(null),
        cascadeModule && castLayout
            ? device.createRenderPipelineAsync({
                  label: `sear-cascade-${name}#${variant}`,
                  layout: castLayout,
                  vertex: { module: cascadeModule, entryPoint: "vs" },
                  fragment: { module: cascadeModule, entryPoint: "fsPoint", targets: [] },
                  primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
                  depthStencil,
                  multisample: { count: 1 },
              })
            : Promise.resolve(null),
        // one single-sample prepass pipeline per color-lane subset (`less` + write, its own depth
        // cleared each frame so the front-most fragment stamps with no prepass below it). The
        // empty subset is position-only depth (the shadow map + a depth-only camera render through
        // it; a `clip` surface adds the `fsPrepass` discard so it holes the depth and casts a
        // holed shadow); `tag` writes the id lane. Binds the group-0 layout alone — the prepass
        // fragments never reference the group-1 shadow bindings, so they carry no lighting
        ...laneSubsets().map((laneSet) =>
            device
                .createRenderPipelineAsync({
                    label: `sear-prepass-${laneKey(laneSet) || "depth"}-${name}#${variant}`,
                    layout: prepassLayout,
                    vertex: { module: prepassModule, entryPoint: "vs" },
                    ...(laneSet.length > 0 || clip
                        ? {
                              fragment: {
                                  module: prepassModule,
                                  entryPoint: prepassEntry(laneSet),
                                  targets: laneSet.map((l) => ({ format: l.format })),
                              },
                          }
                        : {}),
                    primitive,
                    depthStencil,
                })
                .then((p) => {
                    prepass.set(laneKey(laneSet), p);
                    return p;
                }),
        ),
    ]);
    _compiled.set(variantKey(name, variant), {
        color,
        transparent,
        single: null,
        singlePending: false,
        colorArgs,
        prepass,
        point,
        cascade,
        layout,
        slots,
    });
}

// trigger the lazy compile of a specializing surface's `variant` once (the draw skips until it lands).
// Deduped via `_compiling`; on success the key drops so `_compiled` is the sole record, on failure it stays
// (warn once, never retry — the WGSL is deterministic, so a recompile would just fail + spam again)
function ensureVariant(surface: Surface, variant: number): void {
    const key = variantKey(surface.name, variant);
    if (_compiled.has(key) || _compiling.has(key)) return;
    const device = Compute.device;
    if (!device || !_shadowReady) return;
    _compiling.add(key);
    compileVariant(device, surface, variant).then(
        () => _compiling.delete(key),
        (e) =>
            console.warn(`sear: surface "${surface.name}" variant ${variant} failed to compile`, e),
    );
}

// compile a variant's single-sample (AA-off) twin, once, the first frame a no-AA camera draws it (deduped
// via `singlePending`; until it lands the draw skips for that camera, the same shape as a lazy variant).
// The shader module + layout are already compiled, so this is just the 1× pipeline. An all-AA-on scene
// never calls this — `renderColor` invokes it only for a camera whose `Camera.antialias` is off
function ensureSingle(c: Compiled): void {
    if (c.single || c.singlePending) return;
    const device = Compute.device;
    if (!device) return;
    c.singlePending = true;
    colorPipelines(device, c.colorArgs, 1).then(
        (pipes) => {
            c.single = pipes;
            c.singlePending = false;
        },
        (e) => {
            c.singlePending = false;
            const msg = `sear: surface "${c.colorArgs.name}" single-sample pipeline failed to compile`;
            console.warn(msg, e);
        },
    );
}

// the point-shadow atlas, fixed-size, allocated on the first casting frame (the bare path — no
// `Shadow` on any point light — never allocates it)
function ensureAtlas(): void {
    if (_pointAtlas) return;
    const side = pointAtlasSize();
    _pointAtlas = Compute.device.createTexture({
        label: "sear-point-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _pointAtlasView = _pointAtlas.createView();
}

// the cascade atlas, fixed-size (the per-cascade resolution × the grid), allocated on the first casting frame
// — the bare path (no `Shadow` on the sun) never allocates it
function ensureCascadeAtlas(): void {
    if (_cascadeAtlas) return;
    const side = cascadeAtlasSize(sunResolution(), sunCascades());
    _cascadeAtlas = Compute.device.createTexture({
        label: "sear-cascade-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _cascadeAtlasView = _cascadeAtlas.createView();
}

// the cascade pipeline's group 1 — per-cascade tile viewProjs + meta + rects, the point group 1's twin
// (reuses `_pointBgl`). Cached on the bound identities (the buffers are stable post-warm, so this hits once)
function cascadeGroup1(): GPUBindGroup {
    if (
        _cascadeGroup1 &&
        _cascadeGroup1.faceVP === _cascadeVPBuf &&
        _cascadeGroup1.combo === _cascadeMetaBuf &&
        _cascadeGroup1.rects === _cascadeRectsBuf
    ) {
        return _cascadeGroup1.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-cascade-group1",
        layout: _pointBgl!,
        entries: [
            { binding: 0, resource: { buffer: _cascadeVPBuf! } },
            { binding: 1, resource: { buffer: _cascadeMetaBuf! } },
            { binding: 2, resource: { buffer: _cascadeRectsBuf! } },
        ],
    });
    _cascadeGroup1 = {
        faceVP: _cascadeVPBuf!,
        combo: _cascadeMetaBuf!,
        rects: _cascadeRectsBuf!,
        group,
    };
    return group;
}

// the point pipeline's group 1 — combo tile-viewProjs + combo meta + the tile rects, all uniforms (the
// per-instance (eid, combo) rides the re-gathered list at the eids lane). Cached on the bound identities
function pointGroup1(): GPUBindGroup {
    if (
        _pointGroup1 &&
        _pointGroup1.faceVP === _faceVP &&
        _pointGroup1.combo === _comboMeta &&
        _pointGroup1.rects === _pointTileRects
    ) {
        return _pointGroup1.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-point-group1",
        layout: _pointBgl!,
        entries: [
            { binding: 0, resource: { buffer: _faceVP! } },
            { binding: 1, resource: { buffer: _comboMeta! } },
            { binding: 2, resource: { buffer: _pointTileRects! } },
        ],
    });
    _pointGroup1 = { faceVP: _faceVP!, combo: _comboMeta!, rects: _pointTileRects!, group };
    return group;
}

/**
 * render every shadowed caster's depth into the atlas in **one pass, one indirect draw per casting mesh**.
 * Each combo (cube face / spot cone) culled independently through the Part pack into its own depth-only
 * view slot (the per-combo cull, `updatePointShadows` poses the cameras), then a two-pass **re-gather**
 * concatenates each casting mesh's per-combo culled members into one contiguous mesh-major run + a
 * per-instance combo index: so one indirect draw per mesh covers all its combos (the property the deleted
 * amplify trick bought, now reading per-combo *culled* counts, no over-amplification). The VS reads the
 * re-gathered packed list at the eids lane. Writes the PointCaster params the FS matches lights against,
 * uploads the CPU face viewProjs. No casters → params cleared, no pass, no atlas allocated
 */
function renderPointShadows(): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    if (_pointFrames.length === 0) {
        if (_pointF32[3] !== -1) {
            clearPointParams();
            Compute.device.queue.writeBuffer(_pointParams!, 0, _pointBuf);
        }
        return;
    }
    ensureAtlas();
    _pointRegather.ensure(pointCasters() * 6);

    // the caster params the FS samples (pos + source eid, clip planes + bias, + the spot basis —
    // right.xyz/coneTanHalf, up.xyz, fwd.xyz; coneTanHalf 0 routes the FS to the cube-face path). The tile
    // rects ride a separate uniform (uploaded below), indexed slot·6 + face
    clearPointParams();
    for (const frame of _pointFrames) {
        const o = frame.slot * POINT_CASTER_FLOATS;
        _pointF32[o] = frame.pos[0];
        _pointF32[o + 1] = frame.pos[1];
        _pointF32[o + 2] = frame.pos[2];
        _pointF32[o + 3] = frame.light;
        _pointF32[o + 4] = frame.near;
        _pointF32[o + 5] = frame.far;
        _pointF32[o + 6] = frame.depthBias;
        _pointF32[o + 7] = frame.normalBias;
        _pointF32[o + 8] = frame.right[0];
        _pointF32[o + 9] = frame.right[1];
        _pointF32[o + 10] = frame.right[2];
        _pointF32[o + 11] = frame.coneTanHalf;
        _pointF32[o + 12] = frame.up[0];
        _pointF32[o + 13] = frame.up[1];
        _pointF32[o + 14] = frame.up[2];
        _pointF32[o + 16] = frame.fwd[0];
        _pointF32[o + 17] = frame.fwd[1];
        _pointF32[o + 18] = frame.fwd[2];
    }
    Compute.device.queue.writeBuffer(_pointParams!, 0, _pointBuf);
    // the per-(caster, face) tile rects (sparse, slot·6 + face) the receiver samples + the VS discards by
    const tileRects = pointTileRects();
    Compute.device.queue.writeBuffer(
        _pointTileRects!,
        0,
        tileRects as Float32Array<ArrayBuffer>,
        0,
        tileRects.length,
    );
    // the combo viewProjs the VS projects by + their (caster, face) meta (dense, CPU-side in updatePointShadows)
    const faceVP = pointFaceVP();
    Compute.device.queue.writeBuffer(
        _faceVP!,
        0,
        faceVP as Float32Array<ArrayBuffer>,
        0,
        faceVP.length,
    );
    const comboMeta = pointComboMeta();
    Compute.device.queue.writeBuffer(
        _comboMeta!,
        0,
        comboMeta as Uint32Array<ArrayBuffer>,
        0,
        comboMeta.length,
    );

    // the casting draws (a compiled point pipeline + its point bind group) sharing the Part pack's one
    // indirect buffer — read from the Draws, not Part (sear stays part-agnostic). A producer owning its own
    // indirect buffer can't ride the shared-buffer re-gather, so it's skipped (a non-Part caster is unusual)
    _castDraws.length = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    for (const item of _frameDraws) {
        if (!item.r.c.point || !item.r.pointGroup) continue;
        const buf = item.draw.args.indirect;
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            continue;
        }
        _castDraws.push(item);
    }
    const D = _castDraws.length;
    const C = pointComboCount();
    const packed = Compute.buffers.get("eids");
    if (D === 0 || C === 0 || !drawArgs || !packed || pairCount === 0) return;

    // the re-gather inputs: the view slot each dense combo culled into, and the (surface,mesh) pair each
    // casting draw owns. `Regather.run` concatenates each mesh's per-combo culled members into one run +
    // a per-instance combo index (Pass A per-mesh args → Pass B scatter), in one compute pass
    const combos = pointComboEids();
    _comboSlots.length = 0;
    for (let c = 0; c < C; c++) _comboSlots.push(Views.get(combos[c])?.slot ?? 0);
    _drawPairs.length = 0;
    for (let i = 0; i < D; i++)
        _drawPairs.push(Math.floor((_castDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE));
    const cpass = encoder.beginComputePass({
        label: "sear:pointregather",
        timestampWrites: Compute.span?.("sear:pointregather"),
    });
    _pointRegather.run(cpass, drawArgs, packed, _comboSlots, _drawPairs, pairCount);
    cpass.end();

    // one pass into the whole atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its combo's tile. The point VS projects by faceVP (not view), so the view dynamic
    // offset is an unused slot-0 placeholder
    const pass = encoder.beginRenderPass({
        label: "sear-pointshadow",
        timestampWrites: Compute.span?.("sear:pointshadow"),
        colorAttachments: [],
        depthStencilAttachment: {
            view: _pointAtlasView!,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 0,
        },
    });
    const offset = [0];
    const group1 = pointGroup1();
    for (let i = 0; i < D; i++) {
        const { r } = _castDraws[i];
        pass.setPipeline(r.c.point!);
        pass.setBindGroup(0, r.pointGroup!, offset);
        pass.setBindGroup(1, group1);
        pass.setIndexBuffer(r.index, "uint32");
        pass.drawIndexedIndirect(_pointRegather.args()!, i * SHADOW_ARG_STRIDE);
    }
    pass.end();
    // one indirect draw per casting mesh — the Dawn indirect-validation floor (gpu.md); the per-combo
    // fan-out is collapsed by the re-gather, not amplified
    Compute.indirect?.("sear:pointshadow", D);
}

/**
 * render the CSM cascades into the dedicated cascade atlas, then publish the sun seam (`_sun` → the cascade
 * atlas + the per-cascade {@link SunShadow} params) for the color pass to sample inline: the sun's twin of
 * {@link renderPointShadows}. Each cascade is its own frustum-culled depth view (`updateCascades` poses the
 * cameras); the cascade {@link Regather} concatenates each casting mesh's per-cascade culled members into one
 * indirect draw per mesh, the cascade VS projecting each into its atlas tile. No casting sun
 * ({@link cascadeCount} 0) or no casting geometry → `_sun = null` (the fully-lit fallback), no atlas allocated.
 */
function renderCascades(): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    const C = cascadeCount();
    if (C === 0) {
        _sun = null;
        return;
    }
    ensureCascadeAtlas();
    _cascadeRegather.ensure(MAX_CASCADES);

    // upload the per-cascade folded tile viewProjs + meta + rects (CPU-computed in updateCascades)
    const vp = cascadeFaceVP();
    Compute.device.queue.writeBuffer(_cascadeVPBuf!, 0, vp as Float32Array<ArrayBuffer>, 0, C * 16);
    const meta = cascadeMeta();
    Compute.device.queue.writeBuffer(
        _cascadeMetaBuf!,
        0,
        meta as Uint32Array<ArrayBuffer>,
        0,
        C * 4,
    );
    const rects = cascadeTileRects();
    Compute.device.queue.writeBuffer(
        _cascadeRectsBuf!,
        0,
        rects as Float32Array<ArrayBuffer>,
        0,
        C * 4,
    );

    // the casting draws (a compiled cascade pipeline + its cascade bind group) sharing the Part pack's one
    // indirect buffer — same part-agnostic gather as the point path
    _cascadeCastDraws.length = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    for (const item of _frameDraws) {
        if (!item.r.c.cascade || !item.r.cascadeGroup) continue;
        const buf = item.draw.args.indirect;
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            continue;
        }
        _cascadeCastDraws.push(item);
    }
    const D = _cascadeCastDraws.length;
    const packed = Compute.buffers.get("eids");
    if (D === 0 || !drawArgs || !packed || pairCount === 0) {
        _sun = null; // a casting sun with no casting geometry — fully lit, like the no-cast path
        return;
    }

    // the re-gather inputs: the view slot each cascade culled into, the (surface,mesh) pair each casting draw owns
    const combos = cascadeComboEids();
    _comboSlots.length = 0;
    for (let c = 0; c < C; c++) _comboSlots.push(Views.get(combos[c])?.slot ?? 0);
    _drawPairs.length = 0;
    for (let i = 0; i < D; i++)
        _drawPairs.push(
            Math.floor((_cascadeCastDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE),
        );
    const cpass = encoder.beginComputePass({
        label: "sear:cascaderegather",
        timestampWrites: Compute.span?.("sear:cascaderegather"),
    });
    _cascadeRegather.run(cpass, drawArgs, packed, _comboSlots, _drawPairs, pairCount);
    cpass.end();

    // one pass into the cascade atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its cascade's tile. The cascade VS projects by the folded tile viewProj (not view), so
    // the view dynamic offset is an unused slot-0 placeholder
    const pass = encoder.beginRenderPass({
        label: "sear-cascadeshadow",
        timestampWrites: Compute.span?.("sear:cascadeshadow"),
        colorAttachments: [],
        depthStencilAttachment: {
            view: _cascadeAtlasView!,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 0,
        },
    });
    const offset = [0];
    const group1 = cascadeGroup1();
    for (let i = 0; i < D; i++) {
        const { r } = _cascadeCastDraws[i];
        pass.setPipeline(r.c.cascade!);
        pass.setBindGroup(0, r.cascadeGroup!, offset);
        pass.setBindGroup(1, group1);
        pass.setIndexBuffer(r.index, "uint32");
        pass.drawIndexedIndirect(_cascadeRegather.args()!, i * SHADOW_ARG_STRIDE);
    }
    pass.end();
    Compute.indirect?.("sear:cascadeshadow", D);

    // write the per-cascade SunShadow params + publish the seam: the receiver selects a cascade by view-z and
    // samples the cascade atlas (`sampleSunShadow`). One atlas pixel in uv (`texel`) is the PCF tap step; each
    // cascade carries its own world texel size (2·cover/resolution) for the normal-offset bias
    const recv = cascadeRecvVP();
    const tileRects = cascadeTileRects();
    const fars = cascadeFars();
    const covers = cascadeCovers();
    const res = sunResolution();
    _paramsF32.fill(0);
    for (let i = 0; i < C; i++) {
        const base = i * CASCADE_FLOATS;
        _paramsF32.set(recv.subarray(i * 16, i * 16 + 16), base);
        _paramsF32.set(tileRects.subarray(i * 4, i * 4 + 4), base + 16);
        _paramsF32[base + 20] = fars[i];
        _paramsF32[base + 21] = (2 * covers[i]) / res;
    }
    const bias = sunBias();
    _paramsF32[SUN_GLOBALS_OFFSET] = C;
    _paramsF32[SUN_GLOBALS_OFFSET + 1] = SunShadows.overlap;
    _paramsF32[SUN_GLOBALS_OFFSET + 2] = bias.depthBias;
    _paramsF32[SUN_GLOBALS_OFFSET + 3] = 1; // enabled
    _paramsF32[SUN_GLOBALS_OFFSET + 4] = bias.normalBias;
    // one atlas pixel in uv — the actual texture side (allocated for the fixed sunCascades()), not the live
    // count: an ortho main camera runs C = 1 into the whole atlas, so its PCF tap step is still 1 physical pixel
    _paramsF32[SUN_GLOBALS_OFFSET + 5] = 1 / cascadeAtlasSize(res, sunCascades());
    Compute.device.queue.writeBuffer(_sunParams!, 0, _paramsBuf, 0, SHADOW_PARAMS_BYTES);
    _sun = { map: _cascadeAtlasView!, params: _sunParams! };
}

// free every GPU resource sear owns (at plugin dispose): the shadow atlases (point + cascade) + their params,
// and the per-camera prepass depth / lane targets / MSAA color+depth. The cascade Camera entities live in a
// State, so destroyCascades (./shadows) tears those down separately
function disposeSear(): void {
    _fallbackDepth?.destroy();
    _fallbackParams?.destroy();
    _sunParams?.destroy();
    _pointAtlas?.destroy();
    _pointParams?.destroy();
    _pointTileRects?.destroy();
    _faceVP?.destroy();
    _comboMeta?.destroy();
    _pointRegather.dispose();
    _cascadeAtlas?.destroy();
    _cascadeVPBuf?.destroy();
    _cascadeMetaBuf?.destroy();
    _cascadeRectsBuf?.destroy();
    _cascadeRegather.dispose();
    _cascadeAtlas = null;
    _cascadeAtlasView = null;
    _cascadeVPBuf = null;
    _cascadeMetaBuf = null;
    _cascadeRectsBuf = null;
    _cascadeGroup1 = null;
    _faceVP = null;
    _comboMeta = null;
    _pointGroup1 = null;
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointParams = null;
    _pointTileRects = null;
    _pointFrames = [];
    _fallbackDepth = null;
    _fallbackView = null;
    _fallbackParams = null;
    _sunParams = null;
    _sun = null;
    _shadowReady = false;
    for (const c of _depth.values()) c.texture.destroy();
    for (const c of _laneTargets.values()) c.texture.destroy();
    for (const c of _colorTargets.values()) {
        c.color?.destroy();
        c.depth.destroy();
    }
    _depth.clear();
    _laneTargets.clear();
    _colorTargets.clear();
    _backgrounds.clear();
}

// a draw resolving to null is a silent skip — usually a typo'd binding or an
// unpublished resource. Warn once per draw so it's visible without spamming
const _warned = new Set<string>();

function warnSkip(draw: string, cause: string): null {
    if (!_warned.has(draw)) {
        _warned.add(draw);
        console.warn(`sear: draw "${draw}" skipped — ${cause}`);
    }
    return null;
}

const registryFor = (type: Binding["type"]) =>
    type.startsWith("texture")
        ? Compute.textures
        : type.startsWith("sampler")
          ? Compute.samplers
          : Compute.buffers;

// shape a resolved resource into a bind-group entry resource by binding type: a 2d-array texture binds a
// 2d-array view, any other texture its default view, a sampler itself, a buffer wrapped
const bindResource = (type: Binding["type"], res: BindResource): GPUBindingResource =>
    type === "texture-2d-array"
        ? (res as GPUTexture).createView({ dimension: "2d-array" })
        : type.startsWith("texture")
          ? (res as GPUTexture).createView()
          : type.startsWith("sampler")
            ? (res as GPUSampler)
            : { buffer: res as GPUBuffer };

type Recorded = {
    // the compiled surface — the color pass reads `color`/`transparent` (or the single-sample `single`
    // twin for a no-AA camera); the prepass + shadow passes read `prepass`. Carried whole so the per-camera
    // AA selection happens at draw time without baking each pipeline ref per draw
    c: Compiled;
    // the color pass binds `group` (slot 3 = the 16 B main stream); the prepass + shadow passes bind
    // `prepassGroup` (slot 3 = the 8 B position stream). Two groups, one shared group-0 layout — the
    // only per-pass difference is which vertex buffer sits at slot 3 (gpu.md 10-storage ceiling)
    group: GPUBindGroup;
    prepassGroup: GPUBindGroup;
    // the point-shadow pass's group 0 (the prepass group with `eids` → the point re-gathered list); null for
    // a non-casting surface or before the re-gather buffer is allocated
    pointGroup: GPUBindGroup | null;
    // the cascade-atlas pass's group 0 (the prepass group with `eids` → the cascade re-gathered list)
    cascadeGroup: GPUBindGroup | null;
    // the mesh's index buffer, bound via setIndexBuffer before each drawIndexedIndirect (geometry pulls
    // vertices from the storage binding, but the hardware index buffer drives vertex reuse)
    index: GPUBuffer;
};

/**
 * the color + transparent + per-lane-set prepass pipelines and the bind group sear records a draw
 * with, or null to skip it. All pipelines share one bind group (same group-0 layout). A surface with
 * no compiled pipeline isn't sear's (silent skip); a missing mesh or unpublished binding warns once.
 * Bind groups cache per draw, rebuilt only on a resource identity change; the fixed uniforms are
 * stable, so untracked
 */
function record(draw: Draw): Recorded | null {
    const surface = Surfaces.get(draw.surface);
    if (!surface) return null; // not a sear surface — silent skip
    const mesh = Meshes.get(draw.mesh);
    if (!mesh) return warnSkip(draw.name, `mesh "${draw.mesh}" not registered`);
    // a specializing surface (glTF) selects the pipeline for this mesh's material map-set; every other
    // surface is variant 0. The variant is constant per mesh, so the cached bind group (variant-invariant)
    // stays valid across frames
    const variant = surface.specialize ? (mesh.variant ?? 0) : 0;
    const c = _compiled.get(variantKey(draw.surface, variant));
    if (!c) {
        ensureVariant(surface, variant); // kick off the lazy compile; skip the draw until it lands
        return null;
    }

    // a mesh registered before the quantized format (or by an un-migrated producer) has no position /
    // quant stream — skip it loudly rather than bind a garbage decode
    if (!mesh.position || !mesh.quant)
        return warnSkip(draw.name, `mesh "${draw.mesh}" has no quantized position/quant stream`);
    const main = mesh.vertices;
    const position = mesh.position;
    const quant = mesh.quant;

    // resolve the surface bindings to live resources by type (geometry first: main, position, quant, index).
    // A per-mesh override (`mesh.bindings`, e.g. a skinned mesh's own VAT) wins over the published global, so
    // meshes needing distinct resources for the same binding name (one VAT per skinned mesh) each bind their own
    const resources: BindResource[] = [main, position, quant, mesh.indices];
    for (const { name, type } of c.slots) {
        const res = mesh.bindings?.[name] ?? registryFor(type).get(name);
        if (!res) return warnSkip(draw.name, `binding "${name}" (${type}) not published`);
        resources.push(res);
    }

    const prev = _groups.get(draw.name);
    if (
        prev &&
        prev.resources.length === resources.length &&
        prev.resources.every((b, k) => b === resources[k])
    ) {
        return {
            c,
            group: prev.group,
            prepassGroup: prev.prepassGroup,
            pointGroup: prev.pointGroup,
            cascadeGroup: prev.cascadeGroup,
            index: mesh.indices,
        };
    }

    // the bindings shared by both groups (everything but slot 3, the vertex stream)
    const shared: GPUBindGroupEntry[] = [
        { binding: FRAME, resource: { buffer: Frame.buffer } },
        { binding: VIEW, resource: { buffer: Render.viewBuffer, size: VIEW_STRIDE } },
        { binding: LIGHTING, resource: { buffer: Lighting.buffer } },
        { binding: POINT_LIGHTS, resource: { buffer: LightCull.lights! } },
        { binding: LIGHT_GRID, resource: { buffer: LightCull.grid! } },
        { binding: LIGHT_INDICES, resource: { buffer: LightCull.indices! } },
        { binding: MESH_QUANT, resource: { buffer: quant } },
    ];
    // textures bind a default view — cache-miss path only, never per-frame on a hit
    c.slots.forEach(({ type }, k) => {
        shared.push({ binding: k + SURFACE_BASE, resource: bindResource(type, resources[k + 4]) });
    });

    // two groups, one layout — slot 3 is the only difference (color: 16 B main, prepass: 8 B position)
    const makeGroup = (vertex: GPUBuffer) =>
        Compute.device.createBindGroup({
            label: `sear-${draw.name}`,
            layout: c.layout,
            entries: [{ binding: VERTICES, resource: { buffer: vertex } }, ...shared],
        });
    const group = makeGroup(main);
    const prepassGroup = makeGroup(position);
    // a shadow-atlas group 0: the prepass group with the `eids` lane bound to a re-gathered packed instance
    // list (the point atlas's or the cascade atlas's). Reusing the lane keeps the atlas pipelines at zero new
    // storage bindings, so the heaviest surfaces stay within the 10-per-stage ceiling (gpu.md). Built only for
    // a casting surface, and only once that atlas's packed list exists (its alloc clears `_groups`, rebuilding)
    const eidsK = c.slots.findIndex((s) => s.name === "eids");
    const eidsSwap = (
        pipe: GPURenderPipeline | null,
        listEids: GPUBuffer | null,
    ): GPUBindGroup | null => {
        if (!pipe || !listEids || eidsK < 0) return null;
        const entries: GPUBindGroupEntry[] = [
            { binding: VERTICES, resource: { buffer: position } },
            ...shared,
        ].map((e) =>
            e.binding === SURFACE_BASE + eidsK
                ? { binding: e.binding, resource: { buffer: listEids } }
                : e,
        );
        return Compute.device.createBindGroup({
            label: `sear-shadowcast-${draw.name}`,
            layout: c.layout,
            entries,
        });
    };
    const pointGroup = eidsSwap(c.point, _pointRegather.eids());
    const cascadeGroup = eidsSwap(c.cascade, _cascadeRegather.eids());
    _groups.set(draw.name, { group, prepassGroup, pointGroup, cascadeGroup, resources });
    return { c, group, prepassGroup, pointGroup, cascadeGroup, index: mesh.indices };
}

/**
 * the frame's draw list: every registered {@link Draw} with a compiled surface + published
 * bindings, paired with its cached bind group. View-independent (group 0 binds the whole view
 * buffer; the per-view slice is a dynamic offset applied at bind time), so {@link PrepassSystem}
 * resolves it once per frame into `_frameDraws` and the prepass, shadow map, and color pass all
 * render every camera against that one list
 */
function resolveDraws(): { draw: Draw; r: Recorded }[] {
    const items: { draw: Draw; r: Recorded }[] = [];
    for (const draw of Draws.values()) {
        const r = record(draw);
        if (r) items.push({ draw, r });
    }
    return items;
}

const _depth = new Map<
    number,
    { texture: GPUTexture; view: GPUTextureView; w: number; h: number }
>();

// the per-camera single-sample depth the prepass writes — always the front-most-fragment test the id
// lane needs, but only *stored* + published as `view.depth` when the camera carries `Depth` (else the
// store is discarded). Allocated when the prepass runs (any lane marker). TEXTURE_BINDING so a
// screen-space consumer (AO, volumetrics) can sample it the same frame
function depthView(eid: number, w: number, h: number): GPUTextureView {
    const cached = _depth.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = Compute.device.createTexture({
        label: `sear-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    _depth.set(eid, { texture, view, w, h });
    return view;
}

const _laneTargets = new Map<
    string,
    { texture: GPUTexture; view: GPUTextureView; w: number; h: number }
>();

// the per-(camera, color-lane) screen-space target, sibling to depthView — filled by the prepass.
// Sized to the framebuffer, recreated on resize; the format + usage are the lane's (the id lane is
// r32uint + COPY_SRC for a hover readback + TEXTURE_BINDING for an outline sample). Returns the texture
// (published onto `view.<lane>`) + the color-attachment view — one cache keyed `${eid}:${lane.name}`
// that drives both, so adding a lane needs no new allocator
function laneTarget(
    eid: number,
    lane: ColorLane,
    w: number,
    h: number,
): { texture: GPUTexture; view: GPUTextureView } {
    const key = `${eid}:${lane.name}`;
    const cached = _laneTargets.get(key);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.texture.destroy();
    const texture = Compute.device.createTexture({
        label: `sear-${lane.name}-${eid}`,
        size: { width: w, height: h },
        format: lane.format,
        usage: lane.usage,
    });
    const entry = { texture, view: texture.createView(), w, h };
    _laneTargets.set(key, entry);
    return entry;
}

const _colorTargets = new Map<
    number,
    {
        color: GPUTexture | null;
        colorView: GPUTextureView | null;
        depth: GPUTexture;
        depthView: GPUTextureView;
        w: number;
        h: number;
        aa: boolean;
    }
>();

// the per-camera color-pass targets, by AA mode. AA on: a 4× MSAA color (resolved into the offscreen at
// pass end) + a 4× depth. AA off: no MSAA color (the pass renders straight into view.framebuffer) + a 1×
// depth. The color pass owns this depth (`less` + write, cleared each frame); the prepass + shadow map
// keep their own 1× depth (never cross-compared). Sized to the view + keyed on AA, recreated on resize/toggle
function colorTargets(
    eid: number,
    w: number,
    h: number,
    aa: boolean,
): { color: GPUTextureView | null; depth: GPUTextureView } {
    const cached = _colorTargets.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.aa === aa)
        return { color: cached.colorView, depth: cached.depthView };
    cached?.color?.destroy();
    cached?.depth.destroy();
    const samples = aa ? SAMPLE_COUNT : 1;
    const color = aa
        ? Compute.device.createTexture({
              label: `sear-color-msaa-${eid}`,
              size: { width: w, height: h },
              format: Render.format,
              sampleCount: SAMPLE_COUNT,
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null;
    const depth = Compute.device.createTexture({
        label: `sear-color-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        sampleCount: samples,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const entry = {
        color,
        colorView: color?.createView() ?? null,
        depth,
        depthView: depth.createView(),
        w,
        h,
        aa,
    };
    _colorTargets.set(eid, entry);
    return { color: entry.colorView, depth: entry.depthView };
}

// the geometry pass (one color target — the prepass lanes ride their own pass). AA on: the opaque draws
// clear + write `msaaColor`, the transparent draws blend over, and it resolves into the offscreen once at
// pass end (`discard` — the resolve fires regardless and nothing reads the MSAA target after). AA off:
// `msaaColor` is null — render straight into the offscreen, no resolve, **`store`** the result (`discard`
// would throw away the only copy → a black frame). The depth `discard`s either way (transient)
function beginColor(
    eid: number,
    msaaColor: GPUTextureView | null,
    depth: GPUTextureView,
    framebuffer: GPUTextureView,
    clear: ReturnType<typeof unpackColor>,
) {
    const clearValue = { ...clear, a: 1 };
    return Render.encoder!.beginRenderPass({
        label: `sear-color/${eid}`,
        timestampWrites: Compute.span?.("sear:color"),
        colorAttachments: [
            msaaColor
                ? {
                      view: msaaColor,
                      resolveTarget: framebuffer,
                      loadOp: "clear",
                      storeOp: "discard",
                      clearValue,
                  }
                : { view: framebuffer, loadOp: "clear", storeOp: "store", clearValue },
        ],
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            depthStoreOp: "discard",
            depthClearValue: 0,
        },
    });
}

// set the pipeline + per-draw group 0 (with the view's dynamic offset) and issue the indirect
// draw. Shared by the prepass and the color pass's opaque + transparent draws — they differ in pipeline
// and in which group-0 bind group (the color pass binds `r.group`, the prepass `r.prepassGroup`); the
// color pass additionally binds group 1, its caller's concern, not here
function bind(
    pass: GPURenderPassEncoder,
    pipeline: GPURenderPipeline,
    draw: Draw,
    r: Recorded,
    group: GPUBindGroup,
    offset: number[],
    slot: number,
): void {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group, offset);
    pass.setIndexBuffer(r.index, "uint32");
    // per-view-culled producers lay DrawIndexedIndirect records out slot-major (`viewStride`
    // bytes/camera); a view-independent draw leaves it 0
    pass.drawIndexedIndirect(
        draw.args.indirect,
        (draw.args.offset ?? 0) + slot * (draw.args.viewStride ?? 0),
    );
}

/**
 * one camera's prepass (`sear:prepass`) recorded onto `Render.encoder`: a single single-sample pass
 * emitting the camera's opt-in lanes. It owns its own depth (cleared + `less` + write, so only the
 * front-most opaque / `clip` fragment writes each lane); that depth is *stored* + published as
 * `view.depth` when the camera carries {@link Depth}, otherwise discarded (TBDR: it stays in tile memory,
 * never reaching main RAM). Each requested color lane is one MRT attachment cleared to the lane's clear
 * value and published onto `view.<lane>` (today the id lane → `view.tag`). Binds group 0 only: no shadow
 * map, no lighting. `alpha` surfaces are excluded (a transparent pixel has no single owner). **One
 * prepass regardless of lane count**: the lane set selects the pipeline + the attachment list, not the
 * pass count; an empty draw list still clears every lane
 */
function renderPrepass(
    eid: number,
    view: View,
    items: { draw: Draw; r: Recorded }[],
    lanes: ColorLane[],
    storeDepth: boolean,
): void {
    if (!Render.encoder || !view.framebuffer) return;
    const offset = [view.slot * VIEW_STRIDE];
    const depth = depthView(eid, view.width, view.height);
    const key = laneKey(lanes);
    const colorAttachments = lanes.map((lane) => {
        const target = laneTarget(eid, lane, view.width, view.height);
        lane.set(view, target.texture); // publish `view.<lane>`
        return {
            view: target.view,
            loadOp: "clear" as const,
            storeOp: "store" as const,
            clearValue: lane.clear,
        };
    });
    const pass = Render.encoder.beginRenderPass({
        label: `sear-prepass/${eid}`,
        timestampWrites: Compute.span?.("sear:prepass"),
        colorAttachments,
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            // store the depth only for a `Depth` consumer; the id lane needs the test, not the stored
            // result, so a tag-only camera discards it (TBDR keeps it in tile memory)
            depthStoreOp: storeDepth ? "store" : "discard",
            depthClearValue: 0,
        },
    });
    let draws = 0;
    for (const { draw, r } of items) {
        const pipe = r.c.prepass.get(key);
        if (pipe) {
            bind(pass, pipe, draw, r, r.prepassGroup, offset, view.slot);
            draws++;
        }
    }
    pass.end();
    Compute.indirect?.("sear:prepass", draws);
    view.depth = storeDepth ? depth : null;
}

// build (and cache) a background's group 0 — frame / view (whole buffer, the per-camera slice is the
// setBindGroup dynamic offset) / lighting + its own bindings resolved by name. Lazy: the buffers are stable
// post-warm, so it builds once on first use and caches on the CompiledBg (a missing binding skips the draw)
function bgGroup(cb: CompiledBg): GPUBindGroup | null {
    if (cb.group) return cb.group;
    const entries: GPUBindGroupEntry[] = [
        { binding: FRAME, resource: { buffer: Frame.buffer } },
        { binding: VIEW, resource: { buffer: Render.viewBuffer, size: VIEW_STRIDE } },
        { binding: LIGHTING, resource: { buffer: Lighting.buffer } },
    ];
    for (let k = 0; k < cb.slots.length; k++) {
        const { name, type } = cb.slots[k];
        const res = registryFor(type).get(name);
        if (!res)
            return warnSkip(`background:${cb.name}`, `binding "${name}" (${type}) not published`);
        entries.push({ binding: k + BG_BASE, resource: bindResource(type, res) });
    }
    cb.group = Compute.device.createBindGroup({
        label: `sear-bg-${cb.name}`,
        layout: cb.layout,
        entries,
    });
    return cb.group;
}

// the camera's selected backdrop, or null (no `Backdrop` component, or its name isn't a compiled
// background). Membership-gated — a bare `Backdrop.name.get` reads 0 for a non-member, which would alias
// the first registered background, so the `state.has` check is what keeps the no-backdrop path on the clear
function backdrop(state: State, eid: number): CompiledBg | null {
    if (!state.has(eid, Backdrop)) return null;
    const name = Backgrounds.name(Backdrop.name.get(eid));
    return (name && _backgrounds.get(name)) || null;
}

/**
 * one camera's geometry pass (`sear:color`) recorded onto `Render.encoder`: shades every opaque draw,
 * then composites every `blend` draw over them (`less-equal` depth-tested against the opaque depth,
 * depth-write off): one color target, no MRT (the screen-space lanes are {@link renderPrepass}'s). With
 * `Camera.antialias` on (the default) it's a 4× MSAA pass resolved into the offscreen; off, it renders
 * single-sample straight into the offscreen (and binds the surfaces' single-sample pipeline twins,
 * compiled lazily by {@link ensureSingle}). Opaque and transparent share one `beginRenderPass` (nothing
 * reads the color between them, so they fuse into one tile round-trip). Group 1 is the sun shadow seam:
 * sear's own shadow map + light params, or its 1×1 fallback (fully lit) when no light casts. An empty
 * draw list still clears the framebuffer. `bg` (the camera's {@link Backdrop} selection) draws a fullscreen
 * backdrop between the opaque and blend draws: masked to far-plane pixels by the depth test, so geometry
 * overdraws it and blended draws composite over it; null leaves the flat clear color as the only backdrop
 */
function renderColor(
    eid: number,
    view: View,
    items: { draw: Draw; r: Recorded }[],
    bg: CompiledBg | null = null,
): void {
    if (!Render.encoder || !view.framebuffer) return;
    // per-camera AA: 4× MSAA when `Camera.antialias` is on (the default the Camera trait seeds), else
    // single-sample. A scene attribute or a runtime `Camera.antialias.set(eid, 0)` flips it live
    const aa = Camera.antialias.get(eid) !== 0;
    const clear = unpackColor(Camera.clearColor.get(eid));
    const offset = [view.slot * VIEW_STRIDE];
    const { color: msaaColor, depth } = colorTargets(eid, view.width, view.height, aa);
    const pass = beginColor(eid, msaaColor, depth, view.framebuffer, clear);
    pass.setBindGroup(1, shadowGroup());
    // tally the indirect draws this camera issues (opaque + blend) so the profiler derives the injected
    // validation floor (gpu.md); the honest count is post the `if (pipe)` skip
    let draws = 0;
    for (const { draw, r } of items) {
        if (!aa) ensureSingle(r.c); // lazy-compile this surface's single-sample twin for the no-AA camera
        const pipe = aa ? r.c.color : r.c.single?.color;
        if (pipe) {
            bind(pass, pipe, draw, r, r.group, offset, view.slot);
            draws++;
        }
    }
    // the backdrop: a fullscreen triangle at the far plane, after opaque (the depth test masks it to
    // un-rendered pixels) and before blend (so transparent draws composite over it). The bg pipeline
    // carries the shadow group 1 in its layout (unused) like every color pipeline, so the group bound at
    // the pass top survives the switch for the blend draws after it
    if (bg) {
        const pipe = aa ? bg.color : bg.single;
        const group = bgGroup(bg);
        if (group) {
            pass.setPipeline(pipe);
            pass.setBindGroup(0, group, offset);
            pass.draw(3);
        }
    }
    for (const { draw, r } of items) {
        const pipe = aa ? r.c.transparent : r.c.single?.transparent;
        if (pipe) {
            bind(pass, pipe, draw, r, r.group, offset, view.slot);
            draws++;
        }
    }
    pass.end();
    Compute.indirect?.("sear:color", draws);
}

/**
 * sear's geometry-emit ordering anchor **and** prepass, per camera carrying a lane marker
 * ({@link Tag} / {@link Depth}). It collapses the old empty depth anchor + the tag pass into one
 * single-sample pass that emits the camera's opt-in lanes (the id lane → `view.tag`, the depth lane →
 * `view.depth`), the shape Bevy's prepass takes. It's also the **anchor**: a producer whose per-frame
 * compute writes the geometry sear reads (vertices / indices, or an instanced surface's `transforms` /
 * `eids`) declares `before: [PrepassSystem]` so its emit precedes every geometry-reading pass (the
 * prepass, the shadow map, and the color pass all read it within the frame; an emit landing between them
 * would desync the reads). It runs first among the geometry passes (`after: [BeginFrameSystem]`), so it
 * resolves the frame's draw list **once** into `_frameDraws` for the shadow map + color pass to share. A
 * screen-space effect still slots into the `after: [PrepassSystem], before: [ColorSystem]` seam. A camera
 * carrying no lane marker runs no prepass (the bare path), but the anchor + resolve still run
 */
export const PrepassSystem: System = {
    name: "prepass",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    update(state) {
        if (!Render.encoder) return;
        // resolve once for the prepass + shadow map + color pass (they all run after this)
        _frameDraws = resolveDraws();
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            // markers → the requested lanes. The id lane is a color attachment; depth is the
            // depth-stencil, stored only when the camera carries `Depth`. Reset both each frame so a
            // camera that drops a marker stops publishing its lane
            view.tag = null;
            view.depth = null;
            const lanes = COLOR_LANES.filter((l) => state.has(eid, l.marker));
            const storeDepth = state.has(eid, Depth);
            if (lanes.length === 0 && !storeDepth) continue; // no lane requested — bare path
            renderPrepass(eid, view, _frameDraws, lanes, storeDepth);
        }
    },
};

/**
 * sear's geometry pass, per camera: shades every opaque draw then composites every `blend` draw over
 * them in one 4× MSAA pass (its own 4× color + depth), resolved into the offscreen once. Binds the sun
 * shadow seam (group 1): sear's own shadow map + light params it samples inline, or its fallback (fully
 * lit) when no light casts. Renders the shared `_frameDraws` (resolved once by {@link PrepassSystem}).
 * Runs after every screen-space effect ordered `before: [ColorSystem]`; `before: [GlazeSystem]` makes it
 * sear's terminal offscreen write, so glaze reads `view.framebuffer` only after the resolve lands (glaze
 * never imports sear)
 */
export const ColorSystem: System = {
    name: "color",
    group: "draw",
    annotations: { mode: "always" },
    after: [PrepassSystem],
    before: [GlazeSystem],
    update(state) {
        if (!Render.encoder) return;
        for (const eid of state.query([Camera, Sear])) {
            const view = Views.get(eid);
            if (!view?.framebuffer) continue;
            renderColor(eid, view, _frameDraws, backdrop(state, eid));
        }
    },
};

/**
 * pose the sun's CSM cascade cameras + the point/spot combo cameras from the casting lights + the main Sear
 * camera, so `BeginFrameSystem` packs their viewProjs this frame and the Part pack culls casters into each
 * slot as one more view (the unified culled-combo spine). `simulation` group, before the draw frame opens.
 * No-op for the sun when no directional light carries a {@link Shadow} (the zero-cost off path): the atlas
 * pass is skipped and sear falls back to fully lit
 */
const ShadowCameraSystem: System = {
    name: "shadow-camera",
    group: "simulation",
    annotations: { mode: "always" },
    update(state) {
        let main = -1;
        for (const eid of state.query([Camera, Sear])) {
            main = eid;
            break;
        }
        _pointFrames = updatePointShadows(state, main);
        updateCascades(state, main);
        // allocate each atlas's re-gather list here, before record() (PrepassSystem) builds the cast bind
        // groups that bind it — so the first casting frame's groups include it (the alloc clears _groups), no
        // one-frame delay. Idempotent once allocated; the render fns call it again harmlessly
        if (_pointFrames.length > 0 && _shadowReady) _pointRegather.ensure(pointCasters() * 6);
        if (cascadeCount() > 0 && _shadowReady) _cascadeRegather.ensure(MAX_CASCADES);
    },
};

/**
 * render the casters' depth into the shadow atlases (the point/spot tiles + the CSM cascades) and publish the
 * seams for sear's color pass to sample inline. `after: [PrepassSystem]` so every position-writing producer
 * (pinned before the anchor) has emitted and `_frameDraws` is resolved; `before: [ColorSystem]` so the
 * atlases + seams are ready before sear shades. No casting light → no pass, sear falls back to fully lit.
 * Bevy's shape: the shadow maps are light-data-gated, sampled inline, no separate shadow plugin
 */
const ShadowMapSystem: System = {
    name: "shadowmap",
    group: "draw",
    annotations: { mode: "always" },
    after: [PrepassSystem],
    before: [ColorSystem],
    update() {
        renderPointShadows();
        renderCascades();
    },
};

// the bindings sear's default materials read: the `eids` + `transforms` instance convention (sear
// applies the standard transform) plus per-entity `color`. Producers publish these buffers by name
// (Part does); a surface declaring them is instanced
const colorBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    // sRGB-packed LDR color (the `srgb8x4` slab) — one u32 per entity, decoded with `unpackLdrColor`
    color: { type: "storage", element: "u32" },
};

// the lit materials add the per-entity `material` slab (the Material component's metallic / roughness /
// emissive / occlusion as a `vec4<f16>` — f16 keeps emissive HDR-capable as a glow strength while the
// bounded lanes stay finer than unorm8). `unlit` omits it — it never shades, so it stays on
// `colorBindings`. One extra storage binding: the color pass goes 8 → 9 of the 10-per-stage ceiling (gpu.md)
const litBindings: Record<string, Binding> = {
    ...colorBindings,
    material: { type: "storage", element: "vec4<f16>" },
};

/**
 * Sear: the one kitchen renderer. A GPU-driven raster forward pass: a 4× MSAA color pass (opaque draws
 * then `blend` draws composited over them, fused into one render pass) and an opt-in single-sample
 * prepass emitting per-camera lanes (the {@link Tag} → `view.tag` id lane, the {@link Depth} →
 * `view.depth` lane), with sun shadows sampled inline in the FS. Add `SearPlugin` and give a Camera the
 * {@link Sear} marker and the happy path renders. Sun shadows are data-gated on the {@link Shadow}
 * component on a `DirectionalLight`: add it to cast (and tune), omit it for the fully-lit bare path (no
 * shadow map allocated), exactly like a camera without a lane marker runs no prepass. No separate shadow
 * plugin, no coordination singleton: sear owns its own shadow map and binds it (Bevy's clustered-forward
 * shape). Sear renders into the offscreen
 * (`view.framebuffer`) and never the swapchain; presenting it is a separate **composite** the consumer
 * picks: {@link GlazePlugin} (the engine default postfx composite) or a custom one (orrstead ships a
 * fused compute composite). So sear depends only on {@link RenderPlugin}; list a composite alongside it
 * or nothing reaches the swapchain. `ColorSystem` still orders `before: [GlazeSystem]` so glaze, *when
 * present*, composites after the resolve (the ordering ref drops harmlessly when glaze isn't registered).
 */
export const SearPlugin: Plugin = {
    name: "Sear",
    components: { Sear, Tag, Depth, Shadow, Material, Backdrop },
    systems: [PrepassSystem, ColorSystem, ShadowCameraSystem, ShadowMapSystem],
    // SlabPlugin: the `Material` slab is collected + published as `"material"`, and `initMaterial`
    // bases every slot through it (Part brings SlabPlugin anyway; declaring it keeps sear self-sufficient)
    dependencies: [RenderPlugin, SlabPlugin],
    traits: {
        Shadow: { defaults: () => ({ ...SHADOW_DEFAULTS }) },
        Material: MaterialTraits,
        Backdrop: BackdropTraits,
    },

    // sear's default materials, shading per-instance `color` + `material` at three lighting modes. They
    // ship with the renderer, not Part: Part publishes the data (`eids` + `color`), sear adds its own
    // `Material` slab and shades with its metallic-roughness `litPbr`. `Part.surface` defaults to
    // "default" (per-pixel); "vertex" (per-vertex Gouraud) and "unlit" are picked per-Part. The instance
    // transform is sear's convention — these declare the bindings and omit a transform vs chunk. `pbr()`
    // builds the Pbr struct from the packed `material` lanes; the engine default has no specular until a
    // Material sets metallic > 0 (dielectric 0), so a bare Part shades exactly like the pre-PBR diffuse.
    initialize() {
        // a fresh State recreates its own off-screen shadow cameras lazily — drop any eids cached by
        // a prior build so this re-run never aliases recycled entities (ecs.md module-scope contract)
        resetPointShadows();
        resetCascades();
        initMaterial();
        // clear the backdrop registry so a rebuild re-registers identically and a plugin toggled off leaves
        // no stale entry (the Surfaces/Draws/Meshes reload-safety shape — RenderPlugin.initialize clears those)
        Backgrounds.clear();
        // build the Pbr struct + the emissive tint (Color.rgb * the emissive strength lane) from the
        // f16 material lanes (promoted to f32 for the shading math). emissive is an unbounded HDR glow
        // strength; dielectric 0 → metallic 0 is specular-free (the flat shallot default)
        const PbrPreamble = /* wgsl */ `
        fn matOf(eid: u32) -> Pbr {
            let m = material[eid];
            return Pbr(unpackLdrColor(color[eid]).rgb, f32(m.x), f32(m.y), f32(m.w), 0.0);
        }
        fn emissiveOf(eid: u32) -> vec3<f32> {
            return unpackLdrColor(color[eid]).rgb * f32(material[eid].z);
        }`;
        Surfaces.register({
            name: "default",
            bindings: litBindings,
            preamble: PbrPreamble,
            fs: /* wgsl */ `col = vec4<f32>(litPbr(matOf(eid), worldNormal, world) + emissiveOf(eid), 1.0);`,
        });
        Surfaces.register({
            name: "vertex",
            bindings: litBindings,
            preamble: PbrPreamble,
            interpolators: { litColor: "vec3<f32>" },
            vs: /* wgsl */ `litColor = litPbr(matOf(eid), normalize(worldNormal), world.xyz) + emissiveOf(eid);`,
            fs: /* wgsl */ `col = vec4<f32>(litColor, 1.0);`,
        });
        Surfaces.register({
            name: "unlit",
            bindings: colorBindings,
            fs: /* wgsl */ `col = vec4<f32>(unpackLdrColor(color[eid]).rgb, 1.0);`,
        });
    },

    async warm() {
        if (!Compute.device) return;
        await prepareSear(Compute.device);
    },

    dispose(state) {
        destroyPointShadows(state);
        destroyCascades(state);
        disposeSear();
    },
};
