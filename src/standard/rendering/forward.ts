import { registration } from "../../engine";
import { initializeSurfaceState } from "./contract";
import { initializeDrawState } from "./registry";
// StandardRenderer — the one shallot renderer. A GPU-driven raster *forward* pass (Aaltonen-Haar / niagara
// submission spine, primary visibility only) with sun shadows sampled inline in the FS, matching Bevy's
// clustered-forward shape. One renderer, one plugin (`StandardRenderingPlugin`), no layers behind seams: one color
// pass (opaque draws then `blend` draws composited over them in a single `beginRenderPass`), an
// opt-in single-sample **prepass** emitting per-camera lanes (the `PickingPrepass` / `DepthPrepass` markers, Bevy's
// `DepthPrepass` / `NormalPrepass` shape), and sun shadows (the `Shadow` component on a directional
// light) are all sear-internal features, gated by data the way Bevy gates a shadow map on light data —
// not composed plugins coordinating through a singleton.
//
// Sun shadows: the CPU/ECS half (the off-screen light camera + placement) lives in ./shadows; the GPU half
// (the shadow map, its render through sear's compiled prepass depth pipelines, and the group-1 binding the
// FS samples) lives in ./atlas. This file owns the WGSL-scaffold-agnostic renderer plumbing: components +
// registries, per-draw bind-group resolution, per-camera targets, the systems, and the plugin — the pure
// codegen lives in ./codegen, pipeline compilation in ./pipelines. StandardRenderer renders its own map and reads its
// own shadow state directly — nothing publishes into it. Add a `Shadow` to the sun to cast; omit it for the
// fully-lit bare path (no map allocated), exactly like a camera without a lane marker runs no prepass.

import type { TgpuBindGroupLayout, TgpuBuffer, TgpuRenderPipeline } from "typegpu";
import tgpu, { isBuffer, isUsableAsStorage, isUsableAsUniform } from "typegpu";
import type { AnyData } from "typegpu/data";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    type MeshBinding,
    Meshes,
    type MeshIndex,
    MeshInstance,
    MeshPlugin,
} from "../../core/mesh";
import type { View } from "../../core/rendering";
import {
    BeginFrameSystem,
    Camera,
    OverlaySystem,
    Render,
    RenderingPlugin,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { u32, unpackColor, vec4 } from "../../engine";
import { Xform } from "../../engine/utils";
import { GlazeSystem } from "../../transitional/glaze";
import {
    cascadeRegather,
    disposeShadowAtlas,
    initializeShadowAtlasState,
    pointRegather,
    renderCascades,
    renderPointShadows,
    resetShadowAtlas,
    setPointFrames,
    shadowGroup,
    shadowLayout,
    shadowReady,
} from "./atlas";
import { boundPipeline } from "./bound";
import type { BundleDraw, PassBundle } from "./bundle";
import { bundleChanged, bundleDraw, newPassBundle, recordBundle } from "./bundle";
import {
    COLOR_LANES,
    type ColorLane,
    DEPTH_FORMAT,
    laneKey,
    PickingPrepass,
    SAMPLE_COUNT,
} from "./codegen";
import {
    type Background,
    Backgrounds,
    fsCtxSchema,
    type Surface,
    Surfaces,
    surfaceLayout as typedLayout,
    registerSurface as typedRegister,
    VsIn,
    vsPatchSchema,
} from "./contract";
import { engineLayout, litPbr } from "./engine";
import { partTable } from "./part";
import {
    type BindResource,
    bgQuant,
    type CompiledBackground,
    type CompiledSurface,
    clearGroups,
    compileBackground,
    compileSurface,
    engineGroup,
    ensureSingle,
    getBackground,
    getCompiledSurface,
    getGroup,
    initializePipelineState,
    preparePipelines,
    resetPipelineCaches,
    type SurfaceGroupEntry,
    setGroup,
} from "./pipelines";
import { initializeRegatherState, prepareRegather } from "./regather";
import type { Draw } from "./registry";
import { Draws } from "./registry";
import { checkShadowConfig, Pbr } from "./shade";
import {
    cascadeCount,
    destroyCascades,
    destroyPointShadows,
    MAX_CASCADES,
    type PointShadowFrame,
    pointCasters,
    resetCascades,
    resetPointShadows,
    SHADOW_DEFAULTS,
    Shadow,
    updateCascades,
    updatePointShadows,
} from "./shadows";

export { DEPTH_FORMAT, PICKING_ID_FORMAT, PICKING_ID_NONE, PickingPrepass } from "./codegen";

interface SearState {
    warned: Set<string>;
    frameDraws: FrameDraw[];
    frameCount: number;
    depth: Map<number, { texture: GPUTexture; view: GPUTextureView; w: number; h: number }>;
    laneTargets: Map<string, { texture: GPUTexture; view: GPUTextureView; w: number; h: number }>;
    colorTargets: Map<number, ColorTargets>;
    clearValue: { r: number; g: number; b: number; a: number };
    clearPacked: number;
    msaaColor: ViewColorAttachment;
    directColor: ViewColorAttachment;
    colorAttachments: ViewColorAttachment[];
    colorDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & { view: GPUTextureView };
    colorPass: GPURenderPassDescriptor;
    colorBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] };
    colorBundles: Map<number, PassBundle>;
    colorProgram: BundleDraw[];
    prepassBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] };
    prepassBundles: Map<number, PassBundle>;
    prepassProgram: BundleDraw[];
    pointFrames: PointShadowFrame[];
}

const searStateKey = { create: createSearState };

function createSearState(): SearState {
    const clearValue = { r: 0, g: 0, b: 0, a: 1 };
    const msaaColor: ViewColorAttachment = {
        view: null!,
        resolveTarget: null!,
        loadOp: "clear",
        storeOp: "discard",
        clearValue,
    };
    const directColor: ViewColorAttachment = {
        view: null!,
        loadOp: "clear",
        storeOp: "store",
        clearValue,
    };
    const colorAttachments = [msaaColor];
    const colorDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & { view: GPUTextureView } =
        {
            view: null!,
            depthLoadOp: "clear",
            depthStoreOp: "discard",
            depthClearValue: 0,
        };
    return {
        warned: new Set(),
        frameDraws: [],
        frameCount: 0,
        depth: new Map(),
        laneTargets: new Map(),
        colorTargets: new Map(),
        clearValue,
        clearPacked: -1,
        msaaColor,
        directColor,
        colorAttachments,
        colorDepth,
        colorPass: {
            label: "",
            colorAttachments,
            depthStencilAttachment: colorDepth,
        },
        colorBundleDesc: {
            label: "sear-color",
            colorFormats: [],
            depthStencilFormat: DEPTH_FORMAT,
            sampleCount: 1,
        },
        colorBundles: new Map(),
        colorProgram: [],
        prepassBundleDesc: {
            label: "sear-prepass",
            colorFormats: [],
            depthStencilFormat: DEPTH_FORMAT,
            sampleCount: 1,
        },
        prepassBundles: new Map(),
        prepassProgram: [],
        pointFrames: [],
    };
}

function _searState(world: World): SearState {
    return world.resource(searStateKey);
}

/**
 * marker selecting StandardRenderer as the active renderer on a Camera entity. A camera carrying it renders through
 * sear's color pass, plus the opt-in prepass lanes its {@link PickingPrepass} / {@link DepthPrepass} markers request.
 * Lives in the renderer impl with the systems that query it; the thin `sear` barrel re-exports it to the
 * game author, the `sear` barrel exports the systems to an extender.
 *
 * @example
 * ```
 * const camera = world.create();
 * world.add(camera, Camera);
 * world.add(camera, StandardRenderer);
 * world.add(camera, Transform);
 * ```
 */
export const StandardRenderer = {};

/**
 * opt a StandardRenderer camera into the **depth lane**: the prepass *stores* its single-sample depth and publishes
 * it as `view.depth` (without this marker the prepass depth is discarded, never reaching main memory).
 * Requestable on its own (a depth-only consumer needs no id) or alongside {@link PickingPrepass} (one prepass writes
 * both). Bevy's `DepthPrepass`. A screen-space consumer (AO, fog, volumetrics) adds it to read
 * `view.depth`.
 *
 * @example
 * ```
 * const camera = world.create();
 * world.add(camera, Camera);
 * world.add(camera, StandardRenderer);
 * world.add(camera, DepthPrepass);
 * world.add(camera, Transform);
 * ```
 */
export const DepthPrepass = {};

/**
 * Selects a surface by its Surfaces registry ID and supplies its material parameters.
 * Defaults to the "default" surface. Mesh instances without Material use that surface
 * with flat params. The default and vertex surfaces read params as
 * `(metallic, roughness, emissive, occlusion)`; Color supplies linear base albedo.
 * Metallic, roughness and occlusion are in [0,1]; emissive is base-color glow strength.
 */
export const Material = {
    surface: u32,
    params: vec4,
};

const MATERIAL_FLAT: [number, number, number, number] = [0, 1, 0, 1];

const MaterialTraits = {
    defaults: (world: World) => ({
        surface: world.resource(Surfaces).id("default") ?? 0,
        params: MATERIAL_FLAT,
    }),
};

// base every slot to the flat material so a MeshInstance lacking the Material component shades like the pre-PBR
// diffuse default (an entity with Material overwrites its slot via its registration default on add). Mirrors
// MeshInstance.initPart's magenta Color base; the pack gates each slot on membership, so stale slots never draw.
function initMaterial(world: World): void {
    partTable(world).bindFields(Material, { surface: "surface", material: "params" });
    const seedMissingMaterial = (eid: number) => {
        if (!world.has(eid, Material)) {
            const storage = world.storage(Material);
            storage.surface.set(eid, world.resource(Surfaces).id("default") ?? 0);
            storage.params.set(eid, ...MATERIAL_FLAT);
        }
    };
    const removeMissingMaterialDefault = world.observeMembership(MeshInstance, (eid, present) => {
        if (present) seedMissingMaterial(eid);
    });
    for (const eid of world.query([MeshInstance])) seedMissingMaterial(eid);
    world.onDispose(removeMissingMaterialDefault);
}

/**
 * a registered background: a renderer-agnostic *view-ray → HDR color* recipe sear draws as a fullscreen
 * backdrop on the un-rendered pixels (the standard infinite-skybox technique). Its `layout` comes from
 * {@link backgroundLayout} and names the schema-backed resources its TGSL `fs` closes over. The `fs`
 * receives `BackgroundContext.dir`, the normalized world-space view ray sear reconstructs from `view.invViewProj`,
 * and returns an HDR `vec3f`; it may also close over the canonical engine layout's view, lighting, and
 * frame resources. Modeled on {@link Surface}, but backdrop-only: no mesh, instancing, interpolators, or
 * blend modes; the engine names no sky concept, a plugin owns its own sky math.
 */

/**
 * select a StandardRenderer camera's backdrop: the {@link Backgrounds} recipe drawn behind the scene as a fullscreen
 * view-ray → color fill on the un-rendered pixels. Without it the camera shows the flat `Camera.clearColor`
 * (the opt-in fallback). The recipe is registered in code with `registerBackground`; this picks one per
 * camera by name.
 *
 * @example
 * ```
 * const camera = world.create();
 * world.add(camera, Camera);
 * world.add(camera, StandardRenderer);
 * world.add(camera, CameraBackground, { name: world.resource(Backgrounds).id("gradient") ?? 0 });
 * world.add(camera, Transform);
 * ```
 */
export const CameraBackground = {
    /** the {@link Backgrounds} id drawn behind the scene */
    name: u32,
};

// a draw resolving to null is a silent skip — usually a typo'd binding or an
// unpublished resource. Warn once per draw so it's visible without spamming

function warnSkip(world: World, draw: string, cause: string): null {
    const _searState = world.resource(searStateKey);

    if (!_searState.warned.has(draw)) {
        _searState.warned.add(draw);
        console.warn(`sear: draw "${draw}" skipped — ${cause}`);
    }
    return null;
}

type BufferEntry = {
    storage?: (count: number) => d.WgslArray<d.AnyWgslData>;
    uniform?: d.AnyData;
};

/**
 * Validate a typed mesh override against the synthesized TypeGPU layout entry before the deliberately
 * loose createBindGroup boundary. Raw GPUBuffer remains the explicit unchecked WebGPU escape.
 * @internal
 */
export function validateMeshBindingOverrides(
    layout: { entries: Record<string, object> },
    overrides?: Record<string, unknown>,
): void {
    if (!overrides) return;
    for (const [name, rawEntry] of Object.entries(layout.entries)) {
        const resource = overrides[name];
        if (!resource || !isBuffer(resource)) continue;
        const entry = rawEntry as BufferEntry;
        if (entry.storage) {
            if (!isUsableAsStorage(resource)) {
                throw new Error(`mesh binding "${name}" is missing storage usage`);
            }
            const expected = entry.storage(1);
            if (
                !d.isWgslArray(resource.dataType) ||
                !d.deepEqual(
                    resource.dataType.elementType as AnyData,
                    expected.elementType as AnyData,
                )
            ) {
                throw new Error(`mesh binding "${name}" has the wrong storage schema`);
            }
        } else if (entry.uniform) {
            if (!isUsableAsUniform(resource)) {
                throw new Error(`mesh binding "${name}" is missing uniform usage`);
            }
            if (!d.deepEqual(resource.dataType as AnyData, entry.uniform)) {
                throw new Error(`mesh binding "${name}" has the wrong uniform schema`);
            }
        } else {
            throw new Error(`mesh binding "${name}" is not a buffer entry`);
        }
    }
}

// a draw resolved through the surface contract: the compiled pipeline set + the
// per-draw group-2 state (engine group 0 resolves per slot at draw time via `engineGroup`;
// group 1 is the pass's — `shadowGroup` for color, the atlas layouts' own for point/cascade)
type RecordedSurface = { t: CompiledSurface; g: SurfaceGroupEntry; index: MeshIndex };

export type Recorded = RecordedSurface;

// one resolved draw of the frame: the entry-owned record `record` rewrites in place
type FrameDraw = { draw: Draw; r: Recorded };

/**
 * the color + transparent + per-lane-set prepass pipelines and the bind-group state sear records a draw
 * with, or null to skip it. All pipelines share one bind group (same group-0 layout). A surface with
 * no compiled pipeline isn't sear's (silent skip); a missing mesh or unpublished binding warns once.
 * The per-slot bind groups cache per draw, rebuilt only on a resource identity change; the fixed uniforms
 * are stable, so untracked
 */
function record(world: World, draw: Draw, capacity: number): FrameDraw | null {
    const surface = world.resource(Surfaces).get(draw.surface);
    return surface ? recordSurface(world, draw, surface, capacity) : null;
}

// resolve a typed layout's own bindings (never the sear-injected `vertices`) to live resources by the
// entry's kind. Returns the createBindGroup value record + the identity list + each binding's name and the
// registry it resolved from, or the missing binding's name
function typedResources(
    world: World,
    entries: Record<string, object>,
    override?: Record<string, BindResource>,
):
    | {
          values: Record<string, unknown>;
          resources: BindResource[];
          names: string[];
          registries: ReadonlyMap<string, BindResource>[];
      }
    | string {
    const values: Record<string, unknown> = {};
    const resources: BindResource[] = [];
    const names: string[] = [];
    const registries: ReadonlyMap<string, BindResource>[] = [];
    for (const [name, entry] of Object.entries(entries)) {
        if (name === "vertices") continue;
        const registry: ReadonlyMap<string, BindResource> =
            "texture" in entry
                ? world.gpu.textures
                : "sampler" in entry
                  ? world.gpu.samplers
                  : world.gpu.typed;
        const res = override?.[name] ?? registry.get(name);
        if (!res) return name;
        if (isBuffer(res)) validateMeshBindingOverrides({ entries }, { [name]: res });
        resources.push(res);
        names.push(name);
        registries.push(registry);
        // a texture binds a view of the schema's own dimension
        values[name] =
            "texture" in entry
                ? (res as GPUTexture).createView({
                      dimension: (entry as { texture: { dimension: GPUTextureViewDimension } })
                          .texture.dimension,
                  })
                : res;
    }
    return { values, resources, names, registries };
}

// whether a cached entry still binds this frame's live resources: the identities in `resources` order (the
// four mesh streams, the layout's own bindings looked up by their held names, then the atlas lists),
// compared in place so a steady frame neither re-validates nor builds a list
function sameResources(
    g: SurfaceGroupEntry,
    mesh: NonNullable<ReturnType<ReturnType<typeof Meshes.create>["get"]>>,
    pointList: GPUBuffer | null,
    cascadeList: GPUBuffer | null,
): boolean {
    const res = g.resources;
    if (
        res[0] !== mesh.vertices ||
        res[1] !== mesh.position ||
        res[2] !== mesh.quant ||
        res[3] !== mesh.indices
    )
        return false;
    const override = mesh.bindings as Record<string, BindResource> | undefined;
    let k = 4;
    for (let i = 0; i < g.names.length; i++, k++) {
        const name = g.names[i];
        if ((override?.[name] ?? g.registries[i].get(name)) !== res[k]) return false;
    }
    if (pointList && res[k++] !== pointList) return false;
    if (cascadeList && res[k++] !== cascadeList) return false;
    return k === res.length;
}

// one surface bind group against `layout`: the resolved layout values, any override, and the vertex stream.
// The two layout objects share one loose signature here — the color/depth `vertices` element split is real at
// authoring time, but a bind group takes raw buffers either way (the `layout.$` cast class). A module function,
// so the steady `recordSurface` path captures nothing and opens no context
function surfaceGroup(
    world: World,
    values: Record<string, unknown>,
    layout: unknown,
    vertices: TgpuBuffer<AnyData>,
    override?: Record<string, BindResource>,
): GPUBindGroup {
    const root = world.gpu.root;
    return root.unwrap(
        root.createBindGroup(
            layout as TgpuBindGroupLayout,
            {
                ...values,
                ...override,
                vertices,
            } as never,
        ),
    );
}

/**
 * the typed twin of {@link record}: compiled typed pipelines + the per-draw group-2 state cached by
 * layout name — `color` against `layout`; opaque depth-side groups against `layout.depthVariant`; clip
 * depth-side groups against the full layout so cutoff sees material UVs — plus the atlas `eids` swaps
 * and the slot-0 engine group the atlas passes bind.
 */
function recordSurface(
    world: World,
    draw: Draw,
    surface: Surface,
    capacity: number,
): FrameDraw | null {
    const mesh = world.resource(Meshes).get(draw.mesh);
    if (!mesh) return warnSkip(world, draw.name, `mesh "${draw.mesh}" not registered`);
    if (!mesh.position || !mesh.quant)
        return warnSkip(
            world,
            draw.name,
            `mesh "${draw.mesh}" has no quantized position/quant stream`,
        );
    const prev = getGroup(world, draw.name, surface);
    let t = prev?.item.r.t ?? getCompiledSurface(world, surface.name);
    if (!t || t.owner !== surface || t.layout !== surface.layout) {
        // registered after warm (`preparePipelines` compiles the rest) — sync, so no skip frame; a
        // throwing compile (a contract guard, or shader/device validation) must not take down the frame
        // loop, so it degrades to the warn-once skip
        try {
            t = compileSurface(world, surface, capacity);
        } catch (e) {
            return warnSkip(
                world,
                draw.name,
                `typed surface "${surface.name}" failed to compile: ${e}`,
            );
        }
    }

    const pointList = world.resource(pointRegather).eids();
    const cascadeList = world.resource(cascadeRegather).eids();
    // a steady frame compares the cached entry's identities in place; only a changed resource re-resolves,
    // re-validates the overrides and rebuilds the groups
    if (prev && sameResources(prev, mesh, pointList, cascadeList)) {
        prev.item.draw = draw;
        prev.item.r.t = t;
        return prev.item;
    }

    const resolved = typedResources(
        world,
        surface.layout.entries as Record<string, object>,
        mesh.bindings as Record<string, MeshBinding> | undefined,
    );
    if (typeof resolved === "string")
        return warnSkip(world, draw.name, `binding "${resolved}" not published`);
    // geometry + the atlas packed lists join the identity check (a re-gather realloc also clears the
    // whole cache via `clearGroups` — the lists here make the entry self-consistent even without it)
    const resources: BindResource[] = [
        mesh.vertices,
        mesh.position,
        mesh.quant,
        mesh.indices,
        ...resolved.resources,
    ];
    if (pointList) resources.push(pointList);
    if (cascadeList) resources.push(cascadeList);

    const root = world.gpu.root;
    const engineCache = new Map<number, GPUBindGroup>();
    const clip = surface.blend === "clip";
    const depthLayout = clip ? surface.layout : surface.layout.depthVariant;
    const depthVertices = clip ? mesh.vertices : mesh.position;
    const entry: SurfaceGroupEntry = {
        owner: surface,
        layout: surface.layout,
        quant: root.unwrap(mesh.quant),
        color: surfaceGroup(world, resolved.values, surface.layout, mesh.vertices),
        // `alpha` compiles no depth-side pipelines, so it needs no depth-shape groups
        depth:
            surface.blend === "alpha"
                ? null
                : surfaceGroup(world, resolved.values, depthLayout, depthVertices),
        // an authored tag receives the full fragment context (requested uv/localPos + custom varyings),
        // so its tag pair reads the main stream even while an opaque depth-only pass stays compact
        tag: surface.tag
            ? surfaceGroup(world, resolved.values, surface.layout, mesh.vertices)
            : null,
        point:
            t.point && pointList
                ? surfaceGroup(world, resolved.values, depthLayout, depthVertices, {
                      eids: pointList,
                  })
                : null,
        cascade:
            t.cascade && cascadeList
                ? surfaceGroup(world, resolved.values, depthLayout, depthVertices, {
                      eids: cascadeList,
                  })
                : null,
        eids: resolved.values.eids
            ? root.unwrap(resolved.values.eids as TgpuBuffer<AnyData>)
            : null,
        engineCache,
        atlasG0: engineGroup(world, engineCache, 0, root.unwrap(mesh.quant)),
        resources,
        names: resolved.names,
        registries: resolved.registries,
        bound: new Map(),
        item: null!,
    };
    entry.item = { draw, r: { t, g: entry, index: mesh.indices } };
    setGroup(world, draw.name, entry);
    return entry.item;
}

// the frame's resolved draws (the first `_sear.frameCount`), resolved once by RenderPrepassesSystem and shared across
// the prepass, shadow atlases, and color pass — they all draw the same resolved records, so resolving
// per-pass (the old 3×) was wasted work

/**
 * the frame's draw list: every registered {@link Draw} with a compiled surface + published
 * bindings, paired with its cached group-0 state. Camera-independent (the per-slot bind groups it builds
 * against are cached lazily by slot, not baked per camera), so {@link RenderPrepassesSystem}
 * resolves it once per frame into `_sear.frameDraws` and the prepass, shadow map, and color pass all
 * render every camera against that one list
 */
function resolveDraws(world: World, capacity: number): void {
    world.resource(searStateKey).frameCount = 0;
    world.resource(Draws).forEach((draw) => {
        resolveDraw(world, draw, capacity);
    });
}

function resolveDraw(world: World, draw: Draw, capacity: number): void {
    const _searState = world.resource(searStateKey);

    const item = record(world, draw, capacity);
    if (item) _searState.frameDraws[_searState.frameCount++] = item;
}

// the per-camera single-sample depth the prepass writes — always the front-most-fragment test the id
// lane needs, but only *stored* + published as `view.depth` when the camera carries `DepthPrepass` (else the
// store is discarded). Allocated when the prepass runs (any lane marker). TEXTURE_BINDING so a
// screen-space consumer (AO, volumetrics) can sample it the same frame
function depthView(world: World, eid: number, w: number, h: number): GPUTextureView {
    const _searState = world.resource(searStateKey);

    const cached = _searState.depth.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `sear-depth-${eid}`,
        size: { width: w, height: h },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    _searState.depth.set(eid, { texture, view, w, h });
    return view;
}

// the per-(camera, color-lane) screen-space target, sibling to depthView — filled by the prepass.
// Sized to the framebuffer, recreated on resize; the format + usage are the lane's (the id lane is
// r32uint + COPY_SRC for a hover readback + TEXTURE_BINDING for an outline sample). Returns the texture
// (published onto `view.<lane>`) + the color-attachment view — one cache keyed `${eid}:${lane.name}`
// that drives both, so adding a lane needs no new allocator
function laneTarget(
    world: World,
    eid: number,
    lane: ColorLane,
    w: number,
    h: number,
): { texture: GPUTexture; view: GPUTextureView } {
    const _searState = world.resource(searStateKey);

    const key = `${eid}:${lane.name}`;
    const cached = _searState.laneTargets.get(key);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `sear-${lane.name}-${eid}`,
        size: { width: w, height: h },
        format: lane.format,
        usage: lane.usage,
    });
    const entry = { texture, view: texture.createView(), w, h };
    _searState.laneTargets.set(key, entry);
    return entry;
}

type ColorTargets = {
    color: GPUTexture | null;
    colorView: GPUTextureView | null;
    depth: GPUTexture;
    depthView: GPUTextureView;
    w: number;
    h: number;
    aa: boolean;
    /** the camera's color pass label */
    label: string;
};

// the per-camera color-pass targets, by AA mode. AA on: a 4× MSAA color (resolved into the offscreen at
// pass end) + a 4× depth. AA off: no MSAA color (the pass renders straight into view.framebuffer) + a 1×
// depth. The color pass owns this depth (`less` + write, cleared each frame); the prepass + shadow map
// keep their own 1× depth (never cross-compared). Sized to the view + keyed on AA, recreated on resize/toggle
function colorTargets(world: World, eid: number, w: number, h: number, aa: boolean): ColorTargets {
    const _searState = world.resource(searStateKey);

    const cached = _searState.colorTargets.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.aa === aa) return cached;
    cached?.color?.destroy();
    cached?.depth.destroy();
    const samples = aa ? SAMPLE_COUNT : 1;
    const color = aa
        ? world.gpu.device.createTexture({
              label: `sear-color-msaa-${eid}`,
              size: { width: w, height: h },
              format: world.resource(Render).format,
              sampleCount: SAMPLE_COUNT,
              usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null;
    const depth = world.gpu.device.createTexture({
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
        label: `sear-color/${eid}`,
    };
    _searState.colorTargets.set(eid, entry);
    return entry;
}

// the color pass's clear color, its two color attachment shapes and its descriptor, rewritten per camera so
// opening the pass mints only its WebGPU objects
type ViewColorAttachment = Omit<GPURenderPassColorAttachment, "view" | "resolveTarget"> & {
    view: GPUTextureView;
    resolveTarget?: GPUTextureView;
};
// the color pass's clear value, and the packed sRGB it was decoded from: a camera's clear color is
// unpacked only on the frame it changes

// the color bundle's encoder descriptor and per-camera recordings. The bundle holds the camera's whole
// draw program — opaque, backdrop, blend — and is recorded again only when that program or the pass shape
// changes; a steady frame begins the pass and replays it
// per-world color bundle state is retained in `_sear`.

// the prepass bundle's encoder descriptor (its color formats are the camera's lane set, rewritten per
// camera), its per-camera recordings and this frame's program
// per-world prepass bundle state is retained in `_sear`.

// the geometry pass (one color target — the prepass lanes ride their own pass), cleared to `_sear.clearValue`.
// AA on: the opaque draws clear + write `msaaColor`, the transparent draws blend over, and it resolves into
// the offscreen once at pass end (`discard` — the resolve fires regardless and nothing reads the MSAA target
// after). AA off: `msaaColor` is null — render straight into the offscreen, no resolve, **`store`** the
// result (`discard` would throw away the only copy → a black frame). The depth `discard`s either way
// (transient)
function beginColor(
    world: World,
    label: string,
    msaaColor: GPUTextureView | null,
    depth: GPUTextureView,
    framebuffer: GPUTextureView,
): GPURenderPassEncoder {
    const _searState = world.resource(searStateKey);

    if (msaaColor) {
        _searState.msaaColor.view = msaaColor;
        _searState.msaaColor.resolveTarget = framebuffer;
        _searState.colorAttachments[0] = _searState.msaaColor;
    } else {
        _searState.directColor.view = framebuffer;
        _searState.colorAttachments[0] = _searState.directColor;
    }
    _searState.colorDepth.view = depth;
    _searState.colorPass.label = label;
    _searState.colorPass.timestampWrites = world.gpu.span?.("sear:color");
    return world.resource(Render).encoder!.beginRenderPass(_searState.colorPass);
}

/**
 * one camera's prepass (`sear:prepass`) recorded onto `Render.encoder`: a single single-sample pass
 * emitting the camera's opt-in lanes. It owns its own depth (cleared + `less` + write, so only the
 * front-most opaque / `clip` fragment writes each lane); that depth is *stored* + published as
 * `view.depth` when the camera carries {@link DepthPrepass}, otherwise discarded (TBDR: it stays in tile memory,
 * never reaching main RAM). Each requested color lane is one MRT attachment cleared to the lane's clear
 * value and published onto `view.<lane>` (today the id lane → `view.pickingId`). Binds group 0 only: no shadow
 * map, no lighting. `alpha` surfaces are excluded (a transparent pixel has no single owner). **One
 * prepass regardless of lane count**: the lane set selects the pipeline + the attachment list, not the
 * pass count; an empty draw list still clears every lane
 */
function renderPrepass(
    world: World,
    eid: number,
    view: View,
    items: FrameDraw[],
    count: number,
    lanes: ColorLane[],
    storeDepth: boolean,
): void {
    const _render = world.resource(Render);
    const _searState = world.resource(searStateKey);

    if (!_render.encoder || !view.framebuffer) return;
    const depth = depthView(world, eid, view.width, view.height);
    const key = laneKey(lanes);
    const colorAttachments = lanes.map((lane) => {
        const target = laneTarget(world, eid, lane, view.width, view.height);
        lane.set(view, target.texture); // publish `view.<lane>`
        return {
            view: target.view,
            loadOp: "clear" as const,
            storeOp: "store" as const,
            clearValue: lane.clear,
        };
    });
    const pass = _render.encoder.beginRenderPass({
        label: `sear-prepass/${eid}`,
        timestampWrites: world.gpu.span?.("sear:prepass"),
        colorAttachments,
        depthStencilAttachment: {
            view: depth,
            depthLoadOp: "clear",
            // store the depth only for a `DepthPrepass` consumer; the id lane needs the test, not the stored
            // result, so a tag-only camera discards it (TBDR keeps it in tile memory)
            depthStoreOp: storeDepth ? "store" : "discard",
            depthClearValue: 0,
        },
    });
    let draws = 0;
    const tagLane = lanes.some((lane) => lane.name === "tag");
    const shadow = shadowGroup(world);
    for (let i = 0; i < count; i++) {
        const { draw, r } = items[i];
        const pipe = r.t.prepass.get(key);
        const group = tagLane ? (r.g.tag ?? r.g.depth) : r.g.depth;
        if (pipe && group) {
            const step = bundleDraw(_searState.prepassProgram, draws);
            step.pipeline = boundPipeline(r.g, pipe, group, true, r.index) as never;
            step.layout0 = engineLayout;
            step.group0 = engineGroup(world, r.g.engineCache, view.slot, r.g.quant);
            step.layout1 = shadowLayout;
            step.group1 = shadow;
            step.layout2 = null;
            step.group2 = null;
            step.indirect = draw.args.indirect;
            step.offset = (draw.args.offset ?? 0) + view.slot * (draw.args.viewStride ?? 0);
            draws++;
        }
    }
    // the prepass runs only for a camera carrying a lane marker; its lane set, and so its attachment
    // shape, is per camera, so each camera keeps its own recording
    let bundle = _searState.prepassBundles.get(eid);
    if (!bundle) {
        bundle = newPassBundle();
        _searState.prepassBundles.set(eid, bundle);
    }
    _searState.prepassBundleDesc.colorFormats.length = 0;
    for (let l = 0; l < lanes.length; l++)
        _searState.prepassBundleDesc.colorFormats.push(lanes[l].format);
    if (bundleChanged(bundle, _searState.prepassProgram, draws, _searState.prepassBundleDesc)) {
        recordBundle(world, bundle, _searState.prepassProgram, draws, _searState.prepassBundleDesc);
    }
    if (bundle.bundle) pass.executeBundles(bundle.replay);
    pass.end();
    world.gpu.indirect?.("sear:prepass", draws);
    view.depth = storeDepth ? depth : null;
}

// the camera's selected backdrop — or null (no `CameraBackground` component, or its name
// isn't a compiled background). Membership-gated — a bare `CameraBackground.name.get` reads 0 for a non-member,
// which would alias the first registered background, so the `world.has` check is what keeps the
// no-backdrop path on the clear
type BackdropPick = { bg: Background; ct: CompiledBackground };
function backdrop(world: World, eid: number): BackdropPick | null {
    const _backgrounds = world.resource(Backgrounds);

    if (!world.has(eid, CameraBackground)) return null;
    const id = world.storage(CameraBackground).name.get(eid);
    const name = _backgrounds.name(id);
    const bg = name ? _backgrounds.get(name) : undefined;
    if (!bg) return null;
    const ct = getBackground(world, bg.name, bg) ?? compileBackground(world, bg);
    return { bg, ct };
}

// build (and cache on the CompiledBackground) a typed background's own group-2 bind group — slot-invariant
// (the per-slot View rides the engine group 0). Returns null while a binding is unpublished (skip); a
// binding-free background carries no group at all (its empty layout never enters the pipeline layout)
function backgroundGroup(
    world: World,
    bg: Background,
    ct: CompiledBackground,
): GPUBindGroup | null | "none" {
    const entries = bg.layout.entries as Record<string, object>;
    if (Object.keys(entries).length === 0) return "none";
    const resolved = typedResources(world, entries);
    if (typeof resolved === "string") {
        return warnSkip(world, `background:${bg.name}`, `binding "${resolved}" not published`);
    }
    if (
        ct.group2 &&
        ct.group2.resources.length === resolved.resources.length &&
        ct.group2.resources.every((b, k) => b === resolved.resources[k])
    ) {
        return ct.group2.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(bg.layout, resolved.values as never),
    );
    ct.group2 = { group, resources: resolved.resources };
    return group;
}

/**
 * one camera's geometry pass (`sear:color`) recorded onto `Render.encoder`: shades every opaque draw,
 * then composites every `blend` draw over them (`less-equal` depth-tested against the opaque depth,
 * depth-write off): one HDR color target, no MRT (the screen-space lanes are {@link renderPrepass}'s),
 * because each extra target costs bandwidth on every pixel and tile-based GPUs pay it hardest. With
 * `Camera.antialias` on (the default) it's a 4× MSAA pass resolved into the offscreen; off, it renders
 * single-sample straight into the offscreen (and binds the surfaces' single-sample pipeline twins,
 * compiled lazily by {@link ensureSingle}). Opaque and transparent share one `beginRenderPass` (nothing
 * reads the color between them, so they fuse into one tile round-trip). Group 1 is the sun shadow seam:
 * sear's own shadow map + light params, or its 1×1 fallback (fully lit) when no light casts. An empty
 * draw list still clears the framebuffer. `bg` (the camera's {@link CameraBackground} selection) draws a fullscreen
 * backdrop between the opaque and blend draws: masked to far-plane pixels by the depth test, so geometry
 * overdraws it and blended draws composite over it; null leaves the flat clear color as the only backdrop
 */
// one opaque or blended surface draw in a camera's color pass at its view slot, written into the
// camera's bundle program at `at`
function drawColor(
    world: World,
    program: BundleDraw[],
    at: number,
    item: FrameDraw,
    pipe: TgpuRenderPipeline<any>,
    slot: number,
    shadow: GPUBindGroup,
): void {
    const { draw, r } = item;
    const step = bundleDraw(program, at);
    step.pipeline = boundPipeline(r.g, pipe, r.g.color, false, r.index) as never;
    step.layout0 = engineLayout;
    step.group0 = engineGroup(world, r.g.engineCache, slot, r.g.quant);
    step.layout1 = shadowLayout;
    step.group1 = shadow;
    step.layout2 = null;
    step.group2 = null;
    step.indirect = draw.args.indirect;
    step.offset = (draw.args.offset ?? 0) + slot * (draw.args.viewStride ?? 0);
}

function renderColor(
    world: World,
    eid: number,
    view: View,
    items: FrameDraw[],
    count: number,
    bg: BackdropPick | null = null,
): void {
    const _render = world.resource(Render);
    const _searState = world.resource(searStateKey);

    if (!_render.encoder || !view.framebuffer) return;
    // per-camera AA: 4× MSAA when `Camera.antialias` is on (the Camera registration default), else
    // single-sample. `world.storage(Camera).antialias.set(eid, 0)` flips it live
    const aa = world.storage(Camera).antialias.get(eid) !== 0;
    const packed = world.storage(Camera).clearColor.get(eid);
    if (packed !== _searState.clearPacked) {
        const clear = unpackColor(packed);
        _searState.clearValue.r = clear.r;
        _searState.clearValue.g = clear.g;
        _searState.clearValue.b = clear.b;
        _searState.clearPacked = packed;
    }
    const targets = colorTargets(world, eid, view.width, view.height, aa);
    const shadow = shadowGroup(world);
    // the camera's draw program: opaque, then the backdrop, then blend. Building it reads only cached
    // identities, so a steady frame allocates nothing here and the compare below reports no transition
    let draws = 0;
    let indirect = 0;
    for (let i = 0; i < count; i++) {
        const item = items[i];
        if (!aa) ensureSingle(world, item.r.t);
        const pipe = aa ? item.r.t.color : item.r.t.single?.color;
        if (pipe) {
            drawColor(world, _searState.colorProgram, draws++, item, pipe, view.slot, shadow);
            indirect++;
        }
    }
    // the backdrop: a fullscreen triangle at the far plane, after opaque (the depth test masks it to
    // un-rendered pixels) and before blend (so transparent draws composite over it). The bg pipeline
    // carries the shadow group 1 in its layout (unused) like every color pipeline, so the group bound at
    // the pass top survives the switch for the blend draws after it
    if (bg) {
        // a typed backdrop: the shared engine group 0 (a never-read `bgQuant()` fills the meshQuant
        // slot — a background pulls no mesh), the typed shadow group 1 (declared-but-unused, the
        // group-count-compatibility reason `compileBackground` documents), and its own group 2
        const group = backgroundGroup(world, bg.bg, bg.ct);
        if (group) {
            const step = bundleDraw(_searState.colorProgram, draws++);
            step.pipeline = (aa ? bg.ct.color : bg.ct.single) as never;
            step.layout0 = engineLayout;
            step.group0 = engineGroup(world, bg.ct.engineCache, view.slot, bgQuant(world));
            step.layout1 = shadowLayout;
            step.group1 = shadow;
            step.layout2 = group === "none" ? null : bg.bg.layout;
            step.group2 = group === "none" ? null : group;
            step.indirect = null;
            step.offset = 3;
        }
    }
    for (let i = 0; i < count; i++) {
        const item = items[i];
        const pipe = aa ? item.r.t.transparent : item.r.t.single?.transparent;
        if (pipe) {
            drawColor(world, _searState.colorProgram, draws++, item, pipe, view.slot, shadow);
            indirect++;
        }
    }

    let pass = _searState.colorBundles.get(eid);
    if (!pass) {
        pass = newPassBundle();
        _searState.colorBundles.set(eid, pass);
    }
    _searState.colorBundleDesc.colorFormats[0] = _render.format;
    _searState.colorBundleDesc.sampleCount = aa ? SAMPLE_COUNT : 1;
    if (bundleChanged(pass, _searState.colorProgram, draws, _searState.colorBundleDesc)) {
        recordBundle(world, pass, _searState.colorProgram, draws, _searState.colorBundleDesc);
    }
    const encoded = beginColor(
        world,
        targets.label,
        targets.colorView,
        targets.depthView,
        view.framebuffer,
    );
    if (pass.bundle) encoded.executeBundles(pass.replay);
    encoded.end();
    // tally the indirect draws this camera issues (opaque + blend) so the profiler derives the injected
    // validation floor; the honest count is post the `if (pipe)` skip, and excludes the backdrop's
    // three-vertex draw, which is not indirect
    world.gpu.indirect?.("sear:color", indirect);
}

// the StandardRenderer camera query terms, and the point caster frames `ShadowCameraSystem` ranks into (a capacity pool
// `updatePointShadows` grows and rewrites in place)
const SEAR_CAMERAS = [Camera, StandardRenderer];

/**
 * compile the forward pipelines for every registered surface, sharing one shader module: a 4× MSAA
 * single-target color pipeline (its own depth, `less` + write) that writes shaded color resolved into
 * the offscreen framebuffer, a 1× tag pipeline (its own single-sample depth, `less` + write, single
 * `r32uint` target) that stamps the front-most fragment's surface tag into `view.pickingId`, and a 1× depth
 * pipeline (position-only, the shadow map renders through it). Color is one camera-independent shape
 * across opaque / `clip` / `alpha`: no MRT; the tag is its own single-sample lane. Color samples the
 * sun shadow inline (group 1 = the map + comparison sampler + light params); the tag + depth pipelines
 * omit group 1. StandardRenderer declares the vertex-pull bindings itself; each draw selects its mesh via
 * `Draw.mesh`. Uniform across surfaces: no "MeshInstance-shaped" detection. Also (re)creates the sun-shadow
 * GPU resources sear owns (the comparison sampler, the 1×1 fallback, the group-1 layout, and the real
 * params buffer — `./atlas`), surviving HMR re-warms
 */
async function prepareSear(world: World, device: GPUDevice, capacity: number): Promise<void> {
    // the caster count + atlas size fold into the shadow WGSL at its first resolve and the uniforms below
    // size from the same schemas, so a config mutated between builds is a hard error, not a silent mismatch
    checkShadowConfig();
    resetPipelineCaches(world);
    world.resource(searStateKey).warned.clear();
    resetShadowAtlas(world, device);
    // the lazily-allocated packed list binds at each atlas pipeline's `eids` lane, so allocating it clears
    // the resolved-bind-group cache to rebuild with it
    world.resource(pointRegather).reset(() => clearGroups(world));
    world.resource(cascadeRegather).reset(() => clearGroups(world));
    // Compile surfaces and the shared re-gather pipelines before the first draw.
    await Promise.all([
        prepareRegather(world, device, capacity),
        preparePipelines(world, capacity),
    ]);
}

/**
 * sear's geometry-emit ordering anchor **and** prepass, per camera carrying a lane marker
 * ({@link PickingPrepass} / {@link DepthPrepass}). It collapses the old empty depth anchor + the tag pass into one
 * single-sample pass that emits the camera's opt-in lanes (the id lane → `view.pickingId`, the depth lane →
 * `view.depth`), the shape Bevy's prepass takes. It's also the **anchor**: a producer whose per-frame
 * compute writes the geometry sear reads (vertices / indices, or an instanced surface's `globalTransforms` /
 * `eids`) declares `before: [RenderPrepassesSystem]` so its emit precedes every geometry-reading pass (the
 * prepass, the shadow map, and the color pass all read it within the frame; an emit landing between them
 * would desync the reads). It runs first among the geometry passes (`after: [BeginFrameSystem]`), so it
 * resolves the frame's draw list **once** into `_sear.frameDraws` for the shadow map + color pass to share. A
 * screen-space effect still slots into the `after: [RenderPrepassesSystem], before: [RenderMeshColorSystem]` seam. A camera
 * carrying no lane marker runs no prepass (the bare path), but the anchor + resolve still run
 */
export const RenderPrepassesSystem: System = {
    name: "prepass",
    group: "draw",
    after: [BeginFrameSystem],
    update(world) {
        const _searState = world.resource(searStateKey);

        if (!world.resource(Render).encoder) return;
        // resolve once for the prepass + shadow map + color pass (they all run after this)
        resolveDraws(world, world.entityHighWater);
        for (const eid of world.query(SEAR_CAMERAS)) {
            const view = world.resource(Views).get(eid);
            if (!view?.framebuffer) continue;
            // markers → the requested lanes. The id lane is a color attachment; depth is the
            // depth-stencil, stored only when the camera carries `DepthPrepass`. Reset both each frame so a
            // camera that drops a marker stops publishing its lane
            view.pickingId = null;
            view.depth = null;
            const storeDepth = world.has(eid, DepthPrepass);
            let marked = false;
            for (let l = 0; l < COLOR_LANES.length; l++) {
                if (world.has(eid, COLOR_LANES[l].marker)) {
                    marked = true;
                    break;
                }
            }
            if (!marked && !storeDepth) continue; // no lane requested — bare path
            const lanes: ColorLane[] = [];
            for (let l = 0; l < COLOR_LANES.length; l++) {
                if (world.has(eid, COLOR_LANES[l].marker)) lanes.push(COLOR_LANES[l]);
            }
            renderPrepass(
                world,
                eid,
                view,
                _searState.frameDraws,
                _searState.frameCount,
                lanes,
                storeDepth,
            );
        }
    },
};

/**
 * sear's geometry pass, per camera: shades every opaque draw then composites every `blend` draw over
 * them in one 4× MSAA pass (its own 4× color + depth), resolved into the offscreen once. Binds the sun
 * shadow seam (group 1): sear's own shadow map + light params it samples inline, or its fallback (fully
 * lit) when no light casts. Renders the shared `_sear.frameDraws` (resolved once by {@link RenderPrepassesSystem}).
 * Runs after every screen-space effect ordered `before: [RenderMeshColorSystem]`; `before: [GlazeSystem]` makes it
 * sear's terminal offscreen write, so glaze reads `view.framebuffer` only after the resolve lands (glaze
 * never imports sear)
 */
export const RenderMeshColorSystem: System = {
    name: "color",
    group: "draw",
    after: [RenderPrepassesSystem],
    before: [GlazeSystem, OverlaySystem],
    update(world) {
        const _searState = world.resource(searStateKey);

        if (!world.resource(Render).encoder) return;
        for (const eid of world.query(SEAR_CAMERAS)) {
            const view = world.resource(Views).get(eid);
            if (!view?.framebuffer) continue;
            renderColor(
                world,
                eid,
                view,
                _searState.frameDraws,
                _searState.frameCount,
                backdrop(world, eid),
            );
        }
    },
};

/**
 * pose the sun's CSM cascade cameras + the point/spot combo cameras from the casting lights + the main StandardRenderer
 * camera, so `BeginFrameSystem` packs their viewProjs this frame and the MeshInstance pack culls casters into each
 * slot as one more view (the unified culled-combo spine). `simulation` group, before the draw frame opens.
 * No-op for the sun when no directional light carries a {@link Shadow} (the zero-cost off path): the atlas
 * pass is skipped and sear falls back to fully lit
 */
const ShadowCameraSystem: System = {
    name: "shadow-camera",
    group: "simulation",
    update(world) {
        const _searState = world.resource(searStateKey);

        let main = -1;
        for (const eid of world.query(SEAR_CAMERAS)) {
            main = eid;
            break;
        }
        const casters = updatePointShadows(world, main, _searState.pointFrames);
        setPointFrames(world, _searState.pointFrames, casters);
        updateCascades(world, main);
        // allocate each atlas's re-gather list here, before record() (RenderPrepassesSystem) builds the cast bind
        // groups that bind it — so the first casting frame's groups include it (the alloc clears the
        // resolved-bind-group cache), no one-frame delay. Idempotent once allocated; the render fns call it
        // again harmlessly
        if (casters > 0 && shadowReady(world))
            world.resource(pointRegather).ensure(pointCasters() * 6, world.entityHighWater);
        if (cascadeCount(world) > 0 && shadowReady(world))
            world.resource(cascadeRegather).ensure(MAX_CASCADES, world.entityHighWater);
    },
};

/**
 * render the casters' depth into the shadow atlases (the point/spot tiles + the CSM cascades) and publish the
 * seams for sear's color pass to sample inline. `after: [RenderPrepassesSystem]` so every position-writing producer
 * (pinned before the anchor) has emitted and `_sear.frameDraws` is resolved; `before: [RenderMeshColorSystem]` so the
 * atlases + seams are ready before sear shades. No casting light → no pass, sear falls back to fully lit.
 * Bevy's shape: the shadow maps are light-data-gated, sampled inline, no separate shadow plugin
 */
const ShadowMapSystem: System = {
    name: "shadowmap",
    group: "draw",
    after: [RenderPrepassesSystem],
    before: [RenderMeshColorSystem],
    update(world) {
        const _searState = world.resource(searStateKey);

        renderPointShadows(
            world,
            _searState.frameDraws,
            _searState.frameCount,
            world.entityHighWater,
        );
        renderCascades(world, _searState.frameDraws, _searState.frameCount, world.entityHighWater);
    },
};

// the typed twin of `litBindings`, group 2 (`layout()`'s $idx(2) synthesis) — same four bindings, same
// element shapes, feeding the typed `default` surface.
const typedDefaultLayout = typedLayout({
    eids: { type: "storage", element: d.vec4u },
    globalTransforms: { type: "storage", element: Xform },
});

// The dense MeshInstance record carries linear base color and `(metallic, roughness, emissive, occlusion)`.
// `litPbr` (`sear/engine.ts`) reads the fs-scaffold privates the typed pipeline
// builder (`pipelines.ts`) fills before calling this.
const typedDefaultFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const albedo = ctx.color.xyz;
    const pbr = Pbr({
        albedo,
        metallic: ctx.material.x,
        roughness: ctx.material.y,
        occlusion: ctx.material.w,
        dielectric: 0,
    });
    const emissive = std.mul(albedo, ctx.material.z);
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), 1);
});

// the typed twin of `colorBindings` — `unlit`'s three bindings, no `material` (it never shades).
const typedColorLayout = typedLayout({
    eids: { type: "storage", element: d.vec4u },
    globalTransforms: { type: "storage", element: Xform },
});

// The unlit surface reads the same linear color carried in the dense MeshInstance record.
const typedUnlitFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.color.xyz, 1);
});

// the typed twin of the raw `vertex` surface (per-vertex Gouraud): `litColor` crosses vs→fs as a custom
// varying through the `typedVaryingVs`/`typedVaryingFs` copier pair (`pipelines.ts`), so this `vs` runs
// `litPbr` once per vertex. `sunVisibility`/`pointScale`/`fragWorld` sit at their defaults here (per-vertex
// shading runs before the fs scaffold fills them) — the same fully-lit sun / zero-point-contribution the
// raw path's per-vertex mode gets.
const typedVertexVaryings = { litColor: d.vec3f };
const typedVertexPatch = vsPatchSchema(typedVertexVaryings);
const typedVertexVs = tgpu.fn(
    [VsIn],
    typedVertexPatch,
)((vsIn) => {
    "use gpu";
    const albedo = vsIn.color.xyz;
    const pbr = Pbr({
        albedo,
        metallic: vsIn.material.x,
        roughness: vsIn.material.y,
        occlusion: vsIn.material.w,
        dielectric: 0,
    });
    const emissive = std.mul(albedo, vsIn.material.z);
    const litColor = std.add(
        litPbr(pbr, std.normalize(vsIn.worldNormal), vsIn.world.xyz),
        emissive,
    );
    return typedVertexPatch({
        world: vsIn.world,
        worldNormal: vsIn.worldNormal,
        clip: d.vec4f(0),
        litColor,
    });
});
const typedVertexFs = tgpu.fn(
    [fsCtxSchema(typedVertexVaryings)],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.litColor, 1);
});

// free every GPU resource sear owns (at plugin dispose): the shadow atlases (point + cascade, ./atlas) +
// their params, and the per-camera prepass depth / lane targets / MSAA color+depth. The cascade Camera
// entities live in a World, so destroyCascades (./shadows) tears those down separately
function disposeSear(world: World): void {
    const _searState = world.resource(searStateKey);

    disposeShadowAtlas(world);
    for (const c of _searState.depth.values()) c.texture.destroy();
    for (const c of _searState.laneTargets.values()) c.texture.destroy();
    for (const c of _searState.colorTargets.values()) {
        c.color?.destroy();
        c.depth.destroy();
    }
    _searState.depth.clear();
    _searState.laneTargets.clear();
    _searState.colorTargets.clear();
}

export function createSearPlugin(): Plugin {
    return {
        name: "StandardRendering",
        components: [
            registration("StandardRenderer", StandardRenderer),
            registration("PickingPrepass", PickingPrepass),
            registration("DepthPrepass", DepthPrepass),
            registration("Shadow", Shadow, { defaults: () => ({ ...SHADOW_DEFAULTS }) }),
            registration("Material", Material, MaterialTraits),
            registration("CameraBackground", CameraBackground),
        ],
        systems: [
            RenderPrepassesSystem,
            RenderMeshColorSystem,
            ShadowCameraSystem,
            ShadowMapSystem,
        ],
        dependencies: [RenderingPlugin, MeshPlugin],

        // sear's default materials, shading per-instance `color` + `material` at three lighting modes. They
        // ship with the renderer, not MeshInstance: MeshInstance publishes the data (`eids` + `color`), sear adds its own
        // `Material` slab and shades with its metallic-roughness `litPbr`. `Material.surface` defaults to
        // "default" (per-pixel); "vertex" (per-vertex Gouraud) and "unlit" are picked per-MeshInstance. The instance
        // transform is sear's convention — these declare the bindings and omit a transform vs chunk. `pbr()`
        // builds the Pbr struct from the packed `material` lanes; the engine default has no specular until a
        // Material sets metallic > 0 (dielectric 0), so a bare MeshInstance shades exactly like the pre-PBR diffuse.
        initialize(world) {
            initializeSurfaceState(world);
            initializeDrawState(world);
            world.resource(Surfaces).clear();
            world.resource(Backgrounds).clear();
            world.resource(Draws).clear();
            world.resource(searStateKey);
            initializeShadowAtlasState(world);
            initializePipelineState(world);
            initializeRegatherState(world);
            // a fresh World recreates its own off-screen shadow cameras lazily — drop any eids cached by
            // a prior build so this re-run never aliases recycled entities (the module-scope contract)
            resetPointShadows(world);
            resetCascades(world);
            initMaterial(world);
            // build the Pbr struct + the emissive tint (Color.rgb * the emissive strength lane) from the
            // f16 material lanes: word x holds (metallic, roughness), word y (emissive, occlusion), each
            // unpacked to f32 for the shading math. emissive is an unbounded HDR glow strength;
            // dielectric 0 → metallic 0 is specular-free (the flat shallot default)
            // The three built-ins share the same schema-backed instance/material layout.
            typedRegister(world, {
                name: "default",
                layout: typedDefaultLayout,
                fs: typedDefaultFs,
            });
            // the typed twin — the varyings mechanism's first live consumer (`litColor` crosses vs→fs
            // through `typedVaryingVs`/`typedVaryingFs`'s per-surface copier, `pipelines.ts`); drawn typed
            // in every pass, like `default` above.
            typedRegister(world, {
                name: "vertex",
                layout: typedDefaultLayout,
                varyings: typedVertexVaryings,
                vs: typedVertexVs,
                fs: typedVertexFs,
            });
            // the typed twin, drawn typed in every pass like `default` above.
            typedRegister(world, {
                name: "unlit",
                layout: typedColorLayout,
                fs: typedUnlitFs,
            });
        },

        async warm(world) {
            if (!world.gpu.device) return;
            await prepareSear(world, world.gpu.device, world.entityHighWater);
        },

        dispose(world) {
            destroyPointShadows(world);
            destroyCascades(world);
            disposeSear(world);
        },
    };
}
