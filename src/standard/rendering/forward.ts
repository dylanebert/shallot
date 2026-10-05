import {
    CorePipelinePlugin,
    MainPassSystem,
    PrepassSystem,
    RenderPhases,
} from "../../core/rendering";
import { component } from "../../engine";
import {
    CullLightsSystem,
    initializeClusterState,
    UpdateLightClustersSystem,
    warmClusters,
    warmLightCull,
} from "./cluster";
import { initializeSurfaceState } from "./contract";
import {
    initializeLightingState,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    writeLighting,
} from "./lighting";
import { initializeDrawState } from "./registry";
// StandardRenderer — the one shallot renderer. A GPU-driven raster *forward* pass (Aaltonen-Haar / niagara
// submission spine, primary visibility only) with sun shadows sampled inline in the FS, matching Bevy's
// clustered-forward shape. One renderer, one plugin (`StandardRenderingPlugin`), no layers behind seams: one color
// pass (opaque draws then `blend` draws composited over them in a single `beginRenderPass`), an
// opt-in single-sample depth prepass (`DepthPrepass`), and sun shadows (shadowMapsEnabled on a directional
// light) are gated by camera and light data; core owns the view targets and depth marker —
// not composed plugins coordinating through a singleton.
//
// Sun shadows: the CPU/ECS half (the off-screen light camera + placement) lives in ./shadows; the GPU half
// (the shadow map, its render through standard's compiled prepass depth pipelines, and the group-1 binding the
// FS samples) lives in ./atlas. This file owns the WGSL-scaffold-agnostic renderer plumbing: components +
// registries, per-draw bind-group resolution, pass opening, the systems, and the plugin — the shading
// functions live in ./shade, pipeline compilation in ./pipelines. StandardRenderer renders its own map and reads its
// own shadow state directly — nothing publishes into it. Enable the sun's shadowMapsEnabled to cast; disable it for the
// fully-lit bare path (no map allocated), exactly like a camera without a lane marker runs no prepass.

import type { TgpuBindGroupLayout, TgpuBuffer, TgpuRenderPipeline } from "typegpu";
import tgpu, { isBuffer, isUsableAsStorage, isUsableAsUniform } from "typegpu";
import type { AnyData } from "typegpu/data";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { type MeshBinding, Meshes, type MeshIndex, MeshPlugin } from "../../core/mesh";
import type { View } from "../../core/rendering";
import {
    BeginFrameSystem,
    Camera,
    DEPTH_FORMAT,
    RenderContext,
    SAMPLE_COUNT,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { u32 } from "../../engine";
import { Xform } from "../../engine/utils";
import {
    cascadeRegather,
    disposeShadowAtlas,
    ensureCascadeAtlas,
    ensurePointAtlas,
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
    type Background,
    Backgrounds,
    fsCtxSchema,
    registerSurface,
    type Surface,
    Surfaces,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "./contract";
import { engineLayout, litPbr } from "./engine";
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
import { Pbr } from "./shade";
import {
    cascadeCount,
    destroyCascades,
    destroyPointShadows,
    MAX_CASCADES,
    type PointShadowFrame,
    pointCasters,
    resetCascades,
    resetPointShadows,
    updateCascades,
    updatePointShadows,
} from "./shadows";

interface StandardRendererState {
    warned: Set<string>;
    frameDraws: FrameDraw[];
    frameCount: number;
    colorBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] };
    colorBundles: Map<number, PassBundle>;
    colorProgram: BundleDraw[];
    prepassBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] };
    prepassBundles: Map<number, PassBundle>;
    prepassProgram: BundleDraw[];
    pointFrames: PointShadowFrame[];
}

const standardRendererStateKey = { create: createStandardRendererState };

function createStandardRendererState(): StandardRendererState {
    return {
        warned: new Set(),
        frameDraws: [],
        frameCount: 0,
        colorBundleDesc: {
            label: "standard-color",
            colorFormats: [],
            depthStencilFormat: DEPTH_FORMAT,
            sampleCount: 1,
        },
        colorBundles: new Map(),
        colorProgram: [],
        prepassBundleDesc: {
            label: "standard-prepass",
            colorFormats: [],
            depthStencilFormat: DEPTH_FORMAT,
            sampleCount: 1,
        },
        prepassBundles: new Map(),
        prepassProgram: [],
        pointFrames: [],
    };
}

/**
 * marker selecting StandardRenderer as the active renderer on a Camera entity. A camera carrying it renders through
 * standard's opaque and transparent records, plus core's opt-in depth prepass.
 */
export const StandardRenderer = component("StandardRenderer", {});

/**
 * select a StandardRenderer camera's backdrop: the {@link Backgrounds} recipe drawn behind the scene as a fullscreen
 * view-ray → color fill on the un-rendered pixels. Without it the camera shows the flat `Camera.clearColor`
 * (the opt-in fallback). The recipe is registered in code with `registerBackground`; this picks one per
 * camera by name.
 */
export const CameraBackground = component("CameraBackground", {
    /** the {@link Backgrounds} id drawn behind the scene */
    name: u32,
});

// a draw resolving to null is a silent skip — usually a typo'd binding or an
// unpublished resource. Warn once per draw so it's visible without spamming

function warnSkip(world: World, draw: string, cause: string): null {
    const _standardRendererState = world.resource(standardRendererStateKey);

    if (!_standardRendererState.warned.has(draw)) {
        _standardRendererState.warned.add(draw);
        console.warn(`standard: draw "${draw}" skipped — ${cause}`);
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
 * the color, transparent and depth prepass pipelines and the bind-group state standard records a draw
 * with, or null to skip it. All pipelines share one bind group (same group-0 layout). A surface with
 * no compiled pipeline isn't standard's (silent skip); a missing mesh or unpublished binding warns once.
 * The per-slot bind groups cache per draw, rebuilt only on a resource identity change; the fixed uniforms
 * are stable, so untracked
 */
function record(world: World, draw: Draw, capacity: number): FrameDraw | null {
    const surface = world.resource(Surfaces).get(draw.surface);
    return surface ? recordSurface(world, draw, surface, capacity) : null;
}

// resolve a layout's own bindings (never the standard-injected `vertices`) to live resources by the
// entry's kind. Returns the createBindGroup value record + the identity list + each binding's name and the
// registry it resolved from, or the missing binding's name
function layoutResources(
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
        const live =
            name in g.owner.layout.attributes
                ? mesh.attributes?.[name]
                : (override?.[name] ?? g.registries[i].get(name));
        if (live !== res[k]) return false;
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
 * {@link record}'s surface path: compiled pipelines + the per-draw group-2 state cached by
 * layout name — `color` against `layout`; opaque depth-side groups against `layout.depthVariant`; clip
 * depth-side groups against the full layout so cutoff sees material UVs — plus the atlas `eids` swaps
 * used by the atlas passes. Their slot-0 engine group resolves through the same live cache as each view.
 */
function recordSurface(
    world: World,
    draw: Draw,
    surface: Surface,
    capacity: number,
): FrameDraw | null {
    const mesh = world.resource(Meshes).get(draw.mesh);
    if (!mesh) return warnSkip(world, draw.name, `mesh "${draw.mesh}" not registered`);
    if (!mesh.position || !mesh.quant) {
        // registerMesh staged it after this frame's pack; it draws from the next frame
        if (mesh.pending) return null;
        return warnSkip(
            world,
            draw.name,
            `mesh "${draw.mesh}" has no quantized position/quant stream`,
        );
    }
    const prev = getGroup(world, draw.name, surface);
    let t = prev?.item.r.t ?? getCompiledSurface(world, surface.name);
    if (!t || t.owner !== surface || t.layout !== surface.layout) {
        // registered after warm (`preparePipelines` compiles the rest) — sync, so no skip frame; a
        // throwing compile (a contract guard, or shader/device validation) must not take down the frame
        // loop, so it degrades to the warn-once skip
        try {
            t = compileSurface(world, surface, capacity);
        } catch (e) {
            return warnSkip(world, draw.name, `surface "${surface.name}" failed to compile: ${e}`);
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

    const overrides = { ...mesh.bindings } as Record<string, MeshBinding>;
    for (const [name, element] of Object.entries(surface.layout.attributes)) {
        const stream = mesh.attributes?.[name];
        if (!stream || !d.deepEqual(stream.dataType.elementType, element))
            return warnSkip(
                world,
                draw.name,
                `mesh "${mesh.name}" ${stream ? "has a different schema for" : "has no"} attribute "${name}" that surface "${surface.name}" reads`,
            );
        overrides[name] = stream;
    }
    const resolved = layoutResources(
        world,
        surface.layout.entries as Record<string, object>,
        overrides,
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

// the frame's resolved draws (the first `_standardRendererState.frameCount`), resolved once by ResolveDrawsSystem and shared across
// the prepass, shadow atlases, and color pass — they all draw the same resolved records, so resolving
// per pass would repeat the work

/**
 * the frame's draw list: every registered {@link Draw} with a compiled surface + published
 * bindings, paired with its cached group-0 state. Camera-independent (the per-slot bind groups it builds
 * against are cached lazily by slot, not baked per camera), so {@link PrepassSystem}
 * resolves it once per frame into `_standardRendererState.frameDraws` and the prepass, shadow map, and color pass all
 * render every camera against that one list
 */
function resolveDraws(world: World, capacity: number): void {
    world.resource(standardRendererStateKey).frameCount = 0;
    world.resource(Draws).forEach((draw) => {
        resolveDraw(world, draw, capacity);
    });
}

function resolveDraw(world: World, draw: Draw, capacity: number): void {
    const _standardRendererState = world.resource(standardRendererStateKey);

    const item = record(world, draw, capacity);
    if (item) _standardRendererState.frameDraws[_standardRendererState.frameCount++] = item;
}

/** Records opaque and clipped surfaces into core's single-sample depth prepass.
 * Alpha surfaces write no depth; an empty draw list still clears the depth target. */
function renderPrepass(
    world: World,
    eid: number,
    view: View,
    items: FrameDraw[],
    count: number,
    pass: GPURenderPassEncoder,
): void {
    const _render = world.resource(RenderContext);
    const _standardRendererState = world.resource(standardRendererStateKey);

    if (!_render.encoder || !view.framebuffer) return;

    let draws = 0;
    const shadow = shadowGroup(world);
    for (let i = 0; i < count; i++) {
        const { draw, r } = items[i];
        const pipe = r.t.prepass;
        const group = r.g.depth;
        if (pipe && group) {
            const step = bundleDraw(_standardRendererState.prepassProgram, draws);
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
    // Each camera's view slot selects its own indirect range and engine group.
    let bundle = _standardRendererState.prepassBundles.get(eid);
    if (!bundle) {
        bundle = newPassBundle();
        _standardRendererState.prepassBundles.set(eid, bundle);
    }
    if (
        bundleChanged(
            bundle,
            _standardRendererState.prepassProgram,
            draws,
            _standardRendererState.prepassBundleDesc,
        )
    ) {
        recordBundle(
            world,
            bundle,
            _standardRendererState.prepassProgram,
            draws,
            _standardRendererState.prepassBundleDesc,
        );
    }
    if (bundle.bundle) pass.executeBundles(bundle.replay);
    world.gpu.indirect?.("standard:prepass", draws);
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

// build (and cache on the CompiledBackground) a background's own group-2 bind group — slot-invariant
// (the per-slot View rides the engine group 0). Returns null while a binding is unpublished (skip); a
// binding-free background carries no group at all (its empty layout never enters the pipeline layout)
function backgroundGroup(
    world: World,
    bg: Background,
    ct: CompiledBackground,
): GPUBindGroup | null | "none" {
    const entries = bg.layout.entries as Record<string, object>;
    if (Object.keys(entries).length === 0) return "none";
    const resolved = layoutResources(world, entries);
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

/**
 * Records standard's geometry bundles into core's main pass: shades every opaque draw,
 * then composites every `blend` draw over them (`less-equal` depth-tested against the opaque depth,
 * depth-write off) in core's targets: one HDR color target, no MRT,
 * because each extra target costs bandwidth on every pixel and tile-based GPUs pay it hardest. With
 * `Camera.antialias` on (the default) it's a 4× MSAA pass resolved into the offscreen; off, it renders
 * single-sample straight into the offscreen (and binds the surfaces' single-sample pipeline twins,
 * compiled lazily by {@link ensureSingle}). Opaque and transparent share one `beginRenderPass` (nothing
 * reads the color between them, so they fuse into one tile round-trip). Group 1 is the sun shadow seam:
 * standard's own shadow map + light params, or its 1×1 fallback (fully lit) when no light casts. An empty
 * draw list still clears the framebuffer. `bg` (the camera's {@link CameraBackground} selection) draws a fullscreen
 * backdrop between the opaque and blend draws: masked to far-plane pixels by the depth test, so geometry
 * overdraws it and blended draws composite over it; null leaves the flat clear color as the only backdrop
 */
function renderColor(
    world: World,
    eid: number,
    view: View,
    items: FrameDraw[],
    count: number,
    encoded: GPURenderPassEncoder,
    transparent: boolean,
    bg: BackdropPick | null = null,
): void {
    const _render = world.resource(RenderContext);
    const _standardRendererState = world.resource(standardRendererStateKey);

    if (!_render.encoder || !view.framebuffer) return;
    // per-camera AA: 4× MSAA when `Camera.antialias` is on (the Camera component default), else
    // single-sample. `world.storage(Camera).antialias.set(eid, 0)` flips it live
    const aa = world.storage(Camera).antialias.get(eid) !== 0;

    const shadow = shadowGroup(world);
    // Each phase has its own bundle: opaque plus backdrop, or blend. Building it reads only cached
    // identities, so a steady frame allocates nothing here and the compare below reports no transition
    let draws = 0;
    let indirect = 0;
    for (let i = 0; i < count; i++) {
        const item = items[i];
        if (!aa) ensureSingle(world, item.r.t);
        const pipe = !transparent ? (aa ? item.r.t.color : item.r.t.single?.color) : null;
        if (pipe) {
            drawColor(
                world,
                _standardRendererState.colorProgram,
                draws++,
                item,
                pipe,
                view.slot,
                shadow,
            );
            indirect++;
        }
    }
    // the backdrop: a fullscreen triangle at the far plane, after opaque (the depth test masks it to
    // un-rendered pixels) and before blend (so transparent draws composite over it). The bg pipeline
    // carries the shadow group 1 in its layout (unused) like every color pipeline, so the group bound at
    // the pass top survives the switch for the blend draws after it
    if (bg && !transparent) {
        // a backdrop: the shared engine group 0 (a never-read `bgQuant()` fills the meshQuant
        // slot — a background pulls no mesh), the shadow group 1 (declared-but-unused, the
        // group-count-compatibility reason `compileBackground` documents), and its own group 2
        const group = backgroundGroup(world, bg.bg, bg.ct);
        if (group) {
            const step = bundleDraw(_standardRendererState.colorProgram, draws++);
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
        const pipe = transparent
            ? aa
                ? item.r.t.transparent
                : item.r.t.single?.transparent
            : null;
        if (pipe) {
            drawColor(
                world,
                _standardRendererState.colorProgram,
                draws++,
                item,
                pipe,
                view.slot,
                shadow,
            );
            indirect++;
        }
    }

    const bundleKey = eid * 2 + Number(transparent);
    let pass = _standardRendererState.colorBundles.get(bundleKey);
    if (!pass) {
        pass = newPassBundle();
        _standardRendererState.colorBundles.set(bundleKey, pass);
    }
    _standardRendererState.colorBundleDesc.colorFormats[0] = _render.format;
    _standardRendererState.colorBundleDesc.sampleCount = aa ? SAMPLE_COUNT : 1;
    if (
        bundleChanged(
            pass,
            _standardRendererState.colorProgram,
            draws,
            _standardRendererState.colorBundleDesc,
        )
    ) {
        recordBundle(
            world,
            pass,
            _standardRendererState.colorProgram,
            draws,
            _standardRendererState.colorBundleDesc,
        );
    }

    if (pass.bundle) encoded.executeBundles(pass.replay);

    // tally the indirect draws this camera issues (opaque + blend) so the profiler derives the injected
    // validation floor; the honest count is post the `if (pipe)` skip, and excludes the backdrop's
    // three-vertex draw, which is not indirect
    world.gpu.indirect?.("standard:color", indirect);
}

// the StandardRenderer camera query terms, and the point caster frames `ShadowCameraSystem` ranks into (a capacity pool
// `updatePointShadows` grows and rewrites in place)
const STANDARD_RENDERER_CAMERAS = [Camera, StandardRenderer];

/**
 * compile the forward pipelines for every registered surface, sharing one shader module: a 4× MSAA
 * single-target color pipeline (its own depth, `less` + write) that writes shaded color resolved into
 * the offscreen framebuffer, and a 1× depth pipeline (position-only except for clipped surfaces).
 * Color is one camera-independent shape across opaque / `clip` / `alpha`, with no MRT, and samples the
 * sun shadow inline (group 1 = the map + comparison sampler + light params). StandardRenderer declares the vertex-pull bindings itself; each draw selects its mesh via
 * `Draw.mesh`. Uniform across surfaces: no "MeshInstance-shaped" detection. Also (re)creates the sun-shadow
 * GPU resources standard owns (the comparison sampler, the 1×1 fallback, the group-1 layout, and the real
 * params buffer — `./atlas`), surviving HMR re-warms
 */
async function prepareStandardRenderer(
    world: World,
    device: GPUDevice,
    capacity: number,
): Promise<void> {
    resetPipelineCaches(world);
    world.resource(standardRendererStateKey).warned.clear();
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

// Resolve once before core's prepass; the prepass, shadow maps and main pass share this draw list.

const ResolveDrawsSystem: System = {
    group: "draw",
    after: [BeginFrameSystem, CullLightsSystem],
    before: [PrepassSystem],
    update(world) {
        if (world.resource(RenderContext).encoder) resolveDraws(world, world.entityHighWater);
    },
};

/**
 * pose the sun's CSM cascade cameras + the point/spot combo cameras from the casting lights + the main StandardRenderer
 * camera, so `BeginFrameSystem` packs their viewProjs this frame and the MeshInstance pack culls casters into each
 * slot as one more view (the unified culled-combo spine). `simulation` group, before the draw frame opens.
 * No-op for the sun when shadowMapsEnabled is off (the zero-cost off path): the atlas
 * pass is skipped and standard falls back to fully lit
 */
const ShadowCameraSystem: System = {
    name: "shadow-camera",
    group: "simulation",
    update(world) {
        const _standardRendererState = world.resource(standardRendererStateKey);

        let main = -1;
        for (const eid of world.query(STANDARD_RENDERER_CAMERAS)) {
            main = eid;
            break;
        }
        const casters = updatePointShadows(world, main, _standardRendererState.pointFrames);
        setPointFrames(world, _standardRendererState.pointFrames, casters);
        updateCascades(world, main);
        // allocate each atlas's re-gather list here, before record() (PrepassSystem) builds the cast bind
        // groups that bind it — so the first casting frame's groups include it (the alloc clears the
        // resolved-bind-group cache), no one-frame delay. Idempotent once allocated; the render fns call it
        // again harmlessly. Both atlases (re)allocate here too, before the prepass binds the shadow group
        if (casters > 0 && shadowReady(world)) {
            world.resource(pointRegather).ensure(pointCasters(world) * 6, world.entityHighWater);
            ensurePointAtlas(world);
        }
        if (cascadeCount(world) > 0 && shadowReady(world)) {
            world.resource(cascadeRegather).ensure(MAX_CASCADES, world.entityHighWater);
            ensureCascadeAtlas(world);
        }
    },
};

/**
 * render the casters' depth into the shadow atlases (the point/spot tiles + the CSM cascades) and publish the
 * seams for standard's color pass to sample inline. `after: [PrepassSystem]` so every position-writing producer
 * (pinned before the anchor) has emitted and `_standardRendererState.frameDraws` is resolved; `before: [MainPassSystem]` so the
 * atlases + seams are ready before standard shades. No casting light → no pass, standard falls back to fully lit.
 * Bevy's shape: the shadow maps are light-data-gated, sampled inline, no separate shadow plugin
 */
const ShadowMapSystem: System = {
    name: "shadowmap",
    group: "draw",
    after: [PrepassSystem],
    before: [MainPassSystem],
    update(world) {
        const _standardRendererState = world.resource(standardRendererStateKey);

        renderPointShadows(
            world,
            _standardRendererState.frameDraws,
            _standardRendererState.frameCount,
            world.entityHighWater,
        );
        renderCascades(
            world,
            _standardRendererState.frameDraws,
            _standardRendererState.frameCount,
            world.entityHighWater,
        );
    },
};

// the `default` and `vertex` surfaces' group 2 (`layout()`'s $idx(2) synthesis): `eids`, the
// per-instance `vec4u` rows, and `globalTransforms`.
const defaultSurfaceLayout = surfaceLayout({
    eids: { type: "storage", element: d.vec4u },
    globalTransforms: { type: "storage", element: Xform },
});

// The shader scaffold resolves each MeshInstance's material id to linear base color and material lanes.
// `litPbr` (`standard/engine.ts`) reads the fs-scaffold privates the pipeline
// builder (`pipelines.ts`) fills before calling this.
const defaultSurfaceFs = tgpu.fn(
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
        diffuseWrap: engineLayout.$.materials[d.u32(ctx.material.z)].diffuseWrap,
    });
    const emissive = engineLayout.$.materials[d.u32(ctx.material.z)].emissive;
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), 1);
});

// the `unlit` surface's group 2: the same bindings as `default`'s, read without shading.
const unlitSurfaceLayout = surfaceLayout({
    eids: { type: "storage", element: d.vec4u },
    globalTransforms: { type: "storage", element: Xform },
});

// The unlit surface reads the resolved material's linear base color.
const unlitSurfaceFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.color.xyz, 1);
});

// the `vertex` surface (per-vertex Gouraud): `litColor` crosses vs→fs as a custom
// varying through the `varyingVs`/`varyingFs` copier pair (`pipelines.ts`), so this `vs` runs
// `litPbr` once per vertex. `sunVisibility`/`pointScale`/`fragWorld` sit at their defaults here (per-vertex
// shading runs before the fs scaffold fills them), so it shades with a fully-lit sun and no point
// contribution.
const vertexSurfaceVaryings = { litColor: d.vec3f };
const vertexSurfacePatch = vsPatchSchema(vertexSurfaceVaryings);
const vertexSurfaceVs = tgpu.fn(
    [VsIn],
    vertexSurfacePatch,
)((vsIn) => {
    "use gpu";
    const albedo = vsIn.color.xyz;
    const pbr = Pbr({
        albedo,
        metallic: vsIn.material.x,
        roughness: vsIn.material.y,
        occlusion: vsIn.material.w,
        dielectric: 0,
        diffuseWrap: engineLayout.$.materials[d.u32(vsIn.material.z)].diffuseWrap,
    });
    const emissive = engineLayout.$.materials[d.u32(vsIn.material.z)].emissive;
    const litColor = std.add(
        litPbr(pbr, std.normalize(vsIn.worldNormal), vsIn.world.xyz),
        emissive,
    );
    return vertexSurfacePatch({
        world: vsIn.world,
        worldNormal: vsIn.worldNormal,
        clip: d.vec4f(0),
        litColor,
    });
});
const vertexSurfaceFs = tgpu.fn(
    [fsCtxSchema(vertexSurfaceVaryings)],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.litColor, 1);
});

// Standard owns shadow atlases and their params; core releases view targets.
// destroyCascades tears down the off-screen Camera entities separately.
function disposeStandardRenderer(world: World): void {
    disposeShadowAtlas(world);
}

const PackLightingSystem: System = {
    group: "draw",
    after: [BeginFrameSystem],
    before: [UpdateLightClustersSystem],
    update: writeLighting,
};

/** Clustered forward mesh renderer recording into core's prepass and main pass.
 * Includes CorePipelinePlugin; cameras opt in with StandardRenderer. Shadows are
 * sampled inline, and presentation follows the color resolve in core's tonemapping pass.
 */
export const StandardRenderingPlugin: Plugin = {
    name: "StandardRendering",
    components: [StandardRenderer, CameraBackground],
    systems: [
        PackLightingSystem,
        UpdateLightClustersSystem,
        CullLightsSystem,
        ResolveDrawsSystem,
        ShadowCameraSystem,
        ShadowMapSystem,
    ],
    dependencies: [CorePipelinePlugin, MeshPlugin],

    initialize(world) {
        world.resource(RenderPhases).push({
            prepass(world, eid, view, pass) {
                if (!world.has(eid, StandardRenderer)) return;
                const state = world.resource(standardRendererStateKey);
                renderPrepass(world, eid, view, state.frameDraws, state.frameCount, pass);
            },
            opaque(world, eid, view, pass) {
                if (!world.has(eid, StandardRenderer)) return;
                const state = world.resource(standardRendererStateKey);
                renderColor(
                    world,
                    eid,
                    view,
                    state.frameDraws,
                    state.frameCount,
                    pass,
                    false,
                    backdrop(world, eid),
                );
            },
            transparent(world, eid, view, pass) {
                if (!world.has(eid, StandardRenderer)) return;
                const state = world.resource(standardRendererStateKey);
                renderColor(world, eid, view, state.frameDraws, state.frameCount, pass, true);
            },
        });
        initializeClusterState(world);
        initializeLightingState(world);
        world.resource(Lighting).buffer = world.gpu.device.createBuffer({
            label: "shallot-lighting",
            size: LIGHTING_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        initializeSurfaceState(world);
        initializeDrawState(world);
        world.resource(Surfaces).clear();
        world.resource(Backgrounds).clear();
        world.resource(Draws).clear();
        world.resource(standardRendererStateKey);
        initializeShadowAtlasState(world);
        initializePipelineState(world);
        initializeRegatherState(world);
        // a fresh World recreates its own off-screen shadow cameras lazily — drop any eids cached by
        // a prior build so this re-run never aliases recycled entities (the module-scope contract)
        resetPointShadows(world);
        resetCascades(world);
        // Dielectric reflectance stays zero to preserve Shallot's specular-free diffuse default.
        registerSurface(world, {
            name: "default",
            layout: defaultSurfaceLayout,
            fs: defaultSurfaceFs,
        });
        // standard's own varyings consumer (`litColor` crosses vs→fs through
        // `varyingVs`/`varyingFs`'s per-surface copier, `pipelines.ts`).
        registerSurface(world, {
            name: "vertex",
            layout: defaultSurfaceLayout,
            varyings: vertexSurfaceVaryings,
            vs: vertexSurfaceVs,
            fs: vertexSurfaceFs,
        });
        registerSurface(world, {
            name: "unlit",
            layout: unlitSurfaceLayout,
            fs: unlitSurfaceFs,
        });
    },

    async warm(world) {
        if (!world.gpu.device) return;
        warmClusters(world);
        warmLightCull(world);
        await prepareStandardRenderer(world, world.gpu.device, world.entityHighWater);
    },

    dispose(world) {
        destroyPointShadows(world);
        destroyCascades(world);
        disposeStandardRenderer(world);
    },
};
