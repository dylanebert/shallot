import { component } from "../../engine";
// Jump-flood outlines: Ben Golus, "The Quest for Very Wide Outlines".

import type { TgpuBuffer, TgpuComputePipeline, TgpuRenderPipeline, UniformFlag } from "typegpu";
import * as d from "typegpu/data";
import {
    type Mesh,
    Meshes,
    type MeshHandle,
    MeshInstance,
    meshResourcesKey,
} from "../../core/mesh";
import {
    Camera,
    DepthPrepassRequests,
    fullscreenVertex,
    globalTransformTable,
    MainPassSystem,
    OverlaySystem,
    PresentationSystem,
    RenderContext,
    RenderingPlugin,
    renderLayers,
    sceneTransform,
    TonemappingSystem,
    type View,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { f32 } from "../../engine";
import {
    beginComputePass,
    beginRenderPass,
    dispatchWorkgroups2,
    draw,
    endComputePass,
    endRenderPass,
    precompile,
    setComputeBindGroup,
    setComputePipeline,
    setRenderBindGroup,
    setRenderPipeline,
} from "../../engine/runtime";
import { unpackColorInto } from "../../engine/utils";
import { MeshRenderPlugin, StandardRenderer } from "../../standard/rendering";
import {
    compositeHalfKernel,
    compositeKernel,
    compositeLayout,
    distanceFs,
    distanceLayout,
    jfaFs,
    jfaHalfFs,
    jfaLayout,
    jfaPassCount,
    jfaStep,
    MAX_JFA_PASSES,
    MAX_WIDTH,
    maskFragment,
    maskLayoutOcclude,
    maskLayoutOccludeMultisampled,
    maskLayoutPlain,
    maskVertex,
    reduceFs,
    reduceLayout,
    WORKGROUP,
} from "./passes";

/**
 * outline highlight: a colored band hugs the object's silhouette for hover, selection, or grab feedback.
 *
 * Add it to a MeshInstance entity to highlight it; remove it to clear. Fields are per-entity, so different
 * highlights coexist in one pass. On a camera stacked over others, the band covers the cameras below only
 * where it draws.
 */
export const Outline = component(
    "Outline",
    {
        /** Band color as 0xRRGGBB sRGB. */
        color: f32,
        /** Band thickness in device pixels, clamped to 64. */
        width: f32,
        /** 0 = always-on-top (default); 1 = occlusion-aware, hidden where the object is behind other geometry */
        occlude: f32,
    },
    {
        defaults: () => ({
            color: 0xffed7c,
            width: 4,
            occlude: 0,
        }),
    },
);

const OUTLINE_MESH_QUERY = [Outline, MeshInstance];
const OUTLINE_CAMERA_QUERY = [Camera];

// Integer pixel indices avoid half-float rounding of pixel centers, which would tint silhouette interiors.
const SEED_FORMAT: GPUTextureFormat = "rg16uint";
const ATTR_FORMAT: GPUTextureFormat = "rgba16float";
// the "no seed" sentinel the seed textures clear to: a coordinate far off-screen, so any real seed wins
// the nearest-distance test and a pixel that never reaches a seed reads a huge distance (no band)
const SENTINEL = 30000;
const SEED_CLEAR: GPUColor = { r: SENTINEL, g: SENTINEL, b: 0, a: 0 };
const ATTR_CLEAR: GPUColor = { r: 0, g: 0, b: 0, a: 0 };
const INITIAL_INSTANCES = 64;
// the final JFA slot writes seed[(MAX_JFA_PASSES) % 2] whatever the band width, so the composite reads one
// fixed seed texture
const FINAL_SEED = MAX_JFA_PASSES % 2;

type StepBuffer = TgpuBuffer<typeof d.f32> & UniformFlag;

type MaskTargets = { seed: d.Vec4u; attr: d.Vec4f };

// Camera-eid keys keep views' targets independent; resizing recreates them.
interface Targets {
    half: boolean;
    maskSeed: GPUTexture;
    distance: GPUTexture;
    distanceView: GPUTextureView;
    reduceGroup: GPUBindGroup;
    distanceGroup: GPUBindGroup;
    reduce: [GPURenderPassDescriptor, GPURenderPassDescriptor];
    distancePass: GPURenderPassDescriptor;
    seeds: [GPUTexture, GPUTexture];
    seedViews: [GPUTextureView, GPUTextureView];
    attr: GPUTexture;
    attrView: GPUTextureView;
    w: number;
    h: number;
    // mask[i] writes seeds[i]; jfa[i] writes seeds[1 - i], read by the slots whose parity is i
    mask: [GPURenderPassDescriptor, GPURenderPassDescriptor];
    jfa: [GPURenderPassDescriptor, GPURenderPassDescriptor];
    composite: GPUComputePassDescriptor;
    // JFA slot k reads seeds[k % 2] with step uniform k
    jfaGroups: GPUBindGroup[];
}

function destroyTargets(t: Targets): void {
    t.seeds[0].destroy();
    t.seeds[1].destroy();
    t.attr.destroy();
    if (t.half) {
        t.maskSeed.destroy();
        t.distance.destroy();
    }
}

function targets(world: World, eid: number, w: number, h: number, half: boolean): Targets {
    const _outlineState = outlineState(world);
    const cached = _outlineState.targets.get(eid);
    if (cached && cached.w === w && cached.h === h && cached.half === half) return cached;
    if (cached) destroyTargets(cached);
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const fw = half ? Math.ceil(w / 2) : w;
    const fh = half ? Math.ceil(h / 2) : h;
    const tex = (label: string, format: GPUTextureFormat, width = fw, height = fh) =>
        world.gpu.device.createTexture({ label, size: { width, height }, format, usage });
    const seeds: [GPUTexture, GPUTexture] = [
        tex(`outline-seedA-${eid}`, SEED_FORMAT),
        tex(`outline-seedB-${eid}`, SEED_FORMAT),
    ];
    const seedViews: [GPUTextureView, GPUTextureView] = [
        seeds[0].createView(),
        seeds[1].createView(),
    ];
    const attr = tex(`outline-attr-${eid}`, ATTR_FORMAT, w, h);
    const attrView = attr.createView();
    const maskSeed = half ? tex(`outline-mask-seed-${eid}`, SEED_FORMAT, w, h) : seeds[0];
    const distance = half ? tex(`outline-distance-${eid}`, "r16float") : attr;
    const distanceView = distance.createView();
    const mask = (seed: GPUTextureView): GPURenderPassDescriptor => ({
        label: `outline-mask/${eid}`,
        colorAttachments: [
            { view: seed, loadOp: "clear", storeOp: "store", clearValue: SEED_CLEAR },
            { view: attrView, loadOp: "clear", storeOp: "store", clearValue: ATTR_CLEAR },
        ],
    });
    const jfa = (seed: GPUTextureView): GPURenderPassDescriptor => ({
        label: `outline-jfa/${eid}`,
        colorAttachments: [
            { view: seed, loadOp: "clear", storeOp: "store", clearValue: SEED_CLEAR },
        ],
    });
    const jfaGroups: GPUBindGroup[] = [];
    for (let k = 0; k < MAX_JFA_PASSES; k++) {
        jfaGroups.push(
            world.gpu.root.unwrap(
                world.gpu.root.createBindGroup(jfaLayout, {
                    seed: seedViews[k % 2],
                    step: _outlineState.gpu.steps[k],
                }),
            ),
        );
    }
    const entry: Targets = {
        half,
        maskSeed,
        distance,
        distanceView,
        reduceGroup: world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(reduceLayout, { mask: maskSeed.createView() }),
        ),
        distanceGroup: world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(distanceLayout, { seed: seedViews[FINAL_SEED] }),
        ),
        reduce: [jfa(seedViews[0]), jfa(seedViews[1])],
        distancePass: {
            label: `outline-distance/${eid}`,
            colorAttachments: [{ view: distanceView, loadOp: "clear", storeOp: "store" }],
        },
        seeds,
        seedViews,
        attr,
        attrView,
        w,
        h,
        mask: half
            ? [mask(maskSeed.createView()), mask(maskSeed.createView())]
            : [mask(seedViews[0]), mask(seedViews[1])],
        jfa: [jfa(seedViews[1]), jfa(seedViews[0])],
        composite: { label: `outline-composite/${eid}` },
        jfaGroups,
    };
    _outlineState.targets.set(eid, entry);
    return entry;
}

// The seed view is fixed per target set; the other texture identities invalidate this cache.
type CompositeEntry = {
    read: GPUTextureView;
    write: GPUTextureView;
    attr: GPUTextureView;
    group: GPUBindGroup;
};

type MaskVariant = "plain" | "occlude" | "occlude-multisampled";

// per-camera, per-mesh-row mask bind group, rebuilt only when a buffer or view it binds changes. Keyed by
// dense row, not handle: a removed mesh's successor reuses its row, so turnover rebuilds in place.
type MaskEntry = {
    variant: MaskVariant;
    view: GPUBuffer;
    position: unknown;
    indices: unknown;
    globalTransforms: GPUBuffer;
    globalTransformRows: GPUBuffer;
    eids: GPUBuffer;
    attrs: GPUBuffer;
    quant: unknown;
    depth: GPUTextureView | null;
    group: GPUBindGroup;
};

interface OutlineGpuState {
    maskPlain: TgpuRenderPipeline<MaskTargets> | null;
    maskOcclude: TgpuRenderPipeline<MaskTargets> | null;
    maskOccludeMultisampled: TgpuRenderPipeline<MaskTargets> | null;
    jfa: TgpuRenderPipeline<d.Vec4u> | null;
    composite: TgpuComputePipeline | null;
    compositeHalf: TgpuComputePipeline | null;
    jfaHalf: TgpuRenderPipeline<d.Vec4u> | null;
    reduce: TgpuRenderPipeline<d.Vec4u> | null;
    distance: TgpuRenderPipeline<d.Vec4f> | null;
    eids: GPUBuffer | null;
    attrs: GPUBuffer | null;
    steps: StepBuffer[];
    capacity: number;
}

interface MeshBin {
    meshId: number;
    eids: number[];
    count: number;
}

interface Group {
    mesh: Mesh;
    meshId: number;
    first: number;
    count: number;
}

interface OutlineState {
    gpu: OutlineGpuState;
    eidsStaging: Uint32Array;
    attrStaging: Float32Array;
    targets: Map<number, Targets>;
    composites: Map<number, CompositeEntry>;
    masks: Map<number, Map<number, MaskEntry>>;
    binsByMesh: Map<number, MeshBin>;
    bins: MeshBin[];
    binCount: number;
    groups: Group[];
}

const outlineStateKey = { create: () => createOutlineState() };
const createOutlineState = (): OutlineState => ({
    gpu: {
        maskPlain: null,
        maskOcclude: null,
        maskOccludeMultisampled: null,
        jfa: null,
        composite: null,
        compositeHalf: null,
        jfaHalf: null,
        reduce: null,
        distance: null,
        eids: null,
        attrs: null,
        steps: [],
        capacity: 0,
    },
    eidsStaging: new Uint32Array(0),
    attrStaging: new Float32Array(0),
    targets: new Map(),
    composites: new Map(),
    masks: new Map(),
    binsByMesh: new Map(),
    bins: [],
    binCount: 0,
    groups: [],
});
const outlineState = (world: World) => world.resource(outlineStateKey);

function initializeOutlineState(world: World): void {
    world.resource(outlineStateKey);
}

function compositeBind(
    world: World,
    eid: number,
    read: GPUTextureView,
    write: GPUTextureView,
    t: Targets,
): GPUBindGroup {
    const _outlineState = outlineState(world);
    const cached = _outlineState.composites.get(eid);
    if (cached && cached.read === read && cached.write === write && cached.attr === t.attrView)
        return cached.group;
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(compositeLayout, {
            distance: t.distanceView,
            scene: read,
            seed: t.seedViews[FINAL_SEED],
            attr: t.attrView,
            output: write,
        }),
    );
    _outlineState.composites.set(eid, { read, write, attr: t.attrView, group });
    return group;
}

function maskBind(
    world: World,
    camEid: number,
    view: View,
    mesh: Mesh,
    meshId: number,
    globalTransforms: GPUBuffer,
    variant: MaskVariant,
): GPUBindGroup {
    const _outlineState = outlineState(world);
    let byMesh = _outlineState.masks.get(camEid);
    if (!byMesh) {
        byMesh = new Map();
        _outlineState.masks.set(camEid, byMesh);
    }
    const viewBuffer = world.resource(RenderContext).viewBuffers[view.slot]!;
    const globalTransformRows = globalTransformTable(world).eidToRowBuffer!;
    const eids = _outlineState.gpu.eids!;
    const attrs = _outlineState.gpu.attrs!;
    const depth = variant === "plain" ? null : view.depth;
    const row = world.resource(meshResourcesKey).meshes.index(meshId)!;
    const cached = byMesh.get(row);
    if (
        cached &&
        cached.variant === variant &&
        cached.view === viewBuffer &&
        cached.position === mesh.position &&
        cached.indices === mesh.indices &&
        cached.globalTransforms === globalTransforms &&
        cached.globalTransformRows === globalTransformRows &&
        cached.eids === eids &&
        cached.attrs === attrs &&
        cached.quant === mesh.quant &&
        cached.depth === depth
    )
        return cached.group;
    const bindings = {
        view: viewBuffer,
        position: mesh.position!,
        indices: mesh.indices,
        globalTransforms,
        globalTransformRows,
        maskEids: eids,
        maskAttrs: attrs,
        meshQuant: mesh.quant!,
    };
    const root = world.gpu.root;
    const group = root.unwrap(
        variant === "plain"
            ? root.createBindGroup(maskLayoutPlain, bindings)
            : variant === "occlude"
              ? root.createBindGroup(maskLayoutOcclude, { ...bindings, sceneDepth: depth! })
              : root.createBindGroup(maskLayoutOccludeMultisampled, {
                    ...bindings,
                    sceneDepth: depth!,
                }),
    );
    byMesh.set(row, {
        variant,
        view: viewBuffer,
        position: mesh.position,
        indices: mesh.indices,
        globalTransforms,
        globalTransformRows,
        eids,
        attrs,
        quant: mesh.quant,
        depth,
        group,
    });
    return group;
}

function ensureInstances(world: World, n: number): void {
    const _outlineState = world.resource(outlineStateKey);

    if (n <= _outlineState.gpu.capacity) return;
    let cap = Math.max(INITIAL_INSTANCES, _outlineState.gpu.capacity);
    while (cap < n) cap <<= 1;
    _outlineState.gpu.eids?.destroy();
    _outlineState.gpu.attrs?.destroy();
    _outlineState.gpu.eids = world.gpu.device.createBuffer({
        label: "outline-eids",
        size: cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _outlineState.gpu.attrs = world.gpu.device.createBuffer({
        label: "outline-attrs",
        size: cap * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _outlineState.gpu.capacity = cap;
    _outlineState.eidsStaging = new Uint32Array(cap);
    _outlineState.attrStaging = new Float32Array(cap * 8);
}

function renderOutline(
    world: World,
    camEid: number,
    view: View,
    globalTransforms: GPUBuffer,
    groupCount: number,
    passes: number,
    half: boolean,
    occlude: boolean,
): void {
    const _outlineState = world.resource(outlineStateKey);
    const gpu = _outlineState.gpu;
    const root = world.gpu.root;

    if (!view.framebuffer) return;
    const encoder = world.frameEncoder()!;
    const t = targets(world, camEid, view.width, view.height, half);
    const variant = !occlude
        ? "plain"
        : world.storage(Camera).antialias.get(camEid) !== 0
          ? "occlude-multisampled"
          : "occlude";
    // the ladder's first slot; the mask seeds the texture that slot reads
    const offset = MAX_JFA_PASSES - passes;

    const maskDescriptor = t.mask[offset % 2];
    maskDescriptor.timestampWrites = world.gpu.span?.("outline:mask");
    const mask = beginRenderPass(encoder, maskDescriptor);
    setRenderPipeline(
        mask,
        root.unwrap(
            variant === "plain"
                ? gpu.maskPlain!
                : variant === "occlude"
                  ? gpu.maskOcclude!
                  : gpu.maskOccludeMultisampled!,
        ),
    );
    for (let i = 0; i < groupCount; i++) {
        const g = _outlineState.groups[i]!;
        setRenderBindGroup(
            mask,
            0,
            maskBind(world, camEid, view, g.mesh, g.meshId, globalTransforms, variant),
        );
        mask.draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
    }
    endRenderPass(mask);

    // 2. JFA — slot k reads seeds[k % 2] and writes the other; the last slot always writes FINAL_SEED
    if (half) {
        const descriptor = t.reduce[offset % 2];
        descriptor.timestampWrites = world.gpu.span?.("outline:seed");
        const pass = beginRenderPass(encoder, descriptor);
        setRenderPipeline(pass, root.unwrap(gpu.reduce!));
        setRenderBindGroup(pass, 0, t.reduceGroup);
        draw(pass, 3);
        endRenderPass(pass);
    }
    const jfa = root.unwrap(half ? gpu.jfaHalf! : gpu.jfa!);
    for (let k = offset; k < MAX_JFA_PASSES; k++) {
        const descriptor = t.jfa[k % 2];
        descriptor.timestampWrites = world.gpu.span?.("outline:jfa");
        const pass = beginRenderPass(encoder, descriptor);
        setRenderPipeline(pass, jfa);
        setRenderBindGroup(pass, 0, t.jfaGroups[k]!);
        draw(pass, 3);
        endRenderPass(pass);
    }

    if (half) {
        t.distancePass.timestampWrites = world.gpu.span?.("outline:distance");
        const pass = beginRenderPass(encoder, t.distancePass);
        setRenderPipeline(pass, root.unwrap(gpu.distance!));
        setRenderBindGroup(pass, 0, t.distanceGroup);
        draw(pass, 3);
        endRenderPass(pass);
    }

    // Call sceneTransform after early returns: it repoints the framebuffer to the scratch we must write.
    const { read, write } = sceneTransform(world, view, camEid);
    t.composite.timestampWrites = world.gpu.span?.("outline:composite");
    const composite = beginComputePass(encoder, t.composite);
    setComputePipeline(composite, root.unwrap(half ? gpu.compositeHalf! : gpu.composite!));
    setComputeBindGroup(composite, 0, compositeBind(world, camEid, read, write, t));
    dispatchWorkgroups2(
        composite,
        Math.ceil(view.width / WORKGROUP),
        Math.ceil(view.height / WORKGROUP),
    );
    endComputePass(composite);
}

const OutlineSystem: System = {
    name: "outline",
    group: "draw",
    // an overlay: after the scene color (MainPassSystem) and after any scene-transform effect (the OverlaySystem
    // anchor, which fog runs before), so the band composites on top of the haze; before tonemapping presents it.
    // Both anchor refs drop harmlessly when their plugin isn't registered
    after: [MainPassSystem, OverlaySystem],
    before: [TonemappingSystem, PresentationSystem],
    update(world: World) {
        const _outlineState = world.resource(outlineStateKey);
        const _meshes = world.resource(Meshes);

        if (!_outlineState.gpu.maskPlain) return;
        _outlineState.targets.forEach(releaseStaleTargets, world);
        const instances = world.storage(MeshInstance);
        const bins = _outlineState.bins;
        for (let i = 0; i < _outlineState.binCount; i++) bins[i]!.count = 0;
        let binCount = 0;
        let n = 0;
        // Like Bevy's keyed IndexMap bins, active bins follow first-seen mesh order.
        // Keep their storage across frames so stable outlines do not allocate.
        for (const eid of world.query(OUTLINE_MESH_QUERY)) {
            const meshId = instances.mesh.get(eid);
            let bin = _outlineState.binsByMesh.get(meshId);
            if (!bin) {
                bin = { meshId, eids: [], count: 0 };
                _outlineState.binsByMesh.set(meshId, bin);
            }
            if (bin.count === 0) bins[binCount++] = bin;
            bin.eids[bin.count++] = eid;
            n++;
        }
        _outlineState.binCount = binCount;
        if (n === 0) return; // bare path — no passes
        const globalTransforms = globalTransformTable(world).buffer;

        const cameras = world.query(OUTLINE_CAMERA_QUERY);
        let cameraCount = 0;
        for (const camEid of cameras) {
            if (world.resource(Views).get(camEid)?.framebuffer) cameraCount++;
        }
        ensureInstances(world, n * cameraCount);
        const outline = world.storage(Outline);
        const eidsStaging = _outlineState.eidsStaging;
        const attrStaging = _outlineState.attrStaging;
        const groups = _outlineState.groups;
        let cursor = 0;
        for (const camEid of cameras) {
            const view = world.resource(Views).get(camEid);
            if (!view?.framebuffer) continue;
            const cameraMask = renderLayers(world, camEid);
            const start = cursor;
            let groupCount = 0;
            let maxWidth = 1;
            let minWidth = Infinity;
            let occlude = false;
            // group by mesh in first-seen order: each group's eids become a contiguous instance slice
            for (let i = 0; i < binCount; i++) {
                const bin = bins[i]!;
                const meshId = bin.meshId;
                const mesh = _meshes.get(meshId as MeshHandle);
                // mesh deleted / unregistered, or an un-quantized producer — nothing to outline
                if (!mesh?.position || !mesh.quant) continue;
                const first = cursor;
                for (let j = 0; j < bin.count; j++) {
                    const eid = bin.eids[j]!;
                    const meshMask = renderLayers(world, eid);
                    if ((meshMask & cameraMask) === 0) continue;
                    eidsStaging[cursor] = eid;
                    const o = cursor * 8;
                    unpackColorInto(outline.color.get(eid), attrStaging, o);
                    attrStaging[o + 3] = 1;
                    const w = Math.max(0, Math.min(MAX_WIDTH, outline.width.get(eid)));
                    const occ = outline.occlude.get(eid);
                    attrStaging[o + 4] = w;
                    attrStaging[o + 5] = occ;
                    if (w > maxWidth) maxWidth = w;
                    if (w > 0 && w < minWidth) minWidth = w;
                    if (occ > 0.5) occlude = true;
                    cursor++;
                }
                if (cursor === first) continue;
                let group = groups[groupCount];
                if (!group) {
                    group = { mesh, meshId, first, count: 0 };
                    groups[groupCount] = group;
                }
                group.mesh = mesh;
                group.meshId = meshId;
                group.first = first;
                group.count = cursor - first;
                groupCount++;
            }
            if (cursor === start) continue;
            // Each camera reads its own slice: queue writes all precede the shared submission.
            world.gpu.device.queue.writeBuffer(
                _outlineState.gpu.eids!,
                start * 4,
                eidsStaging,
                start,
                cursor - start,
            );
            world.gpu.device.queue.writeBuffer(
                _outlineState.gpu.attrs!,
                start * 32,
                attrStaging,
                start * 8,
                (cursor - start) * 8,
            );

            const half = maxWidth >= 2 && minWidth >= 2;
            const passes = jfaPassCount(half ? maxWidth / 2 : maxWidth);
            // occlusion needs the shared depth lane; without another requester, the outline plugin supplies it.
            renderOutline(
                world,
                camEid,
                view,
                globalTransforms,
                groupCount,
                passes,
                half,
                occlude && !!view.depth,
            );
        }
    },
};

// A camera that lost its framebuffer view, destroyed or recycled, releases its targets, composite and masks.
function releaseStaleTargets(this: World, t: Targets, eid: number): void {
    if (this.resource(Views).get(eid)?.framebuffer) return;
    const _outlineState = outlineState(this);
    destroyTargets(t);
    _outlineState.targets.delete(eid);
    _outlineState.composites.delete(eid);
    _outlineState.masks.delete(eid);
}

function releaseTargets(world: World): void {
    const _outlineState = outlineState(world);
    for (const t of _outlineState.targets.values()) destroyTargets(t);
    _outlineState.targets.clear();
    _outlineState.composites.clear();
    _outlineState.masks.clear();
}

function prepareOutline(world: World): void {
    const _outlineState = world.resource(outlineStateKey);

    // Queue writes precede submission, so each ladder slot needs a distinct step buffer.
    // Recreate buffers and dependent targets on warm to avoid retaining the prior device's resources.
    releaseTargets(world);
    for (const s of _outlineState.gpu.steps) s.destroy();
    _outlineState.gpu.steps.length = 0;
    for (let k = 0; k < MAX_JFA_PASSES; k++) {
        _outlineState.gpu.steps.push(
            world.gpu.root
                .createBuffer(d.f32, jfaStep(k))
                .$usage("uniform")
                .$name(`outline-jfa-step-${k}`),
        );
    }

    const maskTargets = { seed: { format: SEED_FORMAT }, attr: { format: ATTR_FORMAT } } as const;
    const maskPrimitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "back" };
    const fullscreen: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };

    _outlineState.gpu.jfa = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: jfaFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-jfa");
    _outlineState.gpu.jfaHalf = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: jfaHalfFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-jfa-half");
    _outlineState.gpu.reduce = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: reduceFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-seed-reduce");
    _outlineState.gpu.distance = world.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVertex,
            fragment: distanceFs,
            targets: { format: "r16float" },
            primitive: fullscreen,
        })
        .$name("outline-distance");
    _outlineState.gpu.compositeHalf = world.gpu.root
        .createComputePipeline({ compute: compositeHalfKernel })
        .$name("outline-composite-half");
    _outlineState.gpu.composite = world.gpu.root
        .createComputePipeline({ compute: compositeKernel })
        .$name("outline-composite");
    _outlineState.gpu.maskPlain = world.gpu.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutPlain),
            fragment: maskFragment(maskLayoutPlain, false),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask");
    _outlineState.gpu.maskOcclude = world.gpu.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutOcclude),
            fragment: maskFragment(maskLayoutOcclude, true),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask-occlude");
    _outlineState.gpu.maskOccludeMultisampled = world.gpu.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutOccludeMultisampled),
            fragment: maskFragment(maskLayoutOccludeMultisampled, true),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask-occlude-multisampled");

    // typegpu creates pipelines lazily, so an uncompiled pass would stall the first frame a highlight exists
    const gpu = _outlineState.gpu;
    precompile(world, "outline-half", () => [
        gpu.jfaHalf!,
        gpu.reduce!,
        gpu.distance!,
        gpu.compositeHalf!,
    ]);
    precompile(world, "outline-jfa", () => gpu.jfa!);
    precompile(world, "outline-composite", () => gpu.composite!);
    precompile(world, "outline-mask", () => gpu.maskPlain!);
    precompile(world, "outline-mask-occlude", () => gpu.maskOcclude!);
    precompile(world, "outline-mask-occlude-multisampled", () => gpu.maskOccludeMultisampled!);
}

function disposeOutline(world: World): void {
    const _outlineState = world.resource(outlineStateKey);

    _outlineState.gpu.eids?.destroy();
    _outlineState.gpu.attrs?.destroy();
    for (const s of _outlineState.gpu.steps) s.destroy();
    releaseTargets(world);
    _outlineState.gpu.eids = null;
    _outlineState.gpu.attrs = null;
    _outlineState.gpu.steps = [];
    _outlineState.gpu.maskPlain = null;
    _outlineState.gpu.maskOcclude = null;
    _outlineState.gpu.maskOccludeMultisampled = null;
    _outlineState.gpu.jfa = null;
    _outlineState.gpu.jfaHalf = null;
    _outlineState.gpu.reduce = null;
    _outlineState.gpu.distance = null;
    _outlineState.gpu.compositeHalf = null;
    _outlineState.gpu.composite = null;
    _outlineState.gpu.capacity = 0;
    _outlineState.eidsStaging = new Uint32Array(0);
    _outlineState.attrStaging = new Float32Array(0);
}

/**
 * the screen-space outline composite: add it alongside `StandardRenderingPlugin`, then add `Outline` to a MeshInstance entity to highlight it.
 *
 * The band is a mask → jump-flood distance field → composite over the scene color. Cost scales with the
 * highlighted-object count + screen × log(width), not scene geometry; nothing highlighted runs no passes.
 */
export const OutlinePlugin: Plugin = {
    gpu: {},
    name: "Outline",
    components: [Outline],
    systems: [OutlineSystem],
    dependencies: [RenderingPlugin, MeshRenderPlugin],

    initialize(world) {
        initializeOutlineState(world);
        world.resource(DepthPrepassRequests).push((world, eid) => {
            if (!world.has(eid, StandardRenderer)) return false;
            const occlusion = world.storage(Outline).occlude;
            for (const eid of world.query(OUTLINE_MESH_QUERY)) {
                if (occlusion.get(eid) > 0.5) return true;
            }
            return false;
        });
    },

    async warm(world: World) {
        prepareOutline(world);
    },

    dispose(world: World) {
        disposeOutline(world);
    },
};
