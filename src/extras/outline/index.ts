import { component } from "../../engine";
// Outline — the drop-in screen-space highlight. Add the `Outline` component to a MeshInstance entity and a
// uniform-width band hugs its silhouette: hover/select feedback, the player's grab highlight. The
// technique is mask → jump-flood distance field → threshold (Ben Golus, "The Quest for Very Wide
// Outlines"; Bevy's JFA outline crates), NOT an inverted hull (stylistic, non-uniform width). Cost
// scales with the highlighted-object count + screen × log(width), never with scene geometry — only the
// highlighted entities draw into a small coverage mask, and the JFA pass count is bounded by the band
// width, not the screen.
//
// Three passes per camera, single-sample at framebuffer resolution — two render, then one compute:
//   1. mask — draw only the `Outline` entities (a scoped instanced draw, grouped by mesh) into a seed
//      texture (each covered pixel seeds its own coordinate) + an attribute texture (per-entity color +
//      width). Always-on-top by default; `Outline.occlude` depth-tests against core's `view.depth` lane
//      so an occluded object's outline hides. The plugin requests the lane only while an occluding outline exists.
//   2. JFA — ping-pong fullscreen passes (`jfaPassCount(maxWidth)` of them) that flood the nearest seed
//      coordinate outward, producing a distance field within `width` pixels of every silhouette.
//   3. composite — one fullscreen **compute** dispatch through the `sceneTransform` seam: reads the
//      resolved scene (format-agnostic — the offscreen, or the fog scratch), the JFA distance field, and
//      the seed's color/width, blends the band over the scene in linear, and writes the rgba16float scratch.
//
// Runs in the post-color seam, ordered `after: [MainPassSystem, OverlaySystem]` (an overlay — on top of any
// scene-transform effect like fog) `before: [TonemappingSystem]`. The
// composite goes through `sceneTransform` (a compute pass) rather than a render pass into
// `view.framebuffer`, so it never assumes the framebuffer's format/usage — a fog scratch is rgba16float
// storage, not a render attachment — which is what let the two effects collide. Both anchor refs drop
// harmlessly when their plugin isn't registered. Targets the standard rendering path and requests core's shared depth lane.

import type { TgpuBuffer, TgpuComputePipeline, TgpuRenderPipeline, UniformFlag } from "typegpu";
import * as d from "typegpu/data";
import { type Mesh, Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import {
    Camera,
    DepthPrepassRequests,
    MainPassSystem,
    OverlaySystem,
    PresentationSystem,
    RenderContext,
    RenderingPlugin,
    sceneTransform,
    TonemappingSystem,
    type View,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { f32, vec4 } from "../../engine";
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
import { MeshRenderPlugin, StandardRenderer } from "../../standard/rendering";
import {
    compositeKernel,
    compositeLayout,
    fullscreenVs,
    jfaFs,
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
    WORKGROUP,
} from "./passes";

/**
 * outline highlight: a colored band hugs the object's silhouette for hover, selection, or grab feedback.
 *
 * Add it to a MeshInstance entity to highlight it; remove it to clear. Fields are per-entity, so different
 * highlights coexist in one pass.
 */
export const Outline = component(
    "Outline",
    {
        /** band color, linear rgb (alpha unused in v1) */
        color: vec4,
        /** band thickness in pixels, clamped to 64 */
        width: f32,
        /** 0 = always-on-top (default); 1 = occlusion-aware, hidden where the object is behind other geometry */
        occlude: f32,
    },
    {
        defaults: () => ({
            color: [1, 0.85, 0.2, 1],
            width: 4,
            occlude: 0,
        }),
    },
);

const OUTLINE_MESH_QUERY = [Outline, MeshInstance];
const OUTLINE_CAMERA_QUERY = [Camera];

// the seed texture stores the nearest covered-pixel coordinate as an INTEGER pixel index — uint, not
// f16: pixel-center fractions (x + 0.5) stop being f16-representable at 1024, which broke the
// interior's d == 0 test (every covered pixel right of screen x 1024 read d = 0.5 to its own seed →
// a half-alpha wash over the object). Integer indices are exact to 65535 and shift every coordinate
// uniformly by the same half-pixel, so distances are unchanged. The attr texture stores per-seed
// color (rgb) + width (a), read once at composite via the resolved seed coord
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

// per-camera screen-space targets: two ping-pong seed textures + the static attr texture, sized to the
// view and recreated on resize (standard's _laneTargets pattern), with the pass descriptors and JFA bind
// groups that read them. Keyed by camera eid so multi-view never shares one set
interface Targets {
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
}

function targets(world: World, eid: number, w: number, h: number): Targets {
    const _outlineState = outlineState(world);
    const cached = _outlineState.targets.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached;
    if (cached) destroyTargets(cached);
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, format: GPUTextureFormat) =>
        world.gpu.device.createTexture({ label, size: { width: w, height: h }, format, usage });
    const seeds: [GPUTexture, GPUTexture] = [
        tex(`outline-seedA-${eid}`, SEED_FORMAT),
        tex(`outline-seedB-${eid}`, SEED_FORMAT),
    ];
    const seedViews: [GPUTextureView, GPUTextureView] = [
        seeds[0].createView(),
        seeds[1].createView(),
    ];
    const attr = tex(`outline-attr-${eid}`, ATTR_FORMAT);
    const attrView = attr.createView();
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
        seeds,
        seedViews,
        attr,
        attrView,
        w,
        h,
        mask: [mask(seedViews[0]), mask(seedViews[1])],
        jfa: [jfa(seedViews[1]), jfa(seedViews[0])],
        composite: { label: `outline-composite/${eid}` },
        jfaGroups,
    };
    _outlineState.targets.set(eid, entry);
    return entry;
}

// per-camera composite bind group, cached on the sceneTransform read + write + the attr view (mirroring
// fog's per-view cache); the seed it reads is fixed per target set
type CompositeEntry = {
    read: GPUTextureView;
    write: GPUTextureView;
    attr: GPUTextureView;
    group: GPUBindGroup;
};

type MaskVariant = "plain" | "occlude" | "occlude-multisampled";

// per-camera, per-mesh mask bind group, rebuilt only when a buffer or view it binds changes
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
    eids: GPUBuffer | null;
    attrs: GPUBuffer | null;
    steps: StepBuffer[];
    capacity: number;
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
    // persistent per-frame scratch: the highlighted eids, their meshes, and the mesh groups
    eids: number[];
    meshes: number[];
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
    eids: [],
    meshes: [],
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
    const globalTransformRows = world.gpu.buffers.get("globalTransformRows")!;
    const eids = _outlineState.gpu.eids!;
    const attrs = _outlineState.gpu.attrs!;
    const depth = variant === "plain" ? null : view.depth;
    const cached = byMesh.get(meshId);
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
    byMesh.set(meshId, {
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
    occlude: boolean,
): void {
    const _outlineState = world.resource(outlineStateKey);
    const gpu = _outlineState.gpu;
    const root = world.gpu.root;

    if (!view.framebuffer) return;
    const encoder = world.frameEncoder()!;
    const t = targets(world, camEid, view.width, view.height);
    const variant = !occlude
        ? "plain"
        : world.storage(Camera).antialias.get(camEid) !== 0
          ? "occlude-multisampled"
          : "occlude";
    // the ladder's first slot; the mask seeds the texture that slot reads
    const offset = MAX_JFA_PASSES - passes;

    // 1. mask — the scoped instanced draw, grouped by mesh, into seed + attr (MRT, no depth attachment)
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
    const jfa = root.unwrap(gpu.jfa!);
    for (let k = offset; k < MAX_JFA_PASSES; k++) {
        const descriptor = t.jfa[k % 2];
        descriptor.timestampWrites = world.gpu.span?.("outline:jfa");
        const pass = beginRenderPass(encoder, descriptor);
        setRenderPipeline(pass, jfa);
        setRenderBindGroup(pass, 0, t.jfaGroups[k]!);
        draw(pass, 3);
        endRenderPass(pass);
    }

    // 3. composite — blend the band over the resolved scene through the sceneTransform seam. A compute pass
    // reads the scene format-agnostically (offscreen, or fog's scratch) + the
    // JFA field, writes the rgba16float scratch, repoints `view.framebuffer`. `sceneTransform` is called here,
    // last — the caller's early-outs already ran, so the framebuffer is never repointed at an unwritten scratch
    const { read, write } = sceneTransform(world, view, camEid);
    t.composite.timestampWrites = world.gpu.span?.("outline:composite");
    const composite = beginComputePass(encoder, t.composite);
    setComputePipeline(composite, root.unwrap(gpu.composite!));
    setComputeBindGroup(composite, 0, compositeBind(world, camEid, read, write, t));
    dispatchWorkgroups2(
        composite,
        Math.ceil(view.width / WORKGROUP),
        Math.ceil(view.height / WORKGROUP),
    );
    endComputePass(composite);
}

/**
 * draw every camera's outline, after the scene color is resolved. Collects the highlighted MeshInstance entities,
 * groups them by mesh into one instance buffer, then runs mask → JFA → composite per camera. Nothing
 * highlighted → returns before any GPU pass (zero cost on the bare path)
 */
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
        const instances = world.storage(MeshInstance);
        const eids = _outlineState.eids;
        const meshes = _outlineState.meshes;
        let n = 0;
        for (const eid of world.query(OUTLINE_MESH_QUERY)) {
            eids[n] = eid;
            meshes[n] = instances.mesh.get(eid);
            n++;
        }
        if (n === 0) return; // bare path — no passes
        const globalTransforms = world.gpu.buffers.get("global-transform-interpolated");
        if (!globalTransforms) return;

        ensureInstances(world, n);
        const outline = world.storage(Outline);
        const eidsStaging = _outlineState.eidsStaging;
        const attrStaging = _outlineState.attrStaging;
        const groups = _outlineState.groups;
        let groupCount = 0;
        let cursor = 0;
        let maxWidth = 1;
        let occlude = false;
        // group by mesh in first-seen order: each group's eids become a contiguous instance slice
        for (let i = 0; i < n; i++) {
            const meshId = meshes[i]!;
            let seen = false;
            for (let j = 0; j < i; j++) {
                if (meshes[j] === meshId) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            const mesh = _meshes.get(meshId as MeshHandle);
            // mesh deleted / unregistered, or an un-quantized producer — nothing to outline
            if (!mesh?.position || !mesh.quant) continue;
            const first = cursor;
            for (let j = i; j < n; j++) {
                if (meshes[j] !== meshId) continue;
                const eid = eids[j]!;
                eidsStaging[cursor] = eid;
                const o = cursor * 8;
                attrStaging[o] = outline.color.x.get(eid);
                attrStaging[o + 1] = outline.color.y.get(eid);
                attrStaging[o + 2] = outline.color.z.get(eid);
                attrStaging[o + 3] = outline.color.w.get(eid);
                const w = Math.max(0, Math.min(MAX_WIDTH, outline.width.get(eid)));
                const occ = outline.occlude.get(eid);
                attrStaging[o + 4] = w;
                attrStaging[o + 5] = occ;
                if (w > maxWidth) maxWidth = w;
                if (occ > 0.5) occlude = true;
                cursor++;
            }
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
        if (cursor === 0) return;
        world.gpu.device.queue.writeBuffer(_outlineState.gpu.eids!, 0, eidsStaging, 0, cursor);
        world.gpu.device.queue.writeBuffer(_outlineState.gpu.attrs!, 0, attrStaging, 0, cursor * 8);

        const passes = jfaPassCount(maxWidth);
        for (const camEid of world.query(OUTLINE_CAMERA_QUERY)) {
            const view = world.resource(Views).get(camEid);
            if (!view?.framebuffer) continue;
            // occlusion needs the shared depth lane; without another requester, the outline plugin supplies it.
            renderOutline(
                world,
                camEid,
                view,
                globalTransforms,
                groupCount,
                passes,
                occlude && !!view.depth,
            );
        }
    },
};

function releaseTargets(world: World): void {
    const _outlineState = outlineState(world);
    for (const t of _outlineState.targets.values()) destroyTargets(t);
    _outlineState.targets.clear();
    _outlineState.composites.clear();
    _outlineState.masks.clear();
}

function prepareOutline(world: World): void {
    const _outlineState = world.resource(outlineStateKey);

    // the JFA + composite layouts are the `jfaLayout` / `compositeLayout` in passes.ts — declared
    // beside the kernels that read them, bound by layout object, never by group index. Only the per-pass
    // step uniforms are this module's: one buffer per ladder slot, written once here, since a queued write lands
    // before the submit and one rewritten uniform would clobber every pass with the last step. Rebuilt (not
    // reused) so a re-warm on a fresh device can't hold a buffer from the old one; the prior set and the
    // targets whose bind groups read it are freed rather than leaked
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
            vertex: fullscreenVs,
            fragment: jfaFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-jfa");
    _outlineState.gpu.composite = world.gpu.root
        .createComputePipeline({ compute: compositeKernel })
        .$name("outline-composite");
    // the two mask variants: same vs/fs shape over the plain / occlude layout (`maskVertex`/`maskFragment`
    // re-emit per layout), splicing the TGSL `decodePos` /
    // `xformPoint` real references (the resolve-call-graph precedent — no chunk splice needed)
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
