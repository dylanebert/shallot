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
//   2. JFA — ping-pong fullscreen passes (`jfaSteps(maxWidth)` of them) that flood the nearest seed
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

import type {
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    TgpuRenderPipeline,
    UniformFlag,
} from "typegpu";
import * as d from "typegpu/data";
import { type Mesh, Meshes, type MeshHandle, MeshInstance } from "../../core/mesh";
import {
    Camera,
    DEPTH_FORMAT,
    DepthPrepassRequests,
    MainPassSystem,
    OverlaySystem,
    PresentationSystem,
    RenderContext,
    RenderingPlugin,
    SAMPLE_COUNT,
    sceneTransform,
    TonemappingSystem,
    type View,
    Views,
} from "../../core/rendering";
import type { Plugin, System, World } from "../../engine";
import { f32, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { MeshRenderPlugin, StandardRenderer } from "../../standard/rendering";
import {
    compositeKernel,
    compositeLayout,
    fullscreenVs,
    groupByMesh,
    jfaFs,
    jfaLayout,
    jfaSteps,
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
const INITIAL_INSTANCES = 64;
// one uniform buffer per JFA pass slot, all written up front: a queued write lands before the submit, so a
// single rewritten uniform would clobber every pass with the last step. Distinct buffers can't collide by
// construction. The count is exactly what `jfaSteps` can return — the ladder halves from the first power of
// two ≥ MAX_WIDTH down to 1
const MAX_JFA_PASSES = Math.ceil(Math.log2(MAX_WIDTH)) + 1;

type StepBuffer = TgpuBuffer<typeof d.f32> & UniformFlag;

type MaskTargets = { seed: d.Vec4u; attr: d.Vec4f };

// per-camera screen-space targets: two ping-pong seed textures + the static attr texture, sized to the
// view and recreated on resize (standard's _laneTargets pattern). Keyed by camera eid so multi-view never
// shares one set
interface Targets {
    seedA: GPUTexture;
    seedAView: GPUTextureView;
    seedB: GPUTexture;
    seedBView: GPUTextureView;
    attr: GPUTexture;
    attrView: GPUTextureView;
    w: number;
    h: number;
}
function targets(world: World, eid: number, w: number, h: number): Targets {
    const cached = outlineState(world).targets.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.seedA.destroy();
    cached?.seedB.destroy();
    cached?.attr.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, format: GPUTextureFormat) =>
        world.gpu.device.createTexture({ label, size: { width: w, height: h }, format, usage });
    const seedA = tex(`outline-seedA-${eid}`, SEED_FORMAT);
    const seedB = tex(`outline-seedB-${eid}`, SEED_FORMAT);
    const attr = tex(`outline-attr-${eid}`, ATTR_FORMAT);
    const entry: Targets = {
        seedA,
        seedAView: seedA.createView(),
        seedB,
        seedBView: seedB.createView(),
        attr,
        attrView: attr.createView(),
        w,
        h,
    };
    outlineState(world).targets.set(eid, entry);
    return entry;
}

// per-camera composite bind group, cached on the sceneTransform read + write + the final JFA seed + the
// attr view (mirroring fog's per-view cache). All four reallocate only on a resize, and the final seed view
// flips only when the band width changes JFA-pass parity — so this holds across frames, unlike the per-frame
// mask/JFA bind groups (rebuilt each frame because their seed src ping-pongs within the JFA loop)
type CompositeGroup = TgpuBindGroup<(typeof compositeLayout)["entries"]>;

type CompositeEntry = {
    read: GPUTextureView;
    write: GPUTextureView;
    seed: GPUTextureView;
    attr: GPUTextureView;
    group: CompositeGroup;
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

interface OutlineState {
    gpu: OutlineGpuState;
    eidsStaging: Uint32Array;
    attrStaging: Float32Array;
    targets: Map<number, Targets>;
    composites: Map<number, CompositeEntry>;
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
    seed: GPUTextureView,
    attr: GPUTextureView,
): CompositeGroup {
    const cached = outlineState(world).composites.get(eid);
    if (
        cached &&
        cached.read === read &&
        cached.write === write &&
        cached.seed === seed &&
        cached.attr === attr
    )
        return cached.group;
    const group = world.gpu.root.createBindGroup(compositeLayout, {
        scene: read,
        seed,
        attr,
        output: write,
    });
    outlineState(world).composites.set(eid, { read, write, seed, attr, group });
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
    outlineState(world).eidsStaging = new Uint32Array(cap);
    outlineState(world).attrStaging = new Float32Array(cap * 8);
}

interface Group {
    mesh: Mesh;
    first: number;
    count: number;
}

function renderOutline(
    world: World,
    camEid: number,
    view: View,
    globalTransforms: GPUBuffer,
    groups: Group[],
    steps: number[],
    occlude: boolean,
): void {
    const _render = world.resource(RenderContext);
    const _outlineState = world.resource(outlineStateKey);

    if (!view.framebuffer) return;
    const encoder = world.frameEncoder()!;
    const t = targets(world, camEid, view.width, view.height);
    const multisampled = world.storage(Camera).antialias.get(camEid) !== 0;
    const seedClear = { r: SENTINEL, g: SENTINEL, b: 0, a: 0 };

    // 1. mask — the scoped instanced draw, grouped by mesh, into seed + attr (MRT, no depth attachment)
    const mask = encoder.beginRenderPass({
        label: `outline-mask/${camEid}`,
        timestampWrites: world.gpu.span?.("outline:mask"),
        colorAttachments: [
            { view: t.seedAView, loadOp: "clear", storeOp: "store", clearValue: seedClear },
            {
                view: t.attrView,
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
            },
        ],
    });
    for (const g of groups) {
        if (!g.mesh.position || !g.mesh.quant) continue; // un-quantized producer — nothing to outline
        if (occlude) {
            const depthPipeline = multisampled
                ? _outlineState.gpu.maskOccludeMultisampled!
                : _outlineState.gpu.maskOcclude!;
            const bindings = {
                view: _render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                globalTransforms,
                globalTransformRows: world.gpu.buffers.get("globalTransformRows")!,
                maskEids: _outlineState.gpu.eids!,
                maskAttrs: _outlineState.gpu.attrs!,
                meshQuant: g.mesh.quant,
                sceneDepth: view.depth!,
            };
            const group = multisampled
                ? world.gpu.root.createBindGroup(maskLayoutOccludeMultisampled, bindings)
                : world.gpu.root.createBindGroup(maskLayoutOcclude, bindings);
            depthPipeline
                .with(group as never)
                .with(mask)
                .draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
        } else {
            const group = world.gpu.root.createBindGroup(maskLayoutPlain, {
                view: _render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                globalTransforms,
                globalTransformRows: world.gpu.buffers.get("globalTransformRows")!,
                maskEids: _outlineState.gpu.eids!,
                maskAttrs: _outlineState.gpu.attrs!,
                meshQuant: g.mesh.quant,
            });
            _outlineState.gpu
                .maskPlain!.with(group)
                .with(mask)
                .draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
        }
    }
    mask.end();

    // 2. JFA — ping-pong the seed field outward; after the loop `srcView` holds the final distance field
    let srcView = t.seedAView;
    let dstView = t.seedBView;
    for (let k = 0; k < steps.length; k++) {
        const pass = encoder.beginRenderPass({
            label: `outline-jfa/${camEid}`,
            timestampWrites: world.gpu.span?.("outline:jfa"),
            colorAttachments: [
                { view: dstView, loadOp: "clear", storeOp: "store", clearValue: seedClear },
            ],
        });
        const group = world.gpu.root.createBindGroup(jfaLayout, {
            seed: srcView,
            step: _outlineState.gpu.steps[k],
        });
        _outlineState.gpu.jfa!.with(group).with(pass).draw(3);
        pass.end();
        [srcView, dstView] = [dstView, srcView];
    }

    // 3. composite — blend the band over the resolved scene through the sceneTransform seam. A compute pass
    // reads the scene format-agnostically (offscreen, or fog's scratch) + the
    // JFA field, writes the rgba16float scratch, repoints `view.framebuffer`. `sceneTransform` is called here,
    // last — the caller's early-outs already ran, so the framebuffer is never repointed at an unwritten scratch
    const { read, write } = sceneTransform(world, view, camEid);
    const composite = encoder.beginComputePass({
        label: `outline-composite/${camEid}`,
        timestampWrites: world.gpu.span?.("outline:composite"),
    });
    _outlineState.gpu
        .composite!.with(compositeBind(world, camEid, read, write, srcView, t.attrView))
        .with(composite)
        .dispatchWorkgroups(Math.ceil(view.width / WORKGROUP), Math.ceil(view.height / WORKGROUP));
    composite.end();
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
        const eids = [...world.query(OUTLINE_MESH_QUERY)];
        if (eids.length === 0) return; // bare path — no passes
        const globalTransforms = world.gpu.buffers.get("global-transform-interpolated");
        if (!globalTransforms) return;

        ensureInstances(world, eids.length);
        const byMesh = groupByMesh(eids, (eid) => world.storage(MeshInstance).mesh.get(eid));
        const groups: Group[] = [];
        let cursor = 0;
        let maxWidth = 1;
        let occlude = false;
        for (const [meshId, group] of byMesh) {
            const mesh = _meshes.get(meshId as MeshHandle);
            if (!mesh) continue; // mesh deleted / unregistered — skip the group
            const first = cursor;
            for (const eid of group) {
                outlineState(world).eidsStaging[cursor] = eid;
                const o = cursor * 8;
                outlineState(world).attrStaging[o] = world.storage(Outline).color.x.get(eid);
                outlineState(world).attrStaging[o + 1] = world.storage(Outline).color.y.get(eid);
                outlineState(world).attrStaging[o + 2] = world.storage(Outline).color.z.get(eid);
                outlineState(world).attrStaging[o + 3] = world.storage(Outline).color.w.get(eid);
                const w = Math.max(0, Math.min(MAX_WIDTH, world.storage(Outline).width.get(eid)));
                const occ = world.storage(Outline).occlude.get(eid);
                outlineState(world).attrStaging[o + 4] = w;
                outlineState(world).attrStaging[o + 5] = occ;
                if (w > maxWidth) maxWidth = w;
                if (occ > 0.5) occlude = true;
                cursor++;
            }
            groups.push({ mesh, first, count: group.length });
        }
        if (cursor === 0) return;
        world.gpu.device.queue.writeBuffer(
            _outlineState.gpu.eids!,
            0,
            outlineState(world).eidsStaging,
            0,
            cursor,
        );
        world.gpu.device.queue.writeBuffer(
            _outlineState.gpu.attrs!,
            0,
            outlineState(world).attrStaging,
            0,
            cursor * 8,
        );

        const steps = jfaSteps(maxWidth);
        for (let k = 0; k < steps.length; k++) _outlineState.gpu.steps[k].write(steps[k]);

        for (const camEid of world.query(OUTLINE_CAMERA_QUERY)) {
            const view = world.resource(Views).get(camEid);
            if (!view?.framebuffer) continue;
            // occlusion needs the shared depth lane; without another requester, the outline plugin supplies it.
            renderOutline(
                world,
                camEid,
                view,
                globalTransforms,
                groups,
                steps,
                occlude && !!view.depth,
            );
        }
    },
};

function prepareOutline(world: World): void {
    const _outlineState = world.resource(outlineStateKey);

    // the JFA + composite layouts are the `jfaLayout` / `compositeLayout` in passes.ts — declared
    // beside the kernels that read them, bound by layout object, never by group index. Only the per-pass
    // step uniforms are this module's: rebuilt here (not reused) so a re-warm on a fresh device can't hold a
    // buffer from the old one, and the prior set is freed rather than leaked
    for (const s of _outlineState.gpu.steps) s.destroy();
    _outlineState.gpu.steps.length = 0;
    for (let k = 0; k < MAX_JFA_PASSES; k++) {
        _outlineState.gpu.steps.push(
            world.gpu.root.createBuffer(d.f32).$usage("uniform").$name(`outline-jfa-step-${k}`),
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

    forceCompile(world);
}

/**
 * force the two pipelines to compile under the loading screen. typegpu creates pipelines
 * synchronously, so Dawn defers the real compile — and the outline's passes run every frame a
 * highlight exists, which would put that stall on whichever frame the first hover lands. Their real bind
 * groups need per-camera targets that don't exist until a view attaches, so each forcer allocates its own
 * 1×1 stand-ins and binds. Destroying them before the drain is safe because `initAsync` only compiles
 * the pipeline — it records and submits nothing, so compilation never reads the bind groups the
 * stand-ins were bound into.
 */
function forceCompile(world: World): void {
    const stand = (format: GPUTextureFormat, usage: number) =>
        world.gpu.device.createTexture({
            label: "outline-warm",
            size: { width: 1, height: 1 },
            format,
            usage: usage | GPUTextureUsage.TEXTURE_BINDING,
        });

    precompile(world, "outline-jfa", () => {
        const _outlineState = world.resource(outlineStateKey);

        const src = stand(SEED_FORMAT, 0);
        const dst = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = world.gpu.root.createBindGroup(jfaLayout, {
            seed: src.createView(),
            step: _outlineState.gpu.steps[0],
        });
        const bound = _outlineState.gpu
            .jfa!.with(group)
            .withColorAttachment({ view: dst.createView() });
        src.destroy();
        dst.destroy();
        return bound;
    });

    precompile(world, "outline-composite", () => {
        const scene = stand(ATTR_FORMAT, 0);
        const seed = stand(SEED_FORMAT, 0);
        const attr = stand(ATTR_FORMAT, 0);
        const out = stand(ATTR_FORMAT, GPUTextureUsage.STORAGE_BINDING);
        const group = world.gpu.root.createBindGroup(compositeLayout, {
            scene: scene.createView(),
            seed: seed.createView(),
            attr: attr.createView(),
            output: out.createView(),
        });
        const bound = world.resource(outlineStateKey).gpu.composite!.with(group);
        scene.destroy();
        seed.destroy();
        attr.destroy();
        out.destroy();
        return bound;
    });

    // the mask buffers (position/indices/globalTransforms/maskEids/maskAttrs/meshQuant) are storage bindings, not
    // textures — 4-byte throwaways, same shape as `stand()`'s texture stand-ins
    const buf = (size: number) =>
        world.gpu.device.createBuffer({
            label: "outline-mask-warm",
            size,
            usage: GPUBufferUsage.STORAGE,
        });

    precompile(world, "outline-mask", () => {
        const position = buf(8);
        const indices = buf(4);
        const globalTransformsBuffer = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = world.gpu.root.createBindGroup(maskLayoutPlain, {
            view: world.resource(RenderContext).viewBuffers[0],
            position,
            indices,
            globalTransforms: globalTransformsBuffer,
            globalTransformRows: eids,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
        });
        const bound = world
            .resource(outlineStateKey)
            .gpu.maskPlain!.with(group)
            .withColorAttachment({
                seed: { view: seed.createView() },
                attr: { view: attr.createView() },
            });
        position.destroy();
        indices.destroy();
        globalTransformsBuffer.destroy();
        eids.destroy();
        attrs.destroy();
        quant.destroy();
        seed.destroy();
        attr.destroy();
        return bound;
    });

    precompile(world, "outline-mask-occlude", () => {
        const position = buf(8);
        const indices = buf(4);
        const globalTransformsBuffer = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const depth = stand(DEPTH_FORMAT, 0);
        const group = world.gpu.root.createBindGroup(maskLayoutOcclude, {
            view: world.resource(RenderContext).viewBuffers[0],
            position,
            indices,
            globalTransforms: globalTransformsBuffer,
            globalTransformRows: eids,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
            sceneDepth: depth.createView(),
        });
        const bound = world
            .resource(outlineStateKey)
            .gpu.maskOcclude!.with(group)
            .withColorAttachment({
                seed: { view: seed.createView() },
                attr: { view: attr.createView() },
            });
        position.destroy();
        indices.destroy();
        globalTransformsBuffer.destroy();
        eids.destroy();
        attrs.destroy();
        quant.destroy();
        seed.destroy();
        attr.destroy();
        depth.destroy();
        return bound;
    });

    precompile(world, "outline-mask-occlude-multisampled", () => {
        const position = buf(8);
        const indices = buf(4);
        const globalTransformsBuffer = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const depth = world.gpu.device.createTexture({
            label: "outline-mask-warm-depth-msaa",
            size: { width: 1, height: 1 },
            format: DEPTH_FORMAT,
            sampleCount: SAMPLE_COUNT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        const group = world.gpu.root.createBindGroup(maskLayoutOccludeMultisampled, {
            view: world.resource(RenderContext).viewBuffers[0],
            position,
            indices,
            globalTransforms: globalTransformsBuffer,
            globalTransformRows: eids,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
            sceneDepth: depth.createView(),
        });
        const bound = world
            .resource(outlineStateKey)
            .gpu.maskOccludeMultisampled!.with(group)
            .withColorAttachment({
                seed: { view: seed.createView() },
                attr: { view: attr.createView() },
            });
        position.destroy();
        indices.destroy();
        globalTransformsBuffer.destroy();
        eids.destroy();
        attrs.destroy();
        quant.destroy();
        seed.destroy();
        attr.destroy();
        depth.destroy();
        return bound;
    });
}

function disposeOutline(world: World): void {
    const _outlineState = world.resource(outlineStateKey);

    _outlineState.gpu.eids?.destroy();
    _outlineState.gpu.attrs?.destroy();
    for (const s of _outlineState.gpu.steps) s.destroy();
    for (const t of outlineState(world).targets.values()) {
        t.seedA.destroy();
        t.seedB.destroy();
        t.attr.destroy();
    }
    outlineState(world).targets.clear();
    outlineState(world).composites.clear();
    _outlineState.gpu.eids = null;
    _outlineState.gpu.attrs = null;
    _outlineState.gpu.steps = [];
    _outlineState.gpu.maskPlain = null;
    _outlineState.gpu.maskOcclude = null;
    _outlineState.gpu.maskOccludeMultisampled = null;
    _outlineState.gpu.jfa = null;
    _outlineState.gpu.composite = null;
    _outlineState.gpu.capacity = 0;
    outlineState(world).eidsStaging = new Uint32Array(0);
    outlineState(world).attrStaging = new Float32Array(0);
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
        if (!world.gpu.device) return;
        prepareOutline(world);
    },

    dispose(world: World) {
        disposeOutline(world);
    },
};
