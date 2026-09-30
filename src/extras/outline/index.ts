// Outline — the drop-in screen-space highlight. Add the `Outline` component to a Part entity and a
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
//      width). Always-on-top by default; `Outline.occlude` depth-tests against sear's `view.depth` lane
//      so an occluded object's outline hides (needs `Depth` on the camera).
//   2. JFA — ping-pong fullscreen passes (`jfaSteps(maxWidth)` of them) that flood the nearest seed
//      coordinate outward, producing a distance field within `width` pixels of every silhouette.
//   3. composite — one fullscreen **compute** dispatch through the `sceneTransform` seam: reads the
//      resolved scene (format-agnostic — the offscreen, or the fog scratch), the JFA distance field, and
//      the seed's color/width, blends the band over the scene in linear, and writes the rgba16float scratch.
//
// Runs in the post-color seam, ordered `after: [ColorSystem, OverlaySystem]` (an overlay — on top of any
// scene-transform effect like fog) `before: [GlazeSystem]`. The
// composite goes through `sceneTransform` (a compute pass, like glaze) rather than a render pass into
// `view.framebuffer`, so it never assumes the framebuffer's format/usage — a fog scratch is rgba16float
// storage, not a render attachment — which is what let the two effects collide. Both anchor refs drop
// harmlessly when their plugin isn't registered. Targets the sear + glaze path (reads sear's `Depth` lane).

import type {
    TgpuBindGroup,
    TgpuBuffer,
    TgpuComputePipeline,
    TgpuRenderPipeline,
    UniformFlag,
} from "typegpu";
import * as d from "typegpu/data";
import {
    Camera,
    type Mesh,
    Meshes,
    OverlaySystem,
    Render,
    RenderPlugin,
    sceneTransform,
    type View,
    Views,
} from "../../core/rendering";
import type { Plugin, World, System } from "../../engine";
import { f32, GlobalTransform, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { ColorSystem, DEPTH_FORMAT } from "../../standard/rendering";
import { GlazeSystem } from "../../transitional/glaze";
import { Part, PartPlugin } from "../../transitional/part";
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
    maskLayoutPlain,
    maskVertex,
    WORKGROUP,
} from "./passes";

/**
 * outline highlight: a colored band hugs the object's silhouette for hover, selection, or grab feedback.
 *
 * Add it to a Part entity to highlight it; remove it to clear. Fields are per-entity, so different
 * highlights coexist in one pass.
 *
 * @example
 * ```
 * // hover feedback driven by a pick (the cast hands you the hovered eid)
 * if (mode === "hover") state.add(hovered, Outline);
 * else state.remove(hovered, Outline);
 * ```
 */
export const Outline = {
    /** band color, linear rgb (alpha unused in v1) */
    color: vec4,
    /** band thickness in pixels, clamped to 64 */
    width: f32,
    /** 0 = always-on-top (default); 1 = occlusion-aware, hidden where the object is behind other geometry (needs sear's `Depth` on the camera) */
    occlude: f32,
};

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
// view and recreated on resize (sear's _laneTargets pattern). Keyed by camera eid so multi-view never
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
function targets(state: World, eid: number, w: number, h: number): Targets {
    const cached = outlineState(state).targets.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.seedA.destroy();
    cached?.seedB.destroy();
    cached?.attr.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, format: GPUTextureFormat) =>
        state.gpu.device.createTexture({ label, size: { width: w, height: h }, format, usage });
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
    outlineState(state).targets.set(eid, entry);
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
const outlineState = (state: World) => state.resource(outlineStateKey);

function initializeOutlineState(state: World): void {
    state.resource(outlineStateKey);
}

function compositeBind(
    state: World,
    eid: number,
    read: GPUTextureView,
    write: GPUTextureView,
    seed: GPUTextureView,
    attr: GPUTextureView,
): CompositeGroup {
    const cached = outlineState(state).composites.get(eid);
    if (
        cached &&
        cached.read === read &&
        cached.write === write &&
        cached.seed === seed &&
        cached.attr === attr
    )
        return cached.group;
    const group = state.gpu.root.createBindGroup(compositeLayout, {
        scene: read,
        seed,
        attr,
        output: write,
    });
    outlineState(state).composites.set(eid, { read, write, seed, attr, group });
    return group;
}

function ensureInstances(state: World, n: number): void {
    const _outlineState = state.resource(outlineStateKey);

    if (n <= _outlineState.gpu.capacity) return;
    let cap = Math.max(INITIAL_INSTANCES, _outlineState.gpu.capacity);
    while (cap < n) cap <<= 1;
    _outlineState.gpu.eids?.destroy();
    _outlineState.gpu.attrs?.destroy();
    _outlineState.gpu.eids = state.gpu.device.createBuffer({
        label: "outline-eids",
        size: cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _outlineState.gpu.attrs = state.gpu.device.createBuffer({
        label: "outline-attrs",
        size: cap * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _outlineState.gpu.capacity = cap;
    outlineState(state).eidsStaging = new Uint32Array(cap);
    outlineState(state).attrStaging = new Float32Array(cap * 8);
}

interface Group {
    mesh: Mesh;
    first: number;
    count: number;
}

function renderOutline(
    state: World,
    camEid: number,
    view: View,
    globalTransforms: GPUBuffer,
    groups: Group[],
    steps: number[],
    occlude: boolean,
): void {
    const _render = state.resource(Render);
    const _outlineState = state.resource(outlineStateKey);

    const encoder = _render.encoder;
    if (!encoder || !view.framebuffer) return;
    const t = targets(state, camEid, view.width, view.height);
    const seedClear = { r: SENTINEL, g: SENTINEL, b: 0, a: 0 };

    // 1. mask — the scoped instanced draw, grouped by mesh, into seed + attr (MRT, no depth attachment)
    const mask = encoder.beginRenderPass({
        label: `outline-mask/${camEid}`,
        timestampWrites: state.gpu.span?.("outline:mask"),
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
            const group = state.gpu.root.createBindGroup(maskLayoutOcclude, {
                view: _render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                globalTransforms,
                maskEids: _outlineState.gpu.eids!,
                maskAttrs: _outlineState.gpu.attrs!,
                meshQuant: g.mesh.quant,
                sceneDepth: view.depth!,
            });
            _outlineState.gpu
                .maskOcclude!.with(group)
                .with(mask)
                .draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
        } else {
            const group = state.gpu.root.createBindGroup(maskLayoutPlain, {
                view: _render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                globalTransforms,
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
            timestampWrites: state.gpu.span?.("outline:jfa"),
            colorAttachments: [
                { view: dstView, loadOp: "clear", storeOp: "store", clearValue: seedClear },
            ],
        });
        const group = state.gpu.root.createBindGroup(jfaLayout, {
            seed: srcView,
            step: _outlineState.gpu.steps[k],
        });
        _outlineState.gpu.jfa!.with(group).with(pass).draw(3);
        pass.end();
        [srcView, dstView] = [dstView, srcView];
    }

    // 3. composite — blend the band over the resolved scene through the sceneTransform seam. A compute pass
    // (TBDR-friendly, like glaze): reads the scene format-agnostically (offscreen, or fog's scratch) + the
    // JFA field, writes the rgba16float scratch, repoints `view.framebuffer`. `sceneTransform` is called here,
    // last — the caller's early-outs already ran, so the framebuffer is never repointed at an unwritten scratch
    const { read, write } = sceneTransform(state, view, camEid);
    const composite = encoder.beginComputePass({
        label: `outline-composite/${camEid}`,
        timestampWrites: state.gpu.span?.("outline:composite"),
    });
    _outlineState.gpu
        .composite!.with(compositeBind(state, camEid, read, write, srcView, t.attrView))
        .with(composite)
        .dispatchWorkgroups(Math.ceil(view.width / WORKGROUP), Math.ceil(view.height / WORKGROUP));
    composite.end();
}

/**
 * draw every camera's outline, after the scene color is resolved. Collects the highlighted Part entities,
 * groups them by mesh into one instance buffer, then runs mask → JFA → composite per camera. Nothing
 * highlighted → returns before any GPU pass (zero cost on the bare path)
 */
const OutlineSystem: System = {
    name: "outline",
    group: "draw",
    // an overlay: after the scene color (ColorSystem) and after any scene-transform effect (the OverlaySystem
    // anchor, which fog runs before), so the band composites on top of the haze; before glaze presents it.
    // Both anchor refs drop harmlessly when their plugin isn't registered
    after: [ColorSystem, OverlaySystem],
    before: [GlazeSystem],
    update(state: World) {
        const _outlineState = state.resource(outlineStateKey);
        const _meshes = state.resource(Meshes);

        if (!state.resource(Render).encoder || !_outlineState.gpu.maskPlain) return;
        const eids = [...state.query([Outline, Part])];
        if (eids.length === 0) return; // bare path — no passes
        const globalTransforms = state.gpu.buffers.get("global-transform-interpolated");
        if (!globalTransforms) return;

        ensureInstances(state, eids.length);
        const byMesh = groupByMesh(eids, (eid) => state.of(Part).mesh.get(eid));
        const groups: Group[] = [];
        let cursor = 0;
        let maxWidth = 1;
        let occlude = false;
        for (const [meshId, group] of byMesh) {
            const name = _meshes.name(meshId);
            const mesh = name ? _meshes.get(name) : undefined;
            if (!mesh) continue; // mesh deleted / unregistered — skip the group
            const first = cursor;
            for (const eid of group) {
                outlineState(state).eidsStaging[cursor] = eid;
                const o = cursor * 8;
                outlineState(state).attrStaging[o] = state.of(Outline).color.x.get(eid);
                outlineState(state).attrStaging[o + 1] = state.of(Outline).color.y.get(eid);
                outlineState(state).attrStaging[o + 2] = state.of(Outline).color.z.get(eid);
                outlineState(state).attrStaging[o + 3] = state.of(Outline).color.w.get(eid);
                const w = Math.max(0, Math.min(MAX_WIDTH, state.of(Outline).width.get(eid)));
                const occ = state.of(Outline).occlude.get(eid);
                outlineState(state).attrStaging[o + 4] = w;
                outlineState(state).attrStaging[o + 5] = occ;
                if (w > maxWidth) maxWidth = w;
                if (occ > 0.5) occlude = true;
                cursor++;
            }
            groups.push({ mesh, first, count: group.length });
        }
        if (cursor === 0) return;
        state.gpu.device.queue.writeBuffer(
            _outlineState.gpu.eids!,
            0,
            outlineState(state).eidsStaging,
            0,
            cursor,
        );
        state.gpu.device.queue.writeBuffer(
            _outlineState.gpu.attrs!,
            0,
            outlineState(state).attrStaging,
            0,
            cursor * 8,
        );

        const steps = jfaSteps(maxWidth);
        for (let k = 0; k < steps.length; k++) _outlineState.gpu.steps[k].write(steps[k]);

        for (const camEid of state.query([Camera])) {
            const view = state.resource(Views).get(camEid);
            if (!view?.framebuffer) continue;
            // occlusion needs sear's Depth lane; without it, degrade to always-on-top
            renderOutline(
                state,
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

function prepareOutline(state: World): void {
    const _outlineState = state.resource(outlineStateKey);

    // the JFA + composite layouts are the typed `jfaLayout` / `compositeLayout` in passes.ts — declared
    // beside the kernels that read them, bound by layout object, never by group index. Only the per-pass
    // step uniforms are this module's: rebuilt here (not reused) so a re-warm on a fresh device can't hold a
    // buffer from the old one, and the prior set is freed rather than leaked
    for (const s of _outlineState.gpu.steps) s.destroy();
    _outlineState.gpu.steps.length = 0;
    for (let k = 0; k < MAX_JFA_PASSES; k++) {
        _outlineState.gpu.steps.push(
            state.gpu.root.createBuffer(d.f32).$usage("uniform").$name(`outline-jfa-step-${k}`),
        );
    }

    const maskTargets = { seed: { format: SEED_FORMAT }, attr: { format: ATTR_FORMAT } } as const;
    const maskPrimitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "back" };
    const fullscreen: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };

    _outlineState.gpu.jfa = state.gpu.root
        .createRenderPipeline({
            vertex: fullscreenVs,
            fragment: jfaFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-jfa");
    _outlineState.gpu.composite = state.gpu.root
        .createComputePipeline({ compute: compositeKernel })
        .$name("outline-composite");
    // the two mask variants: same vs/fs shape over the plain / occlude layout (`maskVertex`/`maskFragment`
    // re-emit per layout), splicing the already-typed `decodePos` /
    // `xformPoint` real references (the resolve-call-graph precedent — no chunk splice needed)
    _outlineState.gpu.maskPlain = state.gpu.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutPlain),
            fragment: maskFragment(maskLayoutPlain, false),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask");
    _outlineState.gpu.maskOcclude = state.gpu.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutOcclude),
            fragment: maskFragment(maskLayoutOcclude, true),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask-occlude");

    forceCompile(state);
}

/**
 * force the two typed pipelines to compile under the loading screen. typegpu creates pipelines
 * synchronously, so Dawn defers the real compile — and the outline's passes run every frame a
 * highlight exists, which would put that stall on whichever frame the first hover lands. Their real bind
 * groups need per-camera targets that don't exist until a view attaches, so each forcer allocates its own
 * 1×1 stand-ins and binds. Destroying them before the drain is safe because `initAsync` only compiles
 * the pipeline — it records and submits nothing, so compilation never reads the bind groups the
 * stand-ins were bound into.
 */
function forceCompile(state: World): void {
    const stand = (format: GPUTextureFormat, usage: number) =>
        state.gpu.device.createTexture({
            label: "outline-warm",
            size: { width: 1, height: 1 },
            format,
            usage: usage | GPUTextureUsage.TEXTURE_BINDING,
        });

    precompile(state, "outline-jfa", () => {
        const _outlineState = state.resource(outlineStateKey);

        const src = stand(SEED_FORMAT, 0);
        const dst = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = state.gpu.root.createBindGroup(jfaLayout, {
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

    precompile(state, "outline-composite", () => {
        const scene = stand(ATTR_FORMAT, 0);
        const seed = stand(SEED_FORMAT, 0);
        const attr = stand(ATTR_FORMAT, 0);
        const out = stand(ATTR_FORMAT, GPUTextureUsage.STORAGE_BINDING);
        const group = state.gpu.root.createBindGroup(compositeLayout, {
            scene: scene.createView(),
            seed: seed.createView(),
            attr: attr.createView(),
            output: out.createView(),
        });
        const bound = state.resource(outlineStateKey).gpu.composite!.with(group);
        scene.destroy();
        seed.destroy();
        attr.destroy();
        out.destroy();
        return bound;
    });

    // the mask buffers (position/indices/globalTransforms/maskEids/maskAttrs/meshQuant) are storage bindings, not
    // textures — 4-byte throwaways, same shape as `stand()`'s texture stand-ins
    const buf = (size: number) =>
        state.gpu.device.createBuffer({
            label: "outline-mask-warm",
            size,
            usage: GPUBufferUsage.STORAGE,
        });

    precompile(state, "outline-mask", () => {
        const position = buf(8);
        const indices = buf(4);
        const globalTransformsBuffer = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = state.gpu.root.createBindGroup(maskLayoutPlain, {
            view: state.resource(Render).viewBuffers[0],
            position,
            indices,
            globalTransforms: globalTransformsBuffer,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
        });
        const bound = state
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

    precompile(state, "outline-mask-occlude", () => {
        const position = buf(8);
        const indices = buf(4);
        const globalTransformsBuffer = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const depth = stand(DEPTH_FORMAT, 0);
        const group = state.gpu.root.createBindGroup(maskLayoutOcclude, {
            view: state.resource(Render).viewBuffers[0],
            position,
            indices,
            globalTransforms: globalTransformsBuffer,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
            sceneDepth: depth.createView(),
        });
        const bound = state
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
}

function disposeOutline(state: World): void {
    const _outlineState = state.resource(outlineStateKey);

    _outlineState.gpu.eids?.destroy();
    _outlineState.gpu.attrs?.destroy();
    for (const s of _outlineState.gpu.steps) s.destroy();
    for (const t of outlineState(state).targets.values()) {
        t.seedA.destroy();
        t.seedB.destroy();
        t.attr.destroy();
    }
    outlineState(state).targets.clear();
    outlineState(state).composites.clear();
    _outlineState.gpu.eids = null;
    _outlineState.gpu.attrs = null;
    _outlineState.gpu.steps = [];
    _outlineState.gpu.maskPlain = null;
    _outlineState.gpu.maskOcclude = null;
    _outlineState.gpu.jfa = null;
    _outlineState.gpu.composite = null;
    _outlineState.gpu.capacity = 0;
    outlineState(state).eidsStaging = new Uint32Array(0);
    outlineState(state).attrStaging = new Float32Array(0);
}

/**
 * the screen-space outline composite: add it alongside `SearPlugin` + `GlazePlugin`, then add `Outline` to a Part entity to highlight it.
 *
 * The band is a mask → jump-flood distance field → composite over the scene color. Cost scales with the
 * highlighted-object count + screen × log(width), not scene geometry; nothing highlighted runs no passes.
 */
export const OutlinePlugin: Plugin = {
    name: "Outline",
    components: { Outline },
    systems: [OutlineSystem],
    dependencies: [RenderPlugin, PartPlugin],
    traits: {
        Outline: {
            requires: [Part, GlobalTransform],
            defaults: () => ({
                color: [1, 0.85, 0.2, 1],
                width: 4,
                occlude: 0,
            }),
        },
    },

    initialize(state) {
        initializeOutlineState(state);
    },

    async warm(state: World) {
        if (!state.gpu.device) return;
        prepareOutline(state);
    },

    dispose(state: World) {
        disposeOutline(state);
    },
};
