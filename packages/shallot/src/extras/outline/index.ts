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
// scene-transform effect like fog, see render.md "the post-color seam") `before: [GlazeSystem]`. The
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
import type { Plugin, State, System } from "../../engine";
import { Compute, f32, sparse, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { GlazeSystem } from "../../standard/glaze";
import { Part, PartPlugin } from "../../standard/part";
import { Camera, type Mesh, RenderPlugin } from "../../standard/render";
import {
    Meshes,
    OverlaySystem,
    Render,
    sceneTransform,
    type View,
    Views,
} from "../../standard/render/core";
import { ColorSystem, DEPTH_FORMAT } from "../../standard/sear/core";
import { Transform, TransformsPlugin } from "../../standard/transforms";
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
    color: sparse(vec4),
    /** band thickness in pixels, clamped to 64 */
    width: sparse(f32),
    /** 0 = always-on-top (default); 1 = occlusion-aware, hidden where the object is behind other geometry (needs sear's `Depth` on the camera) */
    occlude: sparse(f32),
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

const _gpu = {
    maskPlain: null as TgpuRenderPipeline<MaskTargets> | null,
    maskOcclude: null as TgpuRenderPipeline<MaskTargets> | null,
    jfa: null as TgpuRenderPipeline<d.Vec4u> | null,
    composite: null as TgpuComputePipeline | null,
    eids: null as GPUBuffer | null,
    attrs: null as GPUBuffer | null,
    steps: [] as StepBuffer[],
    capacity: 0,
};

let _eidsStaging = new Uint32Array(0);
let _attrStaging = new Float32Array(0);

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
const _targets = new Map<number, Targets>();

function targets(eid: number, w: number, h: number): Targets {
    const cached = _targets.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached;
    cached?.seedA.destroy();
    cached?.seedB.destroy();
    cached?.attr.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, format: GPUTextureFormat) =>
        Compute.device.createTexture({ label, size: { width: w, height: h }, format, usage });
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
    _targets.set(eid, entry);
    return entry;
}

// per-camera composite bind group, cached on the sceneTransform read + write + the final JFA seed + the
// attr view (mirroring fog's per-view cache). All four reallocate only on a resize, and the final seed view
// flips only when the band width changes JFA-pass parity — so this holds across frames, unlike the per-frame
// mask/JFA bind groups (rebuilt each frame because their seed src ping-pongs within the JFA loop)
type CompositeGroup = TgpuBindGroup<(typeof compositeLayout)["entries"]>;

const _composite = new Map<
    number,
    {
        read: GPUTextureView;
        write: GPUTextureView;
        seed: GPUTextureView;
        attr: GPUTextureView;
        group: CompositeGroup;
    }
>();

function compositeBind(
    eid: number,
    read: GPUTextureView,
    write: GPUTextureView,
    seed: GPUTextureView,
    attr: GPUTextureView,
): CompositeGroup {
    const cached = _composite.get(eid);
    if (
        cached &&
        cached.read === read &&
        cached.write === write &&
        cached.seed === seed &&
        cached.attr === attr
    )
        return cached.group;
    const group = Compute.root.createBindGroup(compositeLayout, {
        scene: read,
        seed,
        attr,
        output: write,
    });
    _composite.set(eid, { read, write, seed, attr, group });
    return group;
}

function ensureInstances(n: number): void {
    if (n <= _gpu.capacity) return;
    let cap = Math.max(INITIAL_INSTANCES, _gpu.capacity);
    while (cap < n) cap <<= 1;
    _gpu.eids?.destroy();
    _gpu.attrs?.destroy();
    _gpu.eids = Compute.device.createBuffer({
        label: "outline-eids",
        size: cap * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _gpu.attrs = Compute.device.createBuffer({
        label: "outline-attrs",
        size: cap * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _gpu.capacity = cap;
    _eidsStaging = new Uint32Array(cap);
    _attrStaging = new Float32Array(cap * 8);
}

interface Group {
    mesh: Mesh;
    first: number;
    count: number;
}

function renderOutline(
    camEid: number,
    view: View,
    transforms: GPUBuffer,
    groups: Group[],
    steps: number[],
    occlude: boolean,
): void {
    const encoder = Render.encoder;
    if (!encoder || !view.framebuffer) return;
    const t = targets(camEid, view.width, view.height);
    const seedClear = { r: SENTINEL, g: SENTINEL, b: 0, a: 0 };

    // 1. mask — the scoped instanced draw, grouped by mesh, into seed + attr (MRT, no depth attachment)
    const mask = encoder.beginRenderPass({
        label: `outline-mask/${camEid}`,
        timestampWrites: Compute.span?.("outline:mask"),
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
            const group = Compute.root.createBindGroup(maskLayoutOcclude, {
                view: Render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                transforms,
                maskEids: _gpu.eids!,
                maskAttrs: _gpu.attrs!,
                meshQuant: g.mesh.quant,
                sceneDepth: view.depth!,
            });
            _gpu.maskOcclude!.with(group)
                .with(mask)
                .draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
        } else {
            const group = Compute.root.createBindGroup(maskLayoutPlain, {
                view: Render.viewBuffers[view.slot],
                position: g.mesh.position,
                indices: g.mesh.indices,
                transforms,
                maskEids: _gpu.eids!,
                maskAttrs: _gpu.attrs!,
                meshQuant: g.mesh.quant,
            });
            _gpu.maskPlain!.with(group)
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
            timestampWrites: Compute.span?.("outline:jfa"),
            colorAttachments: [
                { view: dstView, loadOp: "clear", storeOp: "store", clearValue: seedClear },
            ],
        });
        const group = Compute.root.createBindGroup(jfaLayout, {
            seed: srcView,
            step: _gpu.steps[k],
        });
        _gpu.jfa!.with(group).with(pass).draw(3);
        pass.end();
        [srcView, dstView] = [dstView, srcView];
    }

    // 3. composite — blend the band over the resolved scene through the sceneTransform seam. A compute pass
    // (TBDR-friendly, like glaze): reads the scene format-agnostically (offscreen, or fog's scratch) + the
    // JFA field, writes the rgba16float scratch, repoints `view.framebuffer`. `sceneTransform` is called here,
    // last — the caller's early-outs already ran, so the framebuffer is never repointed at an unwritten scratch
    const { read, write } = sceneTransform(view, camEid);
    const composite = encoder.beginComputePass({
        label: `outline-composite/${camEid}`,
        timestampWrites: Compute.span?.("outline:composite"),
    });
    _gpu.composite!.with(compositeBind(camEid, read, write, srcView, t.attrView))
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
    annotations: { mode: "always" },
    // an overlay: after the scene color (ColorSystem) and after any scene-transform effect (the OverlaySystem
    // anchor, which fog runs before), so the band composites on top of the haze; before glaze presents it.
    // Both anchor refs drop harmlessly when their plugin isn't registered (render.md "the post-color seam")
    after: [ColorSystem, OverlaySystem],
    before: [GlazeSystem],
    update(state: State) {
        if (!Render.encoder || !_gpu.maskPlain) return;
        const eids = [...state.query([Outline, Part])];
        if (eids.length === 0) return; // bare path — no passes
        const transforms = Compute.buffers.get("transforms");
        if (!transforms) return;

        ensureInstances(eids.length);
        const byMesh = groupByMesh(eids, (eid) => Part.mesh.get(eid));
        const groups: Group[] = [];
        let cursor = 0;
        let maxWidth = 1;
        let occlude = false;
        for (const [meshId, group] of byMesh) {
            const name = Meshes.name(meshId);
            const mesh = name ? Meshes.get(name) : undefined;
            if (!mesh) continue; // mesh deleted / unregistered — skip the group
            const first = cursor;
            for (const eid of group) {
                _eidsStaging[cursor] = eid;
                const o = cursor * 8;
                _attrStaging[o] = Outline.color.x.get(eid);
                _attrStaging[o + 1] = Outline.color.y.get(eid);
                _attrStaging[o + 2] = Outline.color.z.get(eid);
                _attrStaging[o + 3] = Outline.color.w.get(eid);
                const w = Math.max(0, Math.min(MAX_WIDTH, Outline.width.get(eid)));
                const occ = Outline.occlude.get(eid);
                _attrStaging[o + 4] = w;
                _attrStaging[o + 5] = occ;
                if (w > maxWidth) maxWidth = w;
                if (occ > 0.5) occlude = true;
                cursor++;
            }
            groups.push({ mesh, first, count: group.length });
        }
        if (cursor === 0) return;
        Compute.device.queue.writeBuffer(_gpu.eids!, 0, _eidsStaging, 0, cursor);
        Compute.device.queue.writeBuffer(_gpu.attrs!, 0, _attrStaging, 0, cursor * 8);

        const steps = jfaSteps(maxWidth);
        for (let k = 0; k < steps.length; k++) _gpu.steps[k].write(steps[k]);

        for (const camEid of state.query([Camera])) {
            const view = Views.get(camEid);
            if (!view?.framebuffer) continue;
            // occlusion needs sear's Depth lane; without it, degrade to always-on-top
            renderOutline(camEid, view, transforms, groups, steps, occlude && !!view.depth);
        }
    },
};

function prepareOutline(): void {
    // the JFA + composite layouts are the typed `jfaLayout` / `compositeLayout` in passes.ts — declared
    // beside the kernels that read them, bound by layout object, never by group index. Only the per-pass
    // step uniforms are this module's: rebuilt here (not reused) so a re-warm on a fresh device can't hold a
    // buffer from the old one, and the prior set is freed rather than leaked
    for (const s of _gpu.steps) s.destroy();
    _gpu.steps.length = 0;
    for (let k = 0; k < MAX_JFA_PASSES; k++) {
        _gpu.steps.push(
            Compute.root.createBuffer(d.f32).$usage("uniform").$name(`outline-jfa-step-${k}`),
        );
    }

    const maskTargets = { seed: { format: SEED_FORMAT }, attr: { format: ATTR_FORMAT } } as const;
    const maskPrimitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "back" };
    const fullscreen: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };

    _gpu.jfa = Compute.root
        .createRenderPipeline({
            vertex: fullscreenVs,
            fragment: jfaFs,
            targets: { format: SEED_FORMAT },
            primitive: fullscreen,
        })
        .$name("outline-jfa");
    _gpu.composite = Compute.root
        .createComputePipeline({ compute: compositeKernel })
        .$name("outline-composite");
    // the two mask variants: same vs/fs shape over the plain / occlude layout (`maskVertex`/`maskFragment`
    // re-emit per layout), splicing the already-typed `decodePos` /
    // `xformPoint` real references (the resolve-call-graph precedent — no chunk splice needed)
    _gpu.maskPlain = Compute.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutPlain),
            fragment: maskFragment(maskLayoutPlain, false),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask");
    _gpu.maskOcclude = Compute.root
        .createRenderPipeline({
            vertex: maskVertex(maskLayoutOcclude),
            fragment: maskFragment(maskLayoutOcclude, true),
            targets: maskTargets,
            primitive: maskPrimitive,
        })
        .$name("outline-mask-occlude");

    forceCompile();
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
function forceCompile(): void {
    const stand = (format: GPUTextureFormat, usage: number) =>
        Compute.device.createTexture({
            label: "outline-warm",
            size: { width: 1, height: 1 },
            format,
            usage: usage | GPUTextureUsage.TEXTURE_BINDING,
        });

    precompile("outline-jfa", () => {
        const src = stand(SEED_FORMAT, 0);
        const dst = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = Compute.root.createBindGroup(jfaLayout, {
            seed: src.createView(),
            step: _gpu.steps[0],
        });
        const bound = _gpu.jfa!.with(group).withColorAttachment({ view: dst.createView() });
        src.destroy();
        dst.destroy();
        return bound;
    });

    precompile("outline-composite", () => {
        const scene = stand(ATTR_FORMAT, 0);
        const seed = stand(SEED_FORMAT, 0);
        const attr = stand(ATTR_FORMAT, 0);
        const out = stand(ATTR_FORMAT, GPUTextureUsage.STORAGE_BINDING);
        const group = Compute.root.createBindGroup(compositeLayout, {
            scene: scene.createView(),
            seed: seed.createView(),
            attr: attr.createView(),
            output: out.createView(),
        });
        const bound = _gpu.composite!.with(group);
        scene.destroy();
        seed.destroy();
        attr.destroy();
        out.destroy();
        return bound;
    });

    // the mask buffers (position/indices/transforms/maskEids/maskAttrs/meshQuant) are storage bindings, not
    // textures — 4-byte throwaways, same shape as `stand()`'s texture stand-ins
    const buf = (size: number) =>
        Compute.device.createBuffer({
            label: "outline-mask-warm",
            size,
            usage: GPUBufferUsage.STORAGE,
        });

    precompile("outline-mask", () => {
        const position = buf(8);
        const indices = buf(4);
        const transformsBuf = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const group = Compute.root.createBindGroup(maskLayoutPlain, {
            view: Render.viewBuffers[0],
            position,
            indices,
            transforms: transformsBuf,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
        });
        const bound = _gpu.maskPlain!.with(group).withColorAttachment({
            seed: { view: seed.createView() },
            attr: { view: attr.createView() },
        });
        position.destroy();
        indices.destroy();
        transformsBuf.destroy();
        eids.destroy();
        attrs.destroy();
        quant.destroy();
        seed.destroy();
        attr.destroy();
        return bound;
    });

    precompile("outline-mask-occlude", () => {
        const position = buf(8);
        const indices = buf(4);
        const transformsBuf = buf(48);
        const eids = buf(4);
        const attrs = buf(32);
        const quant = buf(48);
        const seed = stand(SEED_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const attr = stand(ATTR_FORMAT, GPUTextureUsage.RENDER_ATTACHMENT);
        const depth = stand(DEPTH_FORMAT, 0);
        const group = Compute.root.createBindGroup(maskLayoutOcclude, {
            view: Render.viewBuffers[0],
            position,
            indices,
            transforms: transformsBuf,
            maskEids: eids,
            maskAttrs: attrs,
            meshQuant: quant,
            sceneDepth: depth.createView(),
        });
        const bound = _gpu.maskOcclude!.with(group).withColorAttachment({
            seed: { view: seed.createView() },
            attr: { view: attr.createView() },
        });
        position.destroy();
        indices.destroy();
        transformsBuf.destroy();
        eids.destroy();
        attrs.destroy();
        quant.destroy();
        seed.destroy();
        attr.destroy();
        depth.destroy();
        return bound;
    });
}

function disposeOutline(): void {
    _gpu.eids?.destroy();
    _gpu.attrs?.destroy();
    for (const s of _gpu.steps) s.destroy();
    for (const t of _targets.values()) {
        t.seedA.destroy();
        t.seedB.destroy();
        t.attr.destroy();
    }
    _targets.clear();
    _composite.clear();
    _gpu.eids = null;
    _gpu.attrs = null;
    _gpu.steps = [];
    _gpu.maskPlain = null;
    _gpu.maskOcclude = null;
    _gpu.jfa = null;
    _gpu.composite = null;
    _gpu.capacity = 0;
    _eidsStaging = new Uint32Array(0);
    _attrStaging = new Float32Array(0);
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
    dependencies: [RenderPlugin, PartPlugin, TransformsPlugin],
    traits: {
        Outline: {
            requires: [Part, Transform],
            defaults: () => ({
                color: [1, 0.85, 0.2, 1],
                width: 4,
                occlude: 0,
            }),
        },
    },

    async warm() {
        if (!Compute.device) return;
        prepareOutline();
    },

    dispose() {
        disposeOutline();
    },
};
