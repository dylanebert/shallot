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

import type { Plugin, State, System } from "../../engine";
import { Compute, f32, sparse, vec4 } from "../../engine";
import { GlazeSystem } from "../../standard/glaze";
import { Part, PartPlugin } from "../../standard/part";
import { Camera, type Mesh, RenderPlugin } from "../../standard/render";
import {
    Meshes,
    OverlaySystem,
    Render,
    sceneTransform,
    VIEW_BYTES,
    VIEW_STRIDE,
    type View,
    Views,
} from "../../standard/render/core";
import { ColorSystem } from "../../standard/sear/core";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import { groupByMesh, jfaSteps, MAX_WIDTH, maskCode } from "./passes";

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
// compute workgroup tile — 8×8 = 64 threads, matching glaze/fog's screen-space composite
const WORKGROUP = 8;
// the "no seed" sentinel the seed textures clear to: a coordinate far off-screen, so any real seed wins
// the nearest-distance test and a pixel that never reaches a seed reads a huge distance (no band)
const SENTINEL = 30000;
const INITIAL_INSTANCES = 64;
// dynamic-offset uniform stride for the per-JFA-pass step (minUniformBufferOffsetAlignment ≥ 256). One
// slot per pass, written up front: writeBuffer is queue-ordered before the submit, so a single rewritten
// uniform would clobber every pass with the last step. Slot-major writes never collide (glaze's pattern)
const JFA_STRIDE = 256;
const JFA_BYTES = 4;
const MAX_JFA_PASSES = 16;

const FULLSCREEN_VS = /* wgsl */ `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// one jump-flood iteration: for the 9 neighbours at the current step, keep the seed coordinate nearest
// this pixel. Bounds-guarded — an out-of-bounds textureLoad returns 0, which would read as a seed at the
// corner, so skip neighbours off the texture. The sentinel seed loses naturally (its distance is huge)
const JFA_WGSL = /* wgsl */ `
struct Step { value: f32 }
@group(0) @binding(0) var seed: texture_2d<u32>;
@group(0) @binding(1) var<uniform> step: Step;
${FULLSCREEN_VS}
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec2<u32> {
    let dim = vec2<i32>(textureDimensions(seed));
    let p = vec2<i32>(pos.xy);
    let here = vec2<f32>(p);
    var best = textureLoad(seed, p, 0).xy;
    var bestD = distance(here, vec2<f32>(best));
    let s = i32(step.value);
    for (var oy = -1; oy <= 1; oy = oy + 1) {
        for (var ox = -1; ox <= 1; ox = ox + 1) {
            let q = p + vec2<i32>(ox, oy) * s;
            if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) { continue; }
            let cand = textureLoad(seed, q, 0).xy;
            let d = distance(here, vec2<f32>(cand));
            if (d < bestD) { bestD = d; best = cand; }
        }
    }
    return best;
}`;

// the composite, a compute dispatch through the sceneTransform seam: each pixel's distance to its nearest
// seed gives the band. `smoothstep(0,1,d)` fades the outline in 1px outside the silhouette (interior d≈0 →
// alpha 0, the object keeps its color), the outer `1 - smoothstep(width-1,width,d)` fades it out at the
// band edge — uniform width, antialiased. The seed's color/width come from the attr texture at the resolved
// seed coordinate. `mix(scene, band, alpha)` is straight (non-premultiplied) alpha `over`, in linear scene
// space (the offscreen + scratch are linear HDR) — the analytic twin of the old hardware ALPHA_BLEND. It
// writes *every* pixel (scene-through where there's no band) because the ping-pong target is a separate
// texture, so there's no in-place `discard`; glaze tonemaps the result
const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var scene: texture_2d<f32>;
@group(0) @binding(1) var seed: texture_2d<u32>;
@group(0) @binding(2) var attr: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(output);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let p = vec2<i32>(gid.xy);
    let base = textureLoad(scene, p, 0).rgb;
    let s = textureLoad(seed, p, 0).xy;
    let d = distance(vec2<f32>(p), vec2<f32>(s));
    let a = textureLoad(attr, vec2<i32>(s), 0); // out-of-bounds (sentinel seed) reads 0 → width 0 → no band
    let width = a.w;
    let alpha = smoothstep(0.0, 1.0, d) * (1.0 - smoothstep(width - 1.0, width, d));
    let band = select(0.0, alpha, width > 0.0);
    textureStore(output, p, vec4<f32>(mix(base, a.rgb, band), 1.0));
}`;

const _gpu = {
    maskPlain: null as GPURenderPipeline | null,
    maskOcclude: null as GPURenderPipeline | null,
    jfa: null as GPURenderPipeline | null,
    composite: null as GPUComputePipeline | null,
    maskLayoutPlain: null as GPUBindGroupLayout | null,
    maskLayoutOcclude: null as GPUBindGroupLayout | null,
    jfaLayout: null as GPUBindGroupLayout | null,
    compositeLayout: null as GPUBindGroupLayout | null,
    eids: null as GPUBuffer | null,
    attrs: null as GPUBuffer | null,
    steps: null as GPUBuffer | null,
    capacity: 0,
};

let _eidsStaging = new Uint32Array(0);
let _attrStaging = new Float32Array(0);
const _step = new Float32Array(1);

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
const _composite = new Map<
    number,
    {
        read: GPUTextureView;
        write: GPUTextureView;
        seed: GPUTextureView;
        attr: GPUTextureView;
        group: GPUBindGroup;
    }
>();

function compositeBind(
    eid: number,
    read: GPUTextureView,
    write: GPUTextureView,
    seed: GPUTextureView,
    attr: GPUTextureView,
): GPUBindGroup {
    const cached = _composite.get(eid);
    if (
        cached &&
        cached.read === read &&
        cached.write === write &&
        cached.seed === seed &&
        cached.attr === attr
    )
        return cached.group;
    const group = Compute.device.createBindGroup({
        label: `outline-composite/${eid}`,
        layout: _gpu.compositeLayout!,
        entries: [
            { binding: 0, resource: read },
            { binding: 1, resource: seed },
            { binding: 2, resource: attr },
            { binding: 3, resource: write },
        ],
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
    const device = Compute.device;
    const t = targets(camEid, view.width, view.height);
    const maskPipe = occlude ? _gpu.maskOcclude! : _gpu.maskPlain!;
    const maskLayout = occlude ? _gpu.maskLayoutOcclude! : _gpu.maskLayoutPlain!;
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
    mask.setPipeline(maskPipe);
    const viewOffset = [view.slot * VIEW_STRIDE];
    for (const g of groups) {
        if (!g.mesh.position || !g.mesh.quant) continue; // un-quantized producer — nothing to outline
        const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: Render.viewBuffer, size: VIEW_STRIDE } },
            { binding: 1, resource: { buffer: g.mesh.position } },
            { binding: 2, resource: { buffer: g.mesh.indices } },
            { binding: 3, resource: { buffer: transforms } },
            { binding: 4, resource: { buffer: _gpu.eids! } },
            { binding: 5, resource: { buffer: _gpu.attrs! } },
            { binding: 7, resource: { buffer: g.mesh.quant } },
        ];
        if (occlude) entries.push({ binding: 6, resource: view.depth! });
        mask.setBindGroup(
            0,
            device.createBindGroup({
                label: `outline-mask/${camEid}`,
                layout: maskLayout,
                entries,
            }),
            viewOffset,
        );
        mask.draw(g.mesh.indexCount, g.count, g.mesh.indexBase, g.first);
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
        pass.setPipeline(_gpu.jfa!);
        pass.setBindGroup(
            0,
            device.createBindGroup({
                label: `outline-jfa/${camEid}`,
                layout: _gpu.jfaLayout!,
                entries: [
                    { binding: 0, resource: srcView },
                    { binding: 1, resource: { buffer: _gpu.steps!, size: JFA_BYTES } },
                ],
            }),
            [k * JFA_STRIDE],
        );
        pass.draw(3);
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
    composite.setPipeline(_gpu.composite!);
    composite.setBindGroup(0, compositeBind(camEid, read, write, srcView, t.attrView));
    composite.dispatchWorkgroups(
        Math.ceil(view.width / WORKGROUP),
        Math.ceil(view.height / WORKGROUP),
    );
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
        for (let k = 0; k < steps.length; k++) {
            _step[0] = steps[k];
            Compute.device.queue.writeBuffer(_gpu.steps!, k * JFA_STRIDE, _step);
        }

        for (const camEid of state.query([Camera])) {
            const view = Views.get(camEid);
            if (!view?.framebuffer) continue;
            // occlusion needs sear's Depth lane; without it, degrade to always-on-top
            renderOutline(camEid, view, transforms, groups, steps, occlude && !!view.depth);
        }
    },
};

async function prepareOutline(device: GPUDevice): Promise<void> {
    const maskEntries = (occlude: boolean): GPUBindGroupLayoutEntry[] => {
        const e: GPUBindGroupLayoutEntry[] = [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: VIEW_BYTES },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" },
            },
            // the per-mesh quant table — the vs dequantizes the position stream against it
            {
                binding: 7,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage" },
            },
        ];
        if (occlude)
            e.push({
                binding: 6,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "depth" },
            });
        return e;
    };
    _gpu.maskLayoutPlain = device.createBindGroupLayout({
        label: "outline-mask",
        entries: maskEntries(false),
    });
    _gpu.maskLayoutOcclude = device.createBindGroupLayout({
        label: "outline-mask-occlude",
        entries: maskEntries(true),
    });
    _gpu.jfaLayout = device.createBindGroupLayout({
        label: "outline-jfa",
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "uint" } },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: JFA_BYTES },
            },
        ],
    });
    _gpu.compositeLayout = device.createBindGroupLayout({
        label: "outline-composite",
        entries: [
            // the resolved scene (sceneTransform read) — float, like fog's scene binding (works for the
            // rg11b10ufloat offscreen and the rgba16float scratch alike), then the JFA seed + attr, then the
            // rgba16float scratch the band is written into (sceneTransform write)
            { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: { access: "write-only", format: "rgba16float" },
            },
        ],
    });

    _gpu.steps = device.createBuffer({
        label: "outline-jfa-steps",
        size: JFA_STRIDE * MAX_JFA_PASSES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const maskTargets: GPUColorTargetState[] = [{ format: SEED_FORMAT }, { format: ATTR_FORMAT }];
    const maskPrimitive: GPUPrimitiveState = { topology: "triangle-list", cullMode: "back" };
    const fullscreen: GPUPrimitiveState = { topology: "triangle-list", cullMode: "none" };

    const maskModule = device.createShaderModule({ label: "outline-mask", code: maskCode(false) });
    const maskOccludeModule = device.createShaderModule({
        label: "outline-mask-occlude",
        code: maskCode(true),
    });
    const jfaModule = device.createShaderModule({ label: "outline-jfa", code: JFA_WGSL });
    const compositeModule = device.createShaderModule({
        label: "outline-composite",
        code: COMPOSITE_WGSL,
    });

    const [maskPlain, maskOcclude, jfa, composite] = await Promise.all([
        device.createRenderPipelineAsync({
            label: "outline-mask",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_gpu.maskLayoutPlain] }),
            vertex: { module: maskModule, entryPoint: "vs" },
            fragment: { module: maskModule, entryPoint: "fs", targets: maskTargets },
            primitive: maskPrimitive,
        }),
        device.createRenderPipelineAsync({
            label: "outline-mask-occlude",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_gpu.maskLayoutOcclude] }),
            vertex: { module: maskOccludeModule, entryPoint: "vs" },
            fragment: { module: maskOccludeModule, entryPoint: "fs", targets: maskTargets },
            primitive: maskPrimitive,
        }),
        device.createRenderPipelineAsync({
            label: "outline-jfa",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_gpu.jfaLayout] }),
            vertex: { module: jfaModule, entryPoint: "vs" },
            fragment: { module: jfaModule, entryPoint: "fs", targets: [{ format: SEED_FORMAT }] },
            primitive: fullscreen,
        }),
        device.createComputePipelineAsync({
            label: "outline-composite",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_gpu.compositeLayout] }),
            compute: { module: compositeModule, entryPoint: "main" },
        }),
    ]);
    _gpu.maskPlain = maskPlain;
    _gpu.maskOcclude = maskOcclude;
    _gpu.jfa = jfa;
    _gpu.composite = composite;
}

function disposeOutline(): void {
    _gpu.eids?.destroy();
    _gpu.attrs?.destroy();
    _gpu.steps?.destroy();
    for (const t of _targets.values()) {
        t.seedA.destroy();
        t.seedB.destroy();
        t.attr.destroy();
    }
    _targets.clear();
    _composite.clear();
    _gpu.eids = null;
    _gpu.attrs = null;
    _gpu.steps = null;
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
        await prepareOutline(Compute.device);
    },

    dispose() {
        disposeOutline();
    },
};
