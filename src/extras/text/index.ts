import { component } from "../../engine";
// Placement changes do not rebuild glyph geometry; the vertex shader resolves GlobalTransform rows.

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Meshes, type MeshHandle, MeshPlugin } from "../../core/mesh";
import {
    Anchor,
    Billboard,
    Camera,
    CorePipelinePlugin,
    globalTransformTable,
    initialViewSampleCounts,
    PrepassSystem,
    RenderContext,
    RenderPhases,
    renderLayers,
    type TransparentPhase,
    type View,
    type Xform,
} from "../../core/rendering";
import { composeGlobalTransform, GlobalTransform } from "../../core/transform";
import {
    ClearChangeMarksSystem,
    f32,
    type Plugin,
    precompile,
    Registry,
    type System,
    u32,
    type World,
} from "../../engine";
import { packColor } from "../../engine/utils";
import {
    createGlyphAtlas,
    disposeAtlases,
    ensureString,
    type GlyphAtlas,
    layoutText,
} from "./atlas";
import { DEFAULT_FONT, type Font, loadFont } from "./font";
import { GLYPH_AT, GLYPH_BYTES, GLYPH_FLOATS, Glyph } from "./glyph";
import { Justify } from "./justify";

export { Justify } from "./justify";

import { createTextPipeline, textLayout, viewLayout } from "./pipeline";
import { resetPipelines } from "./sdf";

/** Registered fonts keyed by URL; id 0 is the bundled default face. */
export const Fonts = {
    create: () => {
        const fonts = new Registry<{ name: string; url: string }>();
        fonts.register({ name: DEFAULT_FONT, url: DEFAULT_FONT });
        return fonts;
    },
};
/** interned label strings; id 0 is the empty string (the `Text.content` default) */
export const Content = {
    create: () => {
        const content = new Registry<{ name: string }>();
        content.register({ name: "" });
        return content;
    },
};

/**
 * Register a font by URL, returning the id stored in {@link Text.font}. Repeated URLs return the
 * same id. Register in the owning World during `setup` so the atlas loads at init.
 */
export function registerFont(world: World, url: string): number {
    return world.resource(Fonts).register({ name: url, url });
}

/**
 * intern a label string, returning the id stored in {@link Text.content}. Identical strings dedupe to one
 * id.
 */
export function internText(world: World, content: string): number {
    return world.resource(Content).register({ name: content });
}

/**
 * a world-space text label anchored to an entity's {@link Transform}. Register the string with
 * {@link internText} and, optionally, a face with {@link registerFont}; the glyphs lay out once and ride the entity's
 * transform. Each label retains its packed glyph layout; marked content, font, size, width,
 * justification, colour, opacity or Anchor changes lay out only that label before draw. A loaded
 * font replaces its fallback on the next draw. Placement and billboard changes do not lay out text.
 * The block's bottom-left corner is at the
 * Transform unless the entity has an Anchor. Remove Text to hide it; add it again to show it.
 */
export const Text = component(
    "Text",
    {
        /** interned string id (see {@link internText}) */
        content: u32,
        /** registered font id (see {@link registerFont}); 0 is the default face */
        font: u32,
        /** {@link Billboard} orientation; defaults to World (plain entity transform). */
        billboard: u32,
        /** world height of one em */
        fontSize: f32,
        /** Maximum line width in world units; Infinity (default) is unbounded. Wraps at spaces, never splits a word; \n starts a new line. */
        maxWidth: f32,
        /** {@link Justify} alignment of each line within the block width; defaults to Justify.Left. */
        justify: u32,
        /** 0..1 opacity multiplier */
        opacity: f32,
        /** Clip-depth bias, clamped to [-1, 1]: 0 (default) depth-tests normally, -1 draws on top, 1 moves to the far plane. Negative intermediate values bias exponentially toward the camera. */
        depthBias: f32,
        /** hex sRGB glyph color */
        color: f32,
    },
    {
        defaults: () => ({
            content: 0,
            font: 0,
            billboard: Billboard.World,
            fontSize: 1,
            maxWidth: Infinity,
            justify: Justify.Left,
            opacity: 1,
            depthBias: 0,
            color: 0xffffff,
        }),
    },
);

// One atlas and resource bind group per font; glyph and placement tables are shared.
// the unit quad standard instances per glyph: posU.xyz = (corner.x, corner.y, 0); normalV unused
// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);
const TEXT_TRANSFORM_QUERY = [Text, GlobalTransform];

const INITIAL = 1 << 12;

interface TextLabelDraw {
    eid: number;
    font: number;
    start: number;
    count: number;
    centerX: number;
    centerY: number;
    ref: number;
    atlas: GlyphAtlas | null;
    anchored: boolean;
    seen: boolean;
}

interface TextInstances {
    buffer: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
    indices: Uint32Array;
    starts: Uint32Array;
    counts: Uint32Array;
    bindGroups: (GPUBindGroup | null)[];
    bindKeys: (object[] | null)[];
}

interface TextState {
    loaded: (Font | null)[];
    atlases: (GlyphAtlas | null)[];
    sampler: GPUSampler | null;
    glyphBuf: (TgpuBuffer<d.WgslArray<typeof Glyph>> & StorageFlag) | null;
    staging: ArrayBuffer;
    f32: Float32Array;
    u32: Uint32Array;
    cap: number;
    count: number;
    quad: MeshHandle | null;
    labels: Map<number, TextLabelDraw>;
    labelDraws: TextLabelDraw[];
    lastPipeline: import("typegpu").TgpuRenderPipeline | null;
    lastViewGroup: GPUBindGroup | null;
    lastFontGroup: GPUBindGroup | null;
    transformScratch: Float32Array;
    colorPipelines: {
        single: import("typegpu").TgpuRenderPipeline | null;
        multisample: import("typegpu").TgpuRenderPipeline | null;
    };
    instances: Map<number, TextInstances>;
    viewGroups: Map<number, { buffer: GPUBuffer; quant: object; group: GPUBindGroup }>;
}

export const textStateKey = { create: createTextState };

function createTextState(): TextState {
    const staging = new ArrayBuffer(INITIAL * GLYPH_BYTES);
    return {
        loaded: [],
        atlases: [],
        sampler: null,
        glyphBuf: null,
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        cap: INITIAL,
        count: 0,
        quad: null,
        labels: new Map(),
        labelDraws: [],
        lastPipeline: null,
        lastViewGroup: null,
        lastFontGroup: null,
        transformScratch: new Float32Array(16),
        colorPipelines: { single: null, multisample: null },
        instances: new Map(),
        viewGroups: new Map(),
    };
}

function grow(world: World, min: number): void {
    const _textState = world.resource(textStateKey);

    let cap = _textState.cap;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * GLYPH_BYTES);
    new Uint8Array(next).set(new Uint8Array(_textState.staging, 0, _textState.count * GLYPH_BYTES));
    _textState.staging = next;
    _textState.f32 = new Float32Array(next);
    _textState.u32 = new Uint32Array(next);
    _textState.cap = cap;
}

// Ranges stay packed: resizing moves following glyph bytes, never their layouts.
function resizeLabel(world: World, label: TextLabelDraw, count: number): void {
    const state = world.resource(textStateKey);
    const delta = count - label.count;
    if (!delta) return;
    if (state.count + delta > state.cap) grow(world, state.count + delta);
    const end = label.start + label.count;
    state.u32.copyWithin(
        (end + delta) * GLYPH_FLOATS,
        end * GLYPH_FLOATS,
        state.count * GLYPH_FLOATS,
    );
    let following = false;
    for (const other of state.labelDraws) {
        if (following) other.start += delta;
        if (other === label) following = true;
    }
    label.count = count;
    state.count += delta;
}

function layoutLabel(world: World, label: TextLabelDraw): void {
    const state = world.resource(textStateKey);
    const text = world.storage(Text);
    const eid = label.eid;
    const content = world.resource(Content).name(text.content.get(eid));
    const atlas = label.atlas;
    if (!content || !atlas) {
        resizeLabel(world, label, 0);
        label.centerX = label.centerY = 0;
        return;
    }
    ensureString(world, atlas, content);
    const layout = layoutText(
        content,
        atlas,
        text.fontSize.get(eid),
        text.maxWidth.get(eid),
        text.justify.get(eid),
    );
    const anchor = world.storage(Anchor).offset;
    const ox = -layout.width * (label.anchored ? anchor.x.get(eid) + 0.5 : 0);
    const oy = -layout.height * (label.anchored ? anchor.y.get(eid) + 0.5 : 0);
    const color = packColor(text.color.get(eid), text.opacity.get(eid));
    resizeLabel(world, label, layout.glyphs.length);
    label.centerX = ox + layout.width * 0.5;
    label.centerY = oy + layout.height * 0.5;
    let n = label.start;
    for (const g of layout.glyphs) {
        const o = n++ * GLYPH_FLOATS;
        state.f32[o + GLYPH_AT.pos] = ox + g.x;
        state.f32[o + GLYPH_AT.pos + 1] = oy + g.y;
        state.f32[o + GLYPH_AT.pos + 2] = 0;
        state.u32[o + GLYPH_AT.eid] = eid;
        state.f32[o + GLYPH_AT.uvRect] = g.u0;
        state.f32[o + GLYPH_AT.uvRect + 1] = g.v0;
        state.f32[o + GLYPH_AT.uvRect + 2] = g.u1;
        state.f32[o + GLYPH_AT.uvRect + 3] = g.v1;
        state.f32[o + GLYPH_AT.size] = g.width;
        state.f32[o + GLYPH_AT.size + 1] = g.height;
        state.u32[o + GLYPH_AT.color] = color;
        state.u32[o + GLYPH_AT.billboard] = text.billboard.get(eid);
        state.f32[o + GLYPH_AT.depthBias] = text.depthBias.get(eid);
    }
}

function updateLabels(world: World): void {
    const state = world.resource(textStateKey);
    const text = world.storage(Text);
    const contentDirty = world.fieldStorage(Text, "content").dirty;
    const fontDirty = world.fieldStorage(Text, "font").dirty;
    const sizeDirty = world.fieldStorage(Text, "fontSize").dirty;
    const widthDirty = world.fieldStorage(Text, "maxWidth").dirty;
    const justifyDirty = world.fieldStorage(Text, "justify").dirty;
    const colorDirty = world.fieldStorage(Text, "color").dirty;
    const opacityDirty = world.fieldStorage(Text, "opacity").dirty;
    const anchorDirty = world.fieldStorage(Anchor, "offset").dirty;
    const billboardDirty = world.fieldStorage(Text, "billboard").dirty;
    const depthBiasDirty = world.fieldStorage(Text, "depthBias").dirty;
    let dirtyStart = Infinity;
    let dirtyEnd = 0;
    for (const label of state.labelDraws) label.seen = false;
    for (const eid of world.query(TEXT_TRANSFORM_QUERY)) {
        let font = text.font.get(eid);
        if (!state.atlases[font]) font = 0;
        const atlas = state.atlases[font] ?? null;
        const anchored = world.has(eid, Anchor);
        let label = state.labels.get(eid);
        const fresh = !label || label.ref !== world.ref(eid);
        if (!label) {
            label = {
                eid,
                ref: world.ref(eid),
                font,
                atlas,
                anchored,
                seen: true,
                start: state.count,
                count: 0,
                centerX: 0,
                centerY: 0,
            };
            state.labels.set(eid, label);
            state.labelDraws.push(label);
        }
        label.seen = true;
        const word = eid >>> 5;
        const bit = 1 << (eid & 31);
        if (
            fresh ||
            label.atlas !== atlas ||
            label.anchored !== anchored ||
            ((contentDirty[word] |
                fontDirty[word] |
                sizeDirty[word] |
                widthDirty[word] |
                justifyDirty[word] |
                colorDirty[word] |
                opacityDirty[word] |
                anchorDirty[word]) &
                bit) !==
                0
        ) {
            label.ref = world.ref(eid);
            label.font = font;
            label.atlas = atlas;
            label.anchored = anchored;
            const oldCount = label.count;
            layoutLabel(world, label);
            dirtyStart = Math.min(dirtyStart, label.start);
            dirtyEnd = Math.max(
                dirtyEnd,
                oldCount === label.count ? label.start + label.count : state.count,
            );
        } else if ((billboardDirty[word] | depthBiasDirty[word]) & bit) {
            for (let n = label.start; n < label.start + label.count; n++) {
                state.u32[n * GLYPH_FLOATS + GLYPH_AT.billboard] = text.billboard.get(eid);
                state.f32[n * GLYPH_FLOATS + GLYPH_AT.depthBias] = text.depthBias.get(eid);
            }
            dirtyStart = Math.min(dirtyStart, label.start);
            dirtyEnd = Math.max(dirtyEnd, label.start + label.count);
        }
    }
    for (let i = state.labelDraws.length - 1; i >= 0; i--) {
        const label = state.labelDraws[i];
        if (label.seen) continue;
        dirtyStart = Math.min(dirtyStart, label.start);
        resizeLabel(world, label, 0);
        dirtyEnd = Math.max(dirtyEnd, state.count);
        state.labels.delete(label.eid);
        state.labelDraws.splice(i, 1);
    }
    const device = world.gpu.device;
    if (state.cap * GLYPH_BYTES > world.gpu.root.unwrap(state.glyphBuf!).size) {
        const stale = state.glyphBuf!;
        state.glyphBuf = world.gpu.root
            .createBuffer(d.arrayOf(Glyph, state.cap))
            .$usage("storage")
            .$name("shallot-text-glyphs");
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
        dirtyStart = 0;
        dirtyEnd = state.count;
    }
    dirtyEnd = Math.min(dirtyEnd, state.count);
    if (dirtyStart < dirtyEnd)
        device.queue.writeBuffer(
            world.gpu.root.unwrap(state.glyphBuf!),
            dirtyStart * GLYPH_BYTES,
            state.staging,
            dirtyStart * GLYPH_BYTES,
            (dirtyEnd - dirtyStart) * GLYPH_BYTES,
        );
}

// Publish glyph geometry before the prepass can read it.
const TextSystem: System = {
    name: "text",
    group: "draw",
    first: true,
    before: [ClearChangeMarksSystem, PrepassSystem],
    update(world) {
        const _textState = world.resource(textStateKey);

        if (!_textState.glyphBuf || _textState.atlases.length === 0) return;
        updateLabels(world);
    },
};

const ASCII_CACHE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()";
function queueText(world: World, eid: number, _view: View, phase: TransparentPhase): void {
    const state = world.resource(textStateKey);
    if (!state.glyphBuf || !state.quad) return;
    const position = phase.positionScratch;
    const cameraLayers = renderLayers(world, eid);
    for (let i = 0; i < state.labelDraws.length; i++) {
        const label = state.labelDraws[i];
        if (!label.count || !world.exists(label.eid) || !world.has(label.eid, GlobalTransform))
            continue;
        if (!(renderLayers(world, label.eid) & cameraLayers)) continue;
        composeGlobalTransform(world, label.eid, state.transformScratch);
        const transform = state.transformScratch;
        const mode = world.storage(Text).billboard.get(label.eid);
        if (mode === Billboard.World) {
            position[0] =
                transform[12] + transform[0] * label.centerX + transform[4] * label.centerY;
            position[1] =
                transform[13] + transform[1] * label.centerX + transform[5] * label.centerY;
            position[2] =
                transform[14] + transform[2] * label.centerX + transform[6] * label.centerY;
        } else {
            // Screen corners share the origin's view depth; YLocked's horizontal axis does too.
            position[0] = transform[12];
            position[1] =
                transform[13] +
                (mode === Billboard.YLocked
                    ? label.centerY * Math.hypot(transform[4], transform[5], transform[6])
                    : 0);
            position[2] = transform[14];
        }
        phase.addFromScratch(i, label.font, label.eid, 0);
    }
}

function ensureTextViewGroup(world: World, view: View, state: TextState): GPUBindGroup | null {
    if (!state.quad) return null;
    const mesh = world.resource(Meshes).get(state.quad);
    if (!mesh?.quant) return null;
    const viewBuffer = world.resource(RenderContext).viewBuffers[view.slot];
    if (!viewBuffer) return null;
    let viewGroup = state.viewGroups.get(view.slot);
    if (!viewGroup || viewGroup.buffer !== viewBuffer || viewGroup.quant !== mesh.quant) {
        const group = world.gpu.root.createBindGroup(viewLayout, {
            view: viewBuffer,
            meshQuant: mesh.quant,
        });
        viewGroup = {
            buffer: viewBuffer,
            quant: mesh.quant,
            group: world.gpu.root.unwrap(group),
        };
        state.viewGroups.set(view.slot, viewGroup);
    }
    return viewGroup.group;
}

function ensureTextFontGroup(
    world: World,
    state: TextState,
    instances: TextInstances,
    font: number,
): GPUBindGroup | null {
    if (!state.glyphBuf || !state.quad) return null;
    const mesh = world.resource(Meshes).get(state.quad);
    const atlas = state.atlases[font];
    const table = globalTransformTable(world);
    const rows = table.eidToRowTyped;
    if (!mesh?.quant || !atlas || !rows) return null;
    let group = instances.bindGroups[font];
    const previous = instances.bindKeys[font];
    if (
        !group ||
        !previous ||
        previous[0] !== mesh.vertices ||
        previous[1] !== mesh.quant ||
        previous[2] !== instances.buffer ||
        previous[3] !== table.typed ||
        previous[4] !== rows ||
        previous[5] !== atlas.texture ||
        previous[6] !== state.sampler ||
        previous[7] !== state.glyphBuf
    ) {
        group = world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(textLayout, {
                vertices: mesh.vertices,
                glyphs: state.glyphBuf,
                glyphIndices: instances.buffer,
                globalTransforms: table.typed as TgpuBuffer<d.WgslArray<typeof Xform>> &
                    StorageFlag,
                globalTransformRows: rows as TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag,
                atlas: atlas.texture.createView(),
                sampler: state.sampler!,
            }),
        );
        instances.bindGroups[font] = group;
        instances.bindKeys[font] = [
            mesh.vertices,
            mesh.quant,
            instances.buffer,
            table.typed,
            rows,
            atlas.texture,
            state.sampler!,
            state.glyphBuf,
        ];
    }
    return group;
}

function prepareText(
    world: World,
    eid: number,
    view: View,
    phase: TransparentPhase,
    renderer: number,
): void {
    const state = world.resource(textStateKey);
    if (!ensureTextViewGroup(world, view, state)) return;
    state.lastPipeline = null;
    state.lastViewGroup = null;
    state.lastFontGroup = null;
    const key = world.storage(Camera).antialias.get(eid) ? "multisample" : "single";
    state.colorPipelines[key] ??= createTextPipeline(
        world,
        world.resource(RenderContext).format,
        key === "single" ? 1 : 4,
    );
    const capacity = Math.max(32, 2 ** Math.ceil(Math.log2(phase.length || 1)));
    let instances = state.instances.get(view.slot);
    if (!instances || instances.indices.length < state.cap) {
        const stale = instances?.buffer;
        instances = {
            buffer: world.gpu.root
                .createBuffer(d.arrayOf(d.u32, state.cap))
                .$usage("storage")
                .$name("shallot-text-instances"),
            indices: new Uint32Array(state.cap),
            starts: new Uint32Array(capacity),
            counts: new Uint32Array(capacity),
            bindGroups: [],
            bindKeys: [],
        };
        state.instances.set(view.slot, instances);
        if (stale) world.gpu.device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    if (instances.starts.length < phase.length) {
        instances.starts = new Uint32Array(capacity);
        instances.counts = new Uint32Array(capacity);
    }
    // Retained layouts stay untouched; each view orders only their glyph indices.
    let count = 0;
    for (let i = 0; i < phase.length; i++) {
        const item = phase.item(i);
        if (item.renderer !== renderer) continue;
        const label = state.labelDraws[item.draw];
        instances.starts[i] = count;
        instances.counts[i] = label.count;
        for (let glyph = label.start; glyph < label.start + label.count; glyph++)
            instances.indices[count++] = glyph;
        ensureTextFontGroup(world, state, instances, item.batch);
    }
    if (count)
        world.gpu.device.queue.writeBuffer(
            world.gpu.root.unwrap(instances.buffer),
            0,
            instances.indices.buffer,
            0,
            count * Uint32Array.BYTES_PER_ELEMENT,
        );
}

function drawText(
    world: World,
    eid: number,
    view: View,
    phase: TransparentPhase,
    start: number,
    end: number,
    pass: GPURenderPassEncoder,
    renderer: number,
): void {
    const state = world.resource(textStateKey);
    const viewGroup = ensureTextViewGroup(world, view, state);
    if (!viewGroup || !state.quad) return;
    const mesh = world.resource(Meshes).get(state.quad);
    if (!mesh) return;
    const key = world.storage(Camera).antialias.get(eid) ? "multisample" : "single";
    const pipeline = (state.colorPipelines[key] ??= createTextPipeline(
        world,
        world.resource(RenderContext).format,
        key === "single" ? 1 : 4,
    ));
    const previous = start > 0 ? phase.item(start - 1) : undefined;
    if (previous?.renderer !== renderer) {
        state.lastPipeline = null;
        state.lastViewGroup = null;
        state.lastFontGroup = null;
    }
    const instances = state.instances.get(view.slot);
    if (!instances) return;
    const fontGroup = ensureTextFontGroup(world, state, instances, phase.item(start).batch);
    if (!fontGroup) return;
    pass.setPipeline(world.gpu.root.unwrap(pipeline));
    pass.setBindGroup(0, viewGroup);
    pass.setBindGroup(1, fontGroup);
    pass.setIndexBuffer(world.gpu.root.unwrap(mesh.indices), "uint32");
    if (state.lastPipeline !== pipeline) phase.metrics.pipelineChanges++;
    if (state.lastViewGroup !== viewGroup) phase.metrics.bindGroupChanges++;
    if (state.lastFontGroup !== fontGroup) phase.metrics.bindGroupChanges++;
    state.lastPipeline = pipeline;
    state.lastViewGroup = viewGroup;
    state.lastFontGroup = fontGroup;
    const first = instances.starts[start];
    const last = end - 1;
    const count = instances.starts[last] + instances.counts[last] - first;
    pass.drawIndexed(6, count, mesh.indexBase, 0, first);
    phase.metrics.draws++;
}

/**
 * the shallot text producer: the retained {@link Text} component laid out into instanced SDF glyph quads,
 * drawn in core's transparent phase (one draw per consecutive run of one font). Each label owns a contiguous glyph range;
 * range resizing compacts following bytes without laying out other labels. Storage grows geometrically
 * to the peak live glyph count; unchanged frames allocate and upload nothing for text layout.
 * Register fonts with {@link registerFont} and
 * label strings with {@link internText}. Depends on {@link CorePipelinePlugin}; a core camera renders it
 */
export const TextPlugin: Plugin = {
    gpu: {},
    name: "Text",
    components: [Text],
    systems: [TextSystem],
    dependencies: [MeshPlugin, CorePipelinePlugin],

    async initialize(world) {
        const _textState = world.resource(textStateKey);
        const _fonts = world.resource(Fonts);

        world.resource(RenderPhases).push({
            transparent: queueText,
            prepareTransparent: prepareText,
            drawTransparent: drawText,
        });
        _textState.loaded = [];
        _textState.atlases = [];
        _textState.glyphBuf = null;
        _textState.sampler = null;
        _textState.labels.clear();
        _textState.labelDraws.length = 0;

        const device = world.gpu.device;

        _textState.quad = world.resource(Meshes).add({
            name: "textQuad",
            vertices: QUAD_VERTS,
            indices: QUAD_INDICES,
        });

        await Promise.all(
            Array.from({ length: _fonts.size }, async (_, id) => {
                const _fonts = world.resource(Fonts);
                const _textState = world.resource(textStateKey);

                const url = _fonts.get(_fonts.name(id)!)!.url;
                try {
                    _textState.loaded[id] = await loadFont(url);
                } catch (cause) {
                    throw new Error(`[Text] font ${id} (${url}) failed to load`, { cause });
                }
            }),
        );

        _textState.sampler = device.createSampler({
            label: "text",
            magFilter: "linear",
            minFilter: "linear",
        });
        for (let id = 0; id < _textState.loaded.length; id++) {
            const loaded = _textState.loaded[id];
            if (!loaded) continue;
            const atlas = createGlyphAtlas(device, loaded);
            _textState.atlases[id] = atlas;
        }
    },

    warm(world: World) {
        const _textState = world.resource(textStateKey);

        _textState.cap = INITIAL;
        _textState.staging = new ArrayBuffer(INITIAL * GLYPH_BYTES);
        _textState.f32 = new Float32Array(_textState.staging);
        _textState.u32 = new Uint32Array(_textState.staging);
        _textState.count = 0;
        _textState.labels.clear();
        _textState.labelDraws.length = 0;
        _textState.glyphBuf = world.gpu.root
            .createBuffer(d.arrayOf(Glyph, INITIAL))
            .$usage("storage")
            .$name("shallot-text-glyphs");
        const format = world.resource(RenderContext).format;
        precompile(world, "text-prepare", () => {
            for (const samples of initialViewSampleCounts(world))
                _textState.colorPipelines[samples === 1 ? "single" : "multisample"] =
                    createTextPipeline(world, format, samples);
            return [];
        });
        _textState.viewGroups.clear();
        _textState.instances.clear();
        for (const atlas of _textState.atlases) if (atlas) ensureString(world, atlas, ASCII_CACHE);
    },

    dispose(world: World) {
        const _textState = world.resource(textStateKey);

        _textState.glyphBuf?.destroy();
        disposeAtlases(_textState.atlases);
        resetPipelines(world);
        _textState.glyphBuf = null;
        _textState.atlases = [];
        _textState.loaded = [];
        _textState.count = 0;
        _textState.labels.clear();
        _textState.labelDraws.length = 0;
        for (const instances of _textState.instances.values()) instances.buffer.destroy();
        _textState.instances.clear();
        _textState.viewGroups.clear();
        _textState.colorPipelines = { single: null, multisample: null };
    },
};
