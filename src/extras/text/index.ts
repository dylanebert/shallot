import { component } from "../../engine";
// Text — the shallot SDF-text producer. A retained `Text` component (string content, font, size,
// anchor, color) lays each label out into instanced glyph quads, drawn in core's transparent phase (one
// draw per label). The glyph buffer holds glyph-local positions + the owning entity id; the VS resolves
// the entity's dense row through `globalTransformRows` each frame, so placement flows through
// GlobalTransform and triggers no glyph rebuild.
// The SDF atlas / font / layout substance remains renderer-agnostic. Single-channel SDF (Valve "Improved
// Alpha-Tested Magnification").

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Meshes, type MeshHandle, MeshPlugin } from "../../core/mesh";
import {
    BeginFrameSystem,
    Camera,
    CorePipelinePlugin,
    globalTransformTable,
    PrepassSystem,
    RenderContext,
    RenderPhases,
    type TransparentPhase,
    type View,
    type Xform,
} from "../../core/rendering";
import { composeGlobalTransform, GlobalTransform } from "../../core/transform";
import { f32, type Plugin, Registry, type System, u32, vec2, type World } from "../../engine";
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
import { createTextPipeline, textLayout, viewLayout } from "./pipeline";
import { resetPipelines } from "./sdf";

/** registered fonts, keyed by name (the url when unnamed); the id is the atlas slot */
export const Fonts = { create: () => new Registry<{ name: string; url: string }>() };
/** interned label strings; id 0 is the empty string (the `Text.content` default) */
export const Content = {
    create: () => {
        const content = new Registry<{ name: string }>();
        content.register({ name: "" });
        return content;
    },
};

/**
 * register a font by url, returning the id stored in {@link Text.font}. `name` (optional) is its
 * {@link Fonts} key; unnamed fonts key by url. Register in the owning World during `setup` so the atlas loads at init
 */
export function registerFont(world: World, url: string, name?: string): number {
    return world.resource(Fonts).register({ name: name ?? url, url });
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
 * transform, so moving a label triggers no rebuild
 */
export const Text = component(
    "Text",
    {
        /** interned string id (see {@link internText}) */
        content: u32,
        /** registered font id (see {@link registerFont}); 0 is the default face */
        font: u32,
        /** world height of one em */
        fontSize: f32,
        /** 0..1 opacity multiplier */
        opacity: f32,
        /** drawn when nonzero */
        visible: f32,
        /** 0..1 pivot within the label; 0 0 = bottom-left, 0.5 0.5 centered */
        anchor: vec2,
        /** hex sRGB glyph color */
        color: f32,
    },
    {
        defaults: () => ({
            content: 0,
            font: 0,
            fontSize: 1,
            opacity: 1,
            visible: 1,
            anchor: [0, 0],
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

// initial glyph capacity; the CPU staging + GPU buffer double on demand (long paragraphs push thousands)
const INITIAL = 1 << 12;

interface LabelGlyph {
    label: number;
    eid: number;
    x: number;
    y: number;
    w: number;
    h: number;
    u0: number;
    v0: number;
    u1: number;
    v1: number;
    color: number;
}

interface TextLabelDraw {
    eid: number;
    font: number;
    start: number;
    count: number;
    centerX: number;
    centerY: number;
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
    sig: number;
    byFont: LabelGlyph[][];
    labelDraws: TextLabelDraw[];
    lastPipeline: import("typegpu").TgpuRenderPipeline | null;
    lastViewGroup: GPUBindGroup | null;
    lastFontGroup: GPUBindGroup | null;
    bits: Float32Array;
    bitsU: Uint32Array;
    transformScratch: Float32Array;
    colorPipelines: {
        single: import("typegpu").TgpuRenderPipeline | null;
        multisample: import("typegpu").TgpuRenderPipeline | null;
    };
    bindGroups: (GPUBindGroup | null)[];
    bindKeys: (object[] | null)[];
    viewGroups: Map<number, { buffer: GPUBuffer; quant: object; group: GPUBindGroup }>;
}

const textStateKey = { create: createTextState };

function createTextState(): TextState {
    const staging = new ArrayBuffer(INITIAL * GLYPH_BYTES);
    const bits = new Float32Array(1);
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
        sig: -1,
        byFont: [],
        labelDraws: [],
        lastPipeline: null,
        lastViewGroup: null,
        lastFontGroup: null,
        bits,
        bitsU: new Uint32Array(bits.buffer),
        transformScratch: new Float32Array(16),
        colorPipelines: { single: null, multisample: null },
        bindGroups: [],
        bindKeys: [],
        viewGroups: new Map(),
    };
}

// bitcast scratch + an fnv-1a fold over the layout-affecting fields. The transform is deliberately absent
// — it flows through the slab, so moving a label leaves the signature (and the glyph buffer) untouched
function fbits(v: number, state: TextState): number {
    state.bits[0] = v;
    return state.bitsU[0];
}
function fold(h: number, x: number): number {
    return Math.imul(h ^ x, 16777619);
}

// the dirty key: every visible label's layout-affecting state + membership. Equal to last frame ⇒ the
// glyph buffer still holds the right geometry, so the rebuild + upload are skipped
function signature(world: World): number {
    const scratch = world.resource(textStateKey);
    const text = world.storage(Text);
    let h = 0x811c9dc5 | 0;
    for (const eid of world.query(TEXT_TRANSFORM_QUERY)) {
        if (!text.visible.get(eid)) continue;
        h = fold(h, eid);
        h = fold(h, text.content.get(eid));
        h = fold(h, text.font.get(eid));
        h = fold(h, fbits(text.fontSize.get(eid), scratch));
        h = fold(h, fbits(text.anchor.x.get(eid), scratch));
        h = fold(h, fbits(text.anchor.y.get(eid), scratch));
        h = fold(h, text.color.get(eid));
        h = fold(h, fbits(text.opacity.get(eid), scratch));
    }
    return h;
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

// lay every visible label out into per-font glyph lists, pack them into the shared staging in font-id
// order (each label's draw indexes its contiguous range via firstInstance), then grow + upload the GPU
// buffer. Runs only on a signature change
function rebuild(world: World, device: GPUDevice): void {
    const _textState = world.resource(textStateKey);
    _textState.labelDraws.length = 0;

    while (_textState.byFont.length < _textState.atlases.length) _textState.byFont.push([]);
    for (let i = 0; i < _textState.atlases.length; i++) _textState.byFont[i].length = 0;

    for (const eid of world.query(TEXT_TRANSFORM_QUERY)) {
        if (!world.storage(Text).visible.get(eid)) continue;
        const content = world.resource(Content).name(world.storage(Text).content.get(eid));
        if (!content) continue;
        let fontId = world.storage(Text).font.get(eid);
        if (!_textState.atlases[fontId]) fontId = 0;
        const atlas = _textState.atlases[fontId];
        if (!atlas) continue;
        ensureString(world, atlas, content);
        const layout = layoutText(content, atlas, world.storage(Text).fontSize.get(eid));
        const ox = -layout.width * world.storage(Text).anchor.x.get(eid);
        const oy = -layout.height * world.storage(Text).anchor.y.get(eid);
        const color = packColor(
            world.storage(Text).color.get(eid),
            world.storage(Text).opacity.get(eid),
        );
        const label = layout.glyphs.length
            ? _textState.labelDraws.push({
                  eid,
                  font: fontId,
                  start: 0,
                  count: 0,
                  centerX: ox + layout.width * 0.5,
                  centerY: oy + layout.height * 0.5,
              }) - 1
            : -1;
        for (const g of layout.glyphs) {
            _textState.byFont[fontId].push({
                label,
                eid,
                x: ox + g.x,
                y: oy + g.y,
                w: g.width,
                h: g.height,
                u0: g.u0,
                v0: g.v0,
                u1: g.u1,
                v1: g.v1,
                color,
            });
        }
    }

    let total = 0;
    for (let id = 0; id < _textState.atlases.length; id++)
        total += _textState.byFont[id]?.length ?? 0;
    if (total > _textState.cap) grow(world, total);

    let n = 0;
    for (let id = 0; id < _textState.atlases.length; id++) {
        for (const g of _textState.byFont[id] ?? []) {
            const label = _textState.labelDraws[g.label];
            if (label.count === 0) label.start = n;
            label.count++;
            const o = n * GLYPH_FLOATS;
            _textState.f32[o + GLYPH_AT.pos] = g.x;
            _textState.f32[o + GLYPH_AT.pos + 1] = g.y;
            _textState.f32[o + GLYPH_AT.pos + 2] = 0;
            _textState.u32[o + GLYPH_AT.eid] = g.eid;
            _textState.f32[o + GLYPH_AT.uvRect] = g.u0;
            _textState.f32[o + GLYPH_AT.uvRect + 1] = g.v0;
            _textState.f32[o + GLYPH_AT.uvRect + 2] = g.u1;
            _textState.f32[o + GLYPH_AT.uvRect + 3] = g.v1;
            _textState.f32[o + GLYPH_AT.size] = g.w;
            _textState.f32[o + GLYPH_AT.size + 1] = g.h;
            _textState.u32[o + GLYPH_AT.color] = g.color;
            n++;
        }
    }
    _textState.count = n;

    if (_textState.cap * GLYPH_BYTES > world.gpu.root.unwrap(_textState.glyphBuf!).size) {
        const stale = _textState.glyphBuf!;
        _textState.glyphBuf = world.gpu.root
            .createBuffer(d.arrayOf(Glyph, _textState.cap))
            .$usage("storage")
            .$name("shallot-text-glyphs");
        world.gpu.buffers.set("textGlyphs", world.gpu.root.unwrap(_textState.glyphBuf));
        world.gpu.typed.set("textGlyphs", _textState.glyphBuf);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    if (_textState.count > 0)
        device.queue.writeBuffer(
            world.gpu.root.unwrap(_textState.glyphBuf!),
            0,
            _textState.staging,
            0,
            _textState.count * GLYPH_BYTES,
        );
}

// runs before standard reads the glyph buffer (the VS positions glyphs from it), so it pins before:
// [PrepassSystem] like any geometry producer. Skips the rebuild when the signature is unchanged
const TextSystem: System = {
    name: "text",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(world) {
        const _textState = world.resource(textStateKey);

        if (!_textState.glyphBuf || _textState.atlases.length === 0) return;
        const sig = signature(world);
        if (sig === _textState.sig) return;
        _textState.sig = sig;
        rebuild(world, world.gpu.device);
    },
};

const ASCII_CACHE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()";
function queueText(world: World, _eid: number, _view: View, phase: TransparentPhase): void {
    const state = world.resource(textStateKey);
    if (!state.glyphBuf || !state.quad) return;
    const position = phase.positionScratch;
    for (let i = 0; i < state.labelDraws.length; i++) {
        const label = state.labelDraws[i];
        if (!label.count || !world.exists(label.eid) || !world.has(label.eid, GlobalTransform))
            continue;
        composeGlobalTransform(world, label.eid, state.transformScratch);
        const transform = state.transformScratch;
        const x = transform[12] + transform[0] * label.centerX + transform[4] * label.centerY;
        const y = transform[13] + transform[1] * label.centerX + transform[5] * label.centerY;
        const z = transform[14] + transform[2] * label.centerX + transform[6] * label.centerY;
        position[0] = x;
        position[1] = y;
        position[2] = z;
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

function ensureTextFontGroup(world: World, state: TextState, font: number): GPUBindGroup | null {
    if (!state.glyphBuf || !state.quad) return null;
    const mesh = world.resource(Meshes).get(state.quad);
    const atlas = state.atlases[font];
    const table = globalTransformTable(world);
    const rows = table.eidToRowTyped;
    if (!mesh?.quant || !atlas || !rows) return null;
    let group = state.bindGroups[font];
    const previous = state.bindKeys[font];
    if (
        !group ||
        !previous ||
        previous[0] !== mesh.vertices ||
        previous[1] !== mesh.quant ||
        previous[2] !== state.glyphBuf ||
        previous[3] !== table.typed ||
        previous[4] !== rows ||
        previous[5] !== atlas.texture ||
        previous[6] !== state.sampler
    ) {
        group = world.gpu.root.unwrap(
            world.gpu.root.createBindGroup(textLayout, {
                vertices: mesh.vertices,
                glyphs: state.glyphBuf,
                globalTransforms: table.typed as TgpuBuffer<d.WgslArray<typeof Xform>> &
                    StorageFlag,
                globalTransformRows: rows as TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag,
                atlas: atlas.texture.createView(),
                sampler: state.sampler!,
            }),
        );
        state.bindGroups[font] = group;
        state.bindKeys[font] = [
            mesh.vertices,
            mesh.quant,
            state.glyphBuf,
            table.typed,
            rows,
            atlas.texture,
            state.sampler!,
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
    const pipeline =
        world.storage(Camera).antialias.get(eid) !== 0
            ? state.colorPipelines.multisample
            : state.colorPipelines.single;
    if (!pipeline) return;
    for (let i = 0; i < phase.length; i++) {
        const item = phase.item(i);
        if (item.renderer === renderer) ensureTextFontGroup(world, state, item.batch);
    }
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
    const pipeline =
        world.storage(Camera).antialias.get(eid) !== 0
            ? state.colorPipelines.multisample
            : state.colorPipelines.single;
    if (!pipeline) return;
    const previous = start > 0 ? phase.item(start - 1) : undefined;
    if (previous?.renderer !== renderer) {
        state.lastPipeline = null;
        state.lastViewGroup = null;
        state.lastFontGroup = null;
    }
    const fontGroup = ensureTextFontGroup(world, state, phase.item(start).batch);
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
    for (let i = start; i < end; i++) {
        const label = state.labelDraws[phase.item(i).draw];
        pass.drawIndexed(6, label.count, mesh.indexBase, 0, label.start);
        phase.metrics.draws++;
    }
}

/**
 * the shallot text producer: the retained {@link Text} component laid out into instanced SDF glyph quads,
 * drawn in core's transparent phase (one draw per label). Register fonts with {@link registerFont} and
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
        _textState.sig = -1;

        const device = world.gpu.device;

        if (_fonts.size === 0) registerFont(world, DEFAULT_FONT);

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
        _textState.sig = -1;
        _textState.glyphBuf = world.gpu.root
            .createBuffer(d.arrayOf(Glyph, INITIAL))
            .$usage("storage")
            .$name("shallot-text-glyphs");
        world.gpu.buffers.set("textGlyphs", world.gpu.root.unwrap(_textState.glyphBuf));
        world.gpu.typed.set("textGlyphs", _textState.glyphBuf);
        const format = world.resource(RenderContext).format;
        _textState.colorPipelines = {
            single: createTextPipeline(world.gpu.root, format, 1),
            multisample: createTextPipeline(world.gpu.root, format, 4),
        };
        _textState.viewGroups.clear();
        _textState.bindGroups.length = _textState.atlases.length;
        _textState.bindKeys.length = _textState.atlases.length;
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
        _textState.bindGroups = [];
        _textState.bindKeys = [];
        _textState.viewGroups.clear();
        _textState.colorPipelines = { single: null, multisample: null };
    },
};
