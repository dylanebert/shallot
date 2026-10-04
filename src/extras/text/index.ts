import { component } from "../../engine";
// Text — the shallot SDF-text producer. A retained `Text` component (string content, font, size,
// anchor, color) lays each label out into instanced glyph quads, drawn as a standard `"alpha"` world-space
// surface (one draw per font atlas). The glyph buffer holds glyph-local positions + the owning entity id;
// the VS resolves the entity's dense row through `globalTransformRows` each frame, so placement flows through GlobalTransform
// and triggers no glyph rebuild — the buffer rebuilds only when a layout-affecting field changes (a
// content / size / anchor / color edit, an add / remove), gated by a per-frame signature. The SDF atlas /
// font / layout substance (atlas.ts / font.ts / sdf.ts) is renderer-agnostic; this file is the shallot
// surface + producer around it. Single-channel SDF (Valve "Improved Alpha-Tested Magnification").

import type { StorageFlag, TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Meshes, MeshPlugin, registerMesh } from "../../core/mesh";
import { BeginFrameSystem, PrepassSystem, RenderingPlugin } from "../../core/rendering";
import {
    f32,
    GlobalTransform,
    type Plugin,
    Registry,
    type System,
    u32,
    vec2,
    type World,
} from "../../engine";
import { packColor } from "../../engine/utils";
import {
    DrawIndexedIndirect,
    Draws,
    registerSurface,
    StandardRenderingPlugin,
} from "../../standard/rendering";
import {
    createGlyphAtlas,
    disposeAtlases,
    ensureString,
    type GlyphAtlas,
    layoutText,
} from "./atlas";
import { type Font, loadFont } from "./font";
import { GLYPH_AT, GLYPH_BYTES, GLYPH_FLOATS, Glyph } from "./glyph";
import { initializeSdfState, resetPipelines } from "./sdf";
import { atlasName, textSurface, textVaryings } from "./surface";

// Inter, the default face when the consumer registers no font of its own
const DEFAULT_FONT =
    "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf";

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

// one surface + draw + atlas texture per font. The glyph buffer + sampler are shared (one name each); only
// the atlas texture binding is per-font, so its name carries the id. The default single-font case is one
// surface "text0" binding "textAtlas0"
const surfaceName = (id: number) => `text${id}`;
// the unit quad standard instances per glyph: posU.xyz = (corner.x, corner.y, 0); normalV unused
// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

// initial glyph capacity; the CPU staging + GPU buffer double on demand (long paragraphs push thousands)
const INITIAL = 1 << 12;

interface LabelGlyph {
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

interface TextState {
    loaded: (Font | null)[];
    atlases: (GlyphAtlas | null)[];
    sampler: GPUSampler | null;
    glyphBuf: (TgpuBuffer<d.WgslArray<typeof Glyph>> & StorageFlag) | null;
    argBuf:
        | (TgpuBuffer<d.WgslArray<typeof DrawIndexedIndirect>> & { usableAsIndirect: true })
        | null;
    staging: ArrayBuffer;
    f32: Float32Array;
    u32: Uint32Array;
    cap: number;
    count: number;
    quadBase: number;
    sig: number;
    byFont: LabelGlyph[][];
    ranges: { start: number; count: number }[];
    bits: Float32Array;
    bitsU: Uint32Array;
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
        argBuf: null,
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        cap: INITIAL,
        count: 0,
        quadBase: 0,
        sig: -1,
        byFont: [],
        ranges: [],
        bits,
        bitsU: new Uint32Array(bits.buffer),
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
    for (const eid of world.query([Text, GlobalTransform])) {
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
// order (each font's draw indexes its contiguous range via firstInstance), grow + upload the GPU buffer,
// and write each font's indirect record. Runs only on a signature change
function rebuild(world: World, device: GPUDevice): void {
    const _textState = world.resource(textStateKey);

    while (_textState.byFont.length < _textState.atlases.length) _textState.byFont.push([]);
    while (_textState.ranges.length < _textState.atlases.length)
        _textState.ranges.push({ start: 0, count: 0 });
    for (let i = 0; i < _textState.atlases.length; i++) _textState.byFont[i].length = 0;

    for (const eid of world.query([Text, GlobalTransform])) {
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
        for (const g of layout.glyphs) {
            _textState.byFont[fontId].push({
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
        _textState.ranges[id].start = n;
        for (const g of _textState.byFont[id] ?? []) {
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
        _textState.ranges[id].count = n - _textState.ranges[id].start;
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

    _textState.argBuf!.write(
        Array.from({ length: Math.max(1, _textState.atlases.length) }, (_, id) => ({
            indexCount: 6,
            instanceCount: _textState.atlases[id] ? _textState.ranges[id].count : 0,
            firstIndex: _textState.quadBase,
            baseVertex: 0,
            firstInstance: _textState.atlases[id] ? _textState.ranges[id].start : 0,
        })),
    );
}

// runs before standard reads the glyph buffer (the VS positions glyphs from it), so it pins before:
// [PrepassSystem] like any geometry producer. Skips the rebuild when the signature is unchanged
const TextSystem: System = {
    name: "text",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    setup(world: World) {
        const _textState = world.resource(textStateKey);

        _textState.quadBase = world.resource(Meshes).get("textQuad")?.indexBase ?? 0;
        for (let id = 0; id < _textState.atlases.length; id++) {
            if (!_textState.atlases[id]) continue;
            world.resource(Draws).register({
                name: `text${id}`,
                surface: surfaceName(id),
                mesh: "textQuad",
                args: { indirect: _textState.argBuf!, offset: id * 20 },
            });
        }
    },
    update(world) {
        const _textState = world.resource(textStateKey);

        if (
            !world.gpu.device ||
            !_textState.glyphBuf ||
            !_textState.argBuf ||
            _textState.atlases.length === 0
        )
            return;
        const sig = signature(world);
        if (sig === _textState.sig) return;
        _textState.sig = sig;
        rebuild(world, world.gpu.device);
    },
};

const ASCII_CACHE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()";

/**
 * the shallot text producer: the retained {@link Text} component laid out into instanced SDF glyph quads,
 * drawn as a standard `"alpha"` world-space surface (one draw per font). Register fonts with {@link registerFont} and
 * label strings with {@link internText}. Depends on {@link RenderingPlugin}; a StandardRenderer camera renders it
 */
export const TextPlugin: Plugin = {
    name: "Text",
    components: [Text],
    systems: [TextSystem],
    dependencies: [MeshPlugin, RenderingPlugin, StandardRenderingPlugin],

    async initialize(world) {
        const _textState = world.resource(textStateKey);
        const _fonts = world.resource(Fonts);

        _textState;
        initializeSdfState(world);
        _textState.loaded = [];
        _textState.atlases = [];
        _textState.glyphBuf = null;
        _textState.argBuf = null;
        _textState.sampler = null;
        _textState.sig = -1;

        if (!world.gpu.device) return;
        const device = world.gpu.device;

        if (_fonts.size === 0) registerFont(world, DEFAULT_FONT);

        registerMesh(world, { name: "textQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });

        await Promise.all(
            Array.from({ length: _fonts.size }, async (_, id) => {
                const _fonts = world.resource(Fonts);
                const _textState = world.resource(textStateKey);

                const url = _fonts.get(_fonts.name(id)!)!.url;
                try {
                    _textState.loaded[id] = await loadFont(url);
                } catch (e) {
                    console.warn(`[Text] font ${id} (${url}) failed to load:`, e);
                    _textState.loaded[id] = null;
                }
            }),
        );

        _textState.sampler = device.createSampler({
            label: "text",
            magFilter: "linear",
            minFilter: "linear",
        });
        world.gpu.samplers.set("textSamp", _textState.sampler);

        for (let id = 0; id < _textState.loaded.length; id++) {
            const loaded = _textState.loaded[id];
            if (!loaded) continue;
            const atlas = createGlyphAtlas(device, loaded);
            _textState.atlases[id] = atlas;
            world.gpu.textures.set(atlasName(id), atlas.texture);
            const { layout, vs, fs } = textSurface(id);
            registerSurface(world, {
                name: surfaceName(id),
                layout,
                fragmentInputs: { localPos: true },
                blend: "alpha",
                varyings: textVaryings,
                vs,
                fs,
            });
        }
    },

    warm(world: World) {
        const _textState = world.resource(textStateKey);

        if (!world.gpu.device) return;
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
        _textState.argBuf = world.gpu.root
            .createBuffer(d.arrayOf(DrawIndexedIndirect, Math.max(1, _textState.atlases.length)))
            .$usage("indirect")
            .$name("shallot-text-args");
        for (const atlas of _textState.atlases) if (atlas) ensureString(world, atlas, ASCII_CACHE);
    },

    dispose(world: World) {
        const _textState = world.resource(textStateKey);

        _textState.glyphBuf?.destroy();
        _textState.argBuf?.destroy();
        disposeAtlases(_textState.atlases);
        resetPipelines(world);
        _textState.glyphBuf = null;
        _textState.argBuf = null;
        _textState.atlases = [];
        _textState.loaded = [];
        _textState.count = 0;
    },
};
