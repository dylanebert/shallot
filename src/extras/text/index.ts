import { registration } from "../../engine";
// Text — the shallot SDF-text producer. A retained `Text` component (string content, font, size,
// anchor, color) lays each label out into instanced glyph quads, drawn as a sear `"alpha"` world-space
// surface (one draw per font atlas). The glyph buffer holds glyph-local positions + the owning entity id;
// the VS resolves the entity's dense row through `globalTransformRows` each frame, so placement flows through GlobalTransform
// and triggers no glyph rebuild — the buffer rebuilds only when a layout-affecting field changes (a
// content / size / anchor / color edit, an add / remove), gated by a per-frame signature. The SDF atlas /
// font / layout substance (atlas.ts / font.ts / sdf.ts) is renderer-agnostic; this file is the shallot
// surface + producer around it. Single-channel SDF (Valve "Improved Alpha-Tested Magnification").

import type { StorageFlag, TgpuBuffer } from "typegpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Meshes, MeshPlugin, registerMesh } from "../../core/mesh";
import {
    BeginFrameSystem,
    DrawIndexedIndirect,
    Draws,
    fsCtxSchema,
    RenderingPlugin,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "../../core/rendering";
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

import { packColor, Xform, xformPoint } from "../../engine/utils";
import { RenderPrepassesSystem } from "../../standard/rendering";
import {
    createGlyphAtlas,
    disposeAtlases,
    ensureString,
    type GlyphAtlas,
    layoutText,
} from "./atlas";
import { type Font, loadFont } from "./font";
import {
    GLYPH_AT,
    GLYPH_BYTES,
    GLYPH_FLOATS,
    Glyph,
    sdfToSignedDistance,
    textSrgbToLinear,
} from "./glyph";
import { initializeSdfState, resetPipelines } from "./sdf";

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
 *
 * @example
 * ```
 * registerFont(world, "/fonts/inter.ttf", "inter");
 * ```
 */
export function registerFont(world: World, url: string, name?: string): number {
    return world.resource(Fonts).register({ name: name ?? url, url });
}

/**
 * intern a label string, returning the id stored in {@link Text.content}. Identical strings dedupe to one
 * id.
 *
 * @example
 * ```
 * world.storage(Text).content.set(eid, internText(world, "Hello"));
 * ```
 */
export function internText(world: World, content: string): number {
    return world.resource(Content).register({ name: content });
}

/**
 * a world-space text label anchored to an entity's {@link Transform}. Register the string with
 * {@link internText} and, optionally, a face with {@link registerFont}; the glyphs lay out once and ride the entity's
 * transform, so moving a label triggers no rebuild
 *
 * @example
 * ```
 * const label = world.create();
 * world.add(label, Text, {
 *     content: internText(world, "Score"),
 *     fontSize: 0.5,
 *     anchor: [0.5, 0.5],
 *     color: 0xffcc44,
 * });
 * world.add(label, Transform);
 * ```
 */
export const Text = {
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
};

// one surface + draw + atlas texture per font. The glyph buffer + sampler are shared (one name each); only
// the atlas texture binding is per-font, so its name carries the id. The default single-font case is one
// surface "text0" binding "textAtlas0"
const surfaceName = (id: number) => `text${id}`;
const atlasName = (id: number) => `textAtlas${id}`;

// the two custom interstage slots (within the 4-slot custom budget): `uvSize` folds the mixed atlas uv
// (`.xy`) and the world quad size (`.zw`, what the fs's AA math scales `fwidth(localPos)` by) into one
// vec4 — `vsPatchSchema` has no `uv` field to override (only `world`/`worldNormal`/`clip` + varyings), so
// the atlas uv can't ride the built-in. `gcolor` unpacks `unpack4x8unorm` in the vs (a per-instance
// constant, so it interpolates exactly) rather than crossing the packed u32 and unpacking per-fragment.
const textVaryings = { uvSize: d.vec4f, gcolor: d.vec4f };

// per-font typed surface: a fresh `surfaceLayout` per id (the atlas texture binding's name carries the
// id, so each font gets its own layout object, and a vs/fs built against one layout can't be shared with
// another's). localPos.xy is the quad corner (0,0)..(1,1); signed-distance edge AA decodes the SDF to a
// world-space signed distance, faded over one screen-space derivative either side of the glyph edge
// (Valve "Improved Alpha-Tested Magnification"); fully-transparent texels discard before the blend
/** @internal Shader factory shared by registration and placement verification. */
export function typedTextSurface(id: number) {
    const atlasKey = atlasName(id);
    const layout = surfaceLayout({
        textGlyphs: { type: "storage", element: Glyph },
        globalTransforms: { type: "storage", element: Xform },
        globalTransformRows: { type: "storage", element: d.u32 },
        textSamp: { type: "sampler" },
        [atlasKey]: { type: "texture-2d" },
    });
    // `vsPatchSchema`/`fsCtxSchema` are plain host functions (no "use gpu"), so they must be called OUTSIDE
    // any traced body — a call from inside a "use gpu" closure throws "not marked with the 'use gpu'
    // directive" at pipeline-resolution time (`standard/rendering/forward.ts`'s `typedVertexPatch` is the
    // reference pattern). Hoisted once here, the vs body below references the constructor only
    const VertexPatch = vsPatchSchema(textVaryings);
    const vs = tgpu
        .fn(
            [VsIn],
            VertexPatch,
        )((vsIn) => {
            "use gpu";
            const g = Glyph(layout.$.textGlyphs[vsIn.iid]);
            const encodedRow = layout.$.globalTransformRows[g.eid];
            // Like the instance packer, omit geometry without a placement. All corners collapse.
            if (encodedRow === 0) {
                return VertexPatch({
                    world: d.vec4f(0),
                    worldNormal: d.vec3f(0),
                    clip: d.vec4f(0),
                    uvSize: d.vec4f(0),
                    gcolor: d.vec4f(0),
                } as never);
            }
            const x = Xform(layout.$.globalTransforms[encodedRow - 1]);
            const corner = vsIn.localPos.xy;
            const gp = d.vec3f(
                g.pos.x + corner.x * g.size.x,
                g.pos.y + corner.y * g.size.y,
                g.pos.z,
            );
            const uv = std.mix(g.uvRect.xy, g.uvRect.zw, corner);
            return VertexPatch({
                world: d.vec4f(xformPoint(x, gp), 1),
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(0),
                uvSize: d.vec4f(uv, g.size),
                gcolor: std.unpack4x8unorm(g.color),
            } as never);
        })
        .$name(`text${id}Vs`);

    const fs = tgpu
        .fn(
            [fsCtxSchema(textVaryings)],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            // the atlas texture key is per-font (computed), so `layout.$`'s mapped type can't narrow it
            // the way a fixed key like `layout.$.textSamp` resolves automatically — one cast to the
            // runtime texture-sample representation `$` exposes for a fixed `texture-2d` binding. Read
            // here, inside the traced body: `layout.$[atlasKey]` executes the TypeGPU view accessor for
            // real, which only resolves inside an active codegen/dispatch context — reading it at
            // factory-call time (module JS, before any trace) throws "outside of codegen mode" on a real
            // device (the untyped resolve path bun test exercises doesn't reach the accessor at all).
            // Passed straight into the call, never bound to a `const` first — a texture/sampler handle's
            // snippet origin is untyped-pointer-incompatible ("handle"), and TypeGPU's const-declaration
            // codegen tries to take a pointer to any aliased (non-copyable) RHS, so a `const atlas = ...`
            // binding throws "Creating pointer type from origin handle" at pipeline-resolution time.
            const sdf = std.textureSample(
                (layout.$ as unknown as Record<string, d.texture2d<d.F32>>)[atlasKey],
                layout.$.textSamp,
                ctx.uvSize.xy,
            ).x;
            const gsize = ctx.uvSize.zw;
            const maxDim = std.max(gsize.x, gsize.y);
            const signedDist = sdfToSignedDistance(sdf, maxDim);
            const aa = std.length(std.fwidth(std.mul(ctx.localPos.xy, gsize))) * 0.5;
            const alpha = std.smoothstep(aa, -aa, signedDist);
            if (alpha < 0.01) {
                std.discard();
            }
            return d.vec4f(textSrgbToLinear(ctx.gcolor.xyz), ctx.gcolor.w * alpha);
        })
        .$name(`text${id}Fs`);

    return { layout, vs, fs };
}

// the unit quad sear instances per glyph: posU.xyz = (corner.x, corner.y, 0); normalV unused
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

function _textState(world: World): TextState {
    return world.resource(textStateKey);
}

// bitcast scratch + an fnv-1a fold over the layout-affecting fields. The transform is deliberately absent
// — it flows through the slab, so moving a label leaves the signature (and the glyph buffer) untouched
function fbits(world: World, v: number): number {
    const _textState = world.resource(textStateKey);

    _textState.bits[0] = v;
    return _textState.bitsU[0];
}
function fold(h: number, x: number): number {
    return Math.imul(h ^ x, 16777619);
}

// the dirty key: every visible label's layout-affecting state + membership. Equal to last frame ⇒ the
// glyph buffer still holds the right geometry, so the rebuild + upload are skipped
function signature(world: World): number {
    let h = 0x811c9dc5 | 0;
    for (const eid of world.query([Text, GlobalTransform])) {
        if (!world.storage(Text).visible.get(eid)) continue;
        h = fold(h, eid);
        h = fold(h, world.storage(Text).content.get(eid));
        h = fold(h, world.storage(Text).font.get(eid));
        h = fold(h, fbits(world, world.storage(Text).fontSize.get(eid)));
        h = fold(h, fbits(world, world.storage(Text).anchor.x.get(eid)));
        h = fold(h, fbits(world, world.storage(Text).anchor.y.get(eid)));
        h = fold(h, world.storage(Text).color.get(eid));
        h = fold(h, fbits(world, world.storage(Text).opacity.get(eid)));
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

// runs before sear reads the glyph buffer (the VS positions glyphs from it), so it pins before:
// [RenderPrepassesSystem] like any geometry producer. Skips the rebuild when the signature is unchanged
const TextSystem: System = {
    name: "text",
    group: "draw",
    after: [BeginFrameSystem],
    before: [RenderPrepassesSystem],
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
 * drawn as a sear `"alpha"` world-space surface (one draw per font). Register fonts with {@link registerFont} and
 * label strings with {@link internText}. Depends on {@link RenderingPlugin}; a StandardRenderer camera renders it
 */
export const TextPlugin: Plugin = {
    name: "Text",
    components: [
        registration("Text", Text, {
            defaults: () => ({
                content: 0,
                font: 0,
                fontSize: 1,
                opacity: 1,
                visible: 1,
                anchor: [0, 0],
                color: 0xffffff,
            }),
        }),
    ],
    systems: [TextSystem],
    dependencies: [MeshPlugin, RenderingPlugin],

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
            const { layout, vs, fs } = typedTextSurface(id);
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
