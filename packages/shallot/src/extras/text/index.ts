// Text — the kitchen SDF-text producer. A retained `Text` component (string content, font, size,
// anchor, color) lays each label out into instanced glyph quads, drawn as a sear `"alpha"` world-space
// surface (one draw per font atlas). The glyph buffer holds glyph-local positions + the owning entity id;
// the VS reads `transforms[eid]` per frame, so moving a labeled entity flows through the Transform slab
// and triggers no glyph rebuild — the buffer rebuilds only when a layout-affecting field changes (a
// content / size / anchor / color edit, an add / remove), gated by a per-frame signature. The SDF atlas /
// font / layout substance (atlas.ts / font.ts / sdf.ts) is renderer-agnostic; this file is the kitchen
// surface + producer around it. Single-channel SDF (Valve "Improved Alpha-Tested Magnification").

import {
    Compute,
    f32,
    formatHex,
    type Plugin,
    Registry,
    type State,
    type System,
    sparse,
    u32,
    vec2,
} from "../../engine";
import { packColor } from "../../engine/utils/core";
import { mesh, RenderPlugin } from "../../standard/render";
import { BeginFrameSystem, Draws, Meshes, Surfaces } from "../../standard/render/core";
import { PrepassSystem } from "../../standard/sear/core";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import { createGlyphAtlas, ensureString, type GlyphAtlas, layoutText } from "./atlas";
import { type Font, loadFont } from "./font";

// Inter, the default face when the consumer registers no font of its own
const DEFAULT_FONT =
    "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf";

/** registered fonts, keyed by name (the url when unnamed); the id is the atlas slot */
const Fonts = new Registry<{ name: string; url: string }>();
/** interned label strings; id 0 is the empty string (the `Text.content` default) */
const Content = new Registry<{ name: string }>();
Content.register({ name: "" });

/**
 * register a font by url, returning its id. `name` (optional) is the handle a scene's `font:` attribute
 * resolves; unnamed fonts key by url. Call before `build` (or in `setup`) so the atlas loads at init
 *
 * @example
 * ```
 * font("/fonts/inter.ttf", "inter");
 * ```
 */
export function font(url: string, name?: string): number {
    return Fonts.register({ name: name ?? url, url });
}

/**
 * intern a label string, returning the id stored in {@link Text.content}. Identical strings dedupe to one
 * id. Scene `content:` attributes intern through here; programmatic authors call it directly
 *
 * @example
 * ```
 * Text.content.set(eid, text("Hello"));
 * ```
 */
export function text(content: string): number {
    return Content.register({ name: content });
}

/**
 * a world-space text label anchored to an entity's {@link Transform}. Register the string with
 * {@link text} and, optionally, a face with {@link font}; the glyphs lay out once and ride the entity's
 * transform, so moving a label triggers no rebuild
 *
 * @example
 * ```
 * <a text="content: Score; font-size: 0.5; anchor: 0.5 0.5; color: 0xffcc44" transform />
 * ```
 */
export const Text = {
    /** interned string id (see {@link text}); a scene's `content:` interns the raw string */
    content: sparse(u32),
    /** registered font id (see {@link font}); 0 is the default face */
    font: sparse(u32),
    /** world height of one em */
    fontSize: sparse(f32),
    /** 0..1 opacity multiplier */
    opacity: sparse(f32),
    /** drawn when nonzero */
    visible: sparse(f32),
    /** 0..1 pivot within the label; 0 0 = bottom-left, 0.5 0.5 centered */
    anchor: sparse(vec2),
    /** hex sRGB glyph color */
    color: sparse(f32),
};

// one glyph instance = a glyph-local quad origin + owning eid, the atlas uv rect, the world-space quad
// size, and a packed sRGBA color. 48 bytes / three vec4 reads. `pos.xyz` shares its 16-byte slot with
// `eid` (matching the line segment's pos+width packing)
const GLYPH_FLOATS = 12;
const GLYPH_BYTES = 48;
// initial glyph capacity; the CPU staging + GPU buffer double on demand (long paragraphs push thousands)
const INITIAL = 1 << 12;

// struct + helpers spliced at module scope. Single-channel SDF decode (Valve, exponent-encoded like the
// legacy text shader) and the sRGB→linear the packed color needs before blending into the linear target
const GLYPH_WGSL = /* wgsl */ `
struct Glyph {
    pos: vec3<f32>,
    eid: u32,
    uvRect: vec4<f32>,
    size: vec2<f32>,
    color: u32,
    _pad: u32,
}

const SDF_EXPONENT: f32 = 9.0;

fn sdfToSignedDistance(sdf: f32, maxDimension: f32) -> f32 {
    let a = select(sdf, 1.0 - sdf, sdf > 0.5);
    let absDist = (1.0 - pow(2.0 * a, 1.0 / SDF_EXPONENT)) * maxDimension;
    return absDist * select(1.0, -1.0, sdf > 0.5);
}

fn textSrgbToLinear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}
`;

// localPos.xy is the quad corner (0,0)..(1,1). Build the glyph's local rect, apply the owning entity's
// world matrix (sear projects `view.viewProj * world` after), and interpolate the atlas uv across the
// corner. `gsize` carries the world quad size for the FS antialias; `gcolor` the packed sRGBA
const TEXT_VS = /* wgsl */ `
let g = textGlyphs[iid];
let corner = localPos.xy;
let gp = vec3<f32>(g.pos.x + corner.x * g.size.x, g.pos.y + corner.y * g.size.y, g.pos.z);
world = vec4<f32>(xformPoint(transforms[g.eid], gp), 1.0);
uv = mix(g.uvRect.xy, g.uvRect.zw, corner);
gsize = g.size;
gcolor = g.color;
`;

// signed-distance edge AA: the SDF decodes to a world-space signed distance, faded over one screen-space
// derivative either side of the glyph edge (Valve). Fully-transparent texels discard before the blend
function textFs(atlas: string): string {
    return /* wgsl */ `
let sdf = textureSample(${atlas}, textSamp, uv).r;
let maxDim = max(gsize.x, gsize.y);
let signedDist = sdfToSignedDistance(sdf, maxDim);
let aa = length(fwidth(localPos.xy * gsize)) * 0.5;
let alpha = smoothstep(aa, -aa, signedDist);
if (alpha < 0.01) { discard; }
let unp = unpack4x8unorm(gcolor);
col = vec4<f32>(textSrgbToLinear(unp.rgb), unp.a * alpha);
`;
}

// one surface + draw + atlas texture per font. The glyph buffer + sampler are shared (one name each); only
// the atlas texture binding is per-font, so its name carries the id. The default single-font case is one
// surface "text0" binding "textAtlas0"
const surfaceName = (id: number) => `text${id}`;
const atlasName = (id: number) => `textAtlas${id}`;

function textSurface(id: number) {
    const atlas = atlasName(id);
    return {
        name: surfaceName(id),
        blend: "alpha" as const,
        bindings: {
            textGlyphs: { type: "storage" as const, element: "Glyph" },
            transforms: { type: "storage" as const, element: "Xform" },
            [atlas]: { type: "texture-2d" as const },
            textSamp: { type: "sampler" as const },
        },
        interpolators: { gsize: "vec2<f32>", gcolor: "u32" },
        preamble: GLYPH_WGSL,
        vs: TEXT_VS,
        fs: textFs(atlas),
    };
}

// the unit quad sear instances per glyph: posU.xyz = (corner.x, corner.y, 0); normalV unused
// prettier-ignore
const QUAD_VERTS = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
]);
const QUAD_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

let _loaded: (Font | null)[] = [];
let _atlases: (GlyphAtlas | null)[] = [];
let _sampler: GPUSampler | null = null;
let _glyphBuf: GPUBuffer | null = null;
let _argBuf: GPUBuffer | null = null;
let _staging = new ArrayBuffer(INITIAL * GLYPH_BYTES);
let _f32 = new Float32Array(_staging);
let _u32 = new Uint32Array(_staging);
let _cap = INITIAL;
let _count = 0;
let _quadBase = 0;
// the last signature an upload was built for; -1 forces the first frame's build
let _sig = -1;
// per-font glyph staging lists, rebuilt each dirty frame, then packed into the shared buffer in id order
const _byFont: Glyph[][] = [];
const _ranges: { start: number; count: number }[] = [];
const _args = new Uint32Array(5);

interface Glyph {
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

// bitcast scratch + an fnv-1a fold over the layout-affecting fields. The transform is deliberately absent
// — it flows through the slab, so moving a label leaves the signature (and the glyph buffer) untouched
const _bits = new Float32Array(1);
const _bitsU = new Uint32Array(_bits.buffer);
function fbits(v: number): number {
    _bits[0] = v;
    return _bitsU[0];
}
function fold(h: number, x: number): number {
    return Math.imul(h ^ x, 16777619);
}

// the dirty key: every visible label's layout-affecting state + membership. Equal to last frame ⇒ the
// glyph buffer still holds the right geometry, so the rebuild + upload are skipped
function signature(state: State): number {
    let h = 0x811c9dc5 | 0;
    for (const eid of state.query([Text, Transform])) {
        if (!Text.visible.get(eid)) continue;
        h = fold(h, eid);
        h = fold(h, Text.content.get(eid));
        h = fold(h, Text.font.get(eid));
        h = fold(h, fbits(Text.fontSize.get(eid)));
        h = fold(h, fbits(Text.anchor.x.get(eid)));
        h = fold(h, fbits(Text.anchor.y.get(eid)));
        h = fold(h, Text.color.get(eid));
        h = fold(h, fbits(Text.opacity.get(eid)));
    }
    return h;
}

function grow(min: number): void {
    let cap = _cap;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * GLYPH_BYTES);
    new Uint8Array(next).set(new Uint8Array(_staging, 0, _count * GLYPH_BYTES));
    _staging = next;
    _f32 = new Float32Array(next);
    _u32 = new Uint32Array(next);
    _cap = cap;
}

// lay every visible label out into per-font glyph lists, pack them into the shared staging in font-id
// order (each font's draw indexes its contiguous range via firstInstance), grow + upload the GPU buffer,
// and write each font's indirect record. Runs only on a signature change
function rebuild(state: State, device: GPUDevice): void {
    while (_byFont.length < _atlases.length) _byFont.push([]);
    while (_ranges.length < _atlases.length) _ranges.push({ start: 0, count: 0 });
    for (let i = 0; i < _atlases.length; i++) _byFont[i].length = 0;

    for (const eid of state.query([Text, Transform])) {
        if (!Text.visible.get(eid)) continue;
        const content = Content.name(Text.content.get(eid));
        if (!content) continue;
        let fontId = Text.font.get(eid);
        if (!_atlases[fontId]) fontId = 0;
        const atlas = _atlases[fontId];
        if (!atlas) continue;
        ensureString(atlas, content);
        const layout = layoutText(content, atlas, Text.fontSize.get(eid));
        const ox = -layout.width * Text.anchor.x.get(eid);
        const oy = -layout.height * Text.anchor.y.get(eid);
        const color = packColor(Text.color.get(eid), Text.opacity.get(eid));
        for (const g of layout.glyphs) {
            _byFont[fontId].push({
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
    for (let id = 0; id < _atlases.length; id++) total += _byFont[id]?.length ?? 0;
    if (total > _cap) grow(total);

    let n = 0;
    for (let id = 0; id < _atlases.length; id++) {
        _ranges[id].start = n;
        for (const g of _byFont[id] ?? []) {
            const o = n * GLYPH_FLOATS;
            _f32[o] = g.x;
            _f32[o + 1] = g.y;
            _f32[o + 2] = 0;
            _u32[o + 3] = g.eid;
            _f32[o + 4] = g.u0;
            _f32[o + 5] = g.v0;
            _f32[o + 6] = g.u1;
            _f32[o + 7] = g.v1;
            _f32[o + 8] = g.w;
            _f32[o + 9] = g.h;
            _u32[o + 10] = g.color;
            n++;
        }
        _ranges[id].count = n - _ranges[id].start;
    }
    _count = n;

    if (_cap * GLYPH_BYTES > _glyphBuf!.size) {
        const stale = _glyphBuf!;
        _glyphBuf = device.createBuffer({
            label: "kitchen-text-glyphs",
            size: _cap * GLYPH_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("textGlyphs", _glyphBuf);
        device.queue.onSubmittedWorkDone().then(() => stale.destroy());
    }
    if (_count > 0) device.queue.writeBuffer(_glyphBuf!, 0, _staging, 0, _count * GLYPH_BYTES);

    for (let id = 0; id < _atlases.length; id++) {
        if (!_atlases[id]) continue;
        _args[0] = 6;
        _args[1] = _ranges[id].count;
        _args[2] = _quadBase;
        _args[3] = 0;
        _args[4] = _ranges[id].start;
        device.queue.writeBuffer(_argBuf!, id * 20, _args);
    }
}

// runs before sear reads the glyph buffer (the VS positions glyphs from it), so it pins before:
// [PrepassSystem] like any geometry producer. Skips the rebuild when the signature is unchanged
const TextSystem: System = {
    name: "text",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    setup() {
        _quadBase = Meshes.get("textQuad")?.indexBase ?? 0;
        for (let id = 0; id < _atlases.length; id++) {
            if (!_atlases[id]) continue;
            Draws.register({
                name: `text${id}`,
                surface: surfaceName(id),
                mesh: "textQuad",
                args: { indirect: _argBuf!, offset: id * 20 },
            });
        }
    },
    update(state) {
        if (!Compute.device || !_glyphBuf || !_argBuf || _atlases.length === 0) return;
        const sig = signature(state);
        if (sig === _sig) return;
        _sig = sig;
        rebuild(state, Compute.device);
    },
};

const ASCII_CACHE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()";

/**
 * the kitchen text producer: the retained {@link Text} component laid out into instanced SDF glyph quads,
 * drawn as a sear `"alpha"` world-space surface (one draw per font). Register fonts with {@link font} and
 * label strings with {@link text}. Depends on {@link RenderPlugin}; a Sear camera renders it
 */
export const TextPlugin: Plugin = {
    name: "Text",
    components: { Text },
    systems: [TextSystem],
    dependencies: [RenderPlugin, TransformsPlugin],
    traits: {
        Text: {
            requires: [Transform],
            defaults: () => ({
                content: 0,
                font: 0,
                fontSize: 1,
                opacity: 1,
                visible: 1,
                anchor: [0, 0],
                color: 0xffffff,
            }),
            parse: {
                font: (name: string) => Fonts.id(name) ?? 0,
                content: (raw: string) => text(raw),
            },
            format: {
                color: formatHex,
                content: (id: number) => Content.name(id) ?? "",
            },
        },
    },

    async initialize() {
        _loaded = [];
        _atlases = [];
        _glyphBuf = null;
        _argBuf = null;
        _sampler = null;
        _sig = -1;

        if (!Compute.device) return;
        const device = Compute.device;

        if (Fonts.size === 0) font(DEFAULT_FONT);

        mesh({ name: "textQuad", vertices: QUAD_VERTS, indices: QUAD_INDICES });

        await Promise.all(
            Array.from({ length: Fonts.size }, async (_, id) => {
                const url = Fonts.get(Fonts.name(id)!)!.url;
                try {
                    _loaded[id] = await loadFont(url);
                } catch (e) {
                    console.warn(`[Text] font ${id} (${url}) failed to load:`, e);
                    _loaded[id] = null;
                }
            }),
        );

        _sampler = device.createSampler({
            label: "text",
            magFilter: "linear",
            minFilter: "linear",
        });
        Compute.samplers.set("textSamp", _sampler);

        for (let id = 0; id < _loaded.length; id++) {
            const loaded = _loaded[id];
            if (!loaded) continue;
            const atlas = createGlyphAtlas(device, loaded);
            _atlases[id] = atlas;
            Compute.textures.set(atlasName(id), atlas.texture);
            Surfaces.register(textSurface(id));
        }
    },

    warm() {
        if (!Compute.device) return;
        const device = Compute.device;
        _cap = INITIAL;
        _staging = new ArrayBuffer(INITIAL * GLYPH_BYTES);
        _f32 = new Float32Array(_staging);
        _u32 = new Uint32Array(_staging);
        _count = 0;
        _sig = -1;
        _glyphBuf = device.createBuffer({
            label: "kitchen-text-glyphs",
            size: INITIAL * GLYPH_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        Compute.buffers.set("textGlyphs", _glyphBuf);
        _argBuf = device.createBuffer({
            label: "kitchen-text-args",
            // one DrawIndexedIndirect record per font; COPY_SRC so a gym Mirror can read back instanceCount
            size: Math.max(1, _atlases.length) * 20,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        for (const atlas of _atlases) if (atlas) ensureString(atlas, ASCII_CACHE);
    },

    dispose() {
        _glyphBuf?.destroy();
        _argBuf?.destroy();
        for (const atlas of _atlases) atlas?.texture.destroy();
        _glyphBuf = null;
        _argBuf = null;
        _atlases = [];
        _loaded = [];
        _count = 0;
    },
};
