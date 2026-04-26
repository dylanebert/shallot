import { type Font, loadFont } from "./font";
import { type GlyphAtlas, createGlyphAtlas, ensureString, layoutText } from "./atlas";
import {
    resource,
    traits,
    buf,
    capacity,
    CHUNK_SHIFT,
    CHUNK_MASK,
    type Plugin,
    type State,
    type System,
} from "../../engine";
import {
    createColorProxy,
    createFieldProxy,
    formatHex,
    type FieldProxy,
} from "../../engine/ecs/core";
import { Compute, ComputePlugin } from "../../standard/compute";
import type { GBuf } from "../../standard/compute";
import { registry } from "../../engine";
import { Render, RenderPlugin } from "../../standard/render";
import {
    Z_FORMAT,
    SCENE_STRUCT_WGSL,
    type OverlayDraw,
    type SharedPassContext,
} from "../../standard/render/core";
import { Transform } from "../../standard/transforms";

const MAX_GLYPHS = 50000;
const GLYPH_FLOATS = 16;
const MAX_FONTS = 64;
export const fontRegistry = registry<string>(MAX_FONTS);
const loadedFonts: (Font | null)[] = [];

const DEFAULT_FONT =
    "https://fonts.gstatic.com/s/inter/v20/UcCO3FwrK3iLTeHuS_nVMrMxCp50SjIw2boKoduKmMEVuLyfMZg.ttf";

export function font(url: string, name?: string): number {
    const id = fontRegistry.add(url, name);
    loadedFonts.push(null);
    return id;
}

async function loadFonts(): Promise<void> {
    const urls = fontRegistry.all();
    await Promise.all(
        urls.map(async (url, id) => {
            loadedFonts[id] = await loadFont(url);
        }),
    );
}

export const TextData = buf(Float32Array, 12, 0);
export const TextFonts = buf(Uint32Array, 1, 0);

const textContent = new Map<number, string>();

interface TextContentProxy {
    [eid: number]: string | undefined;
}

function contentProxy(): TextContentProxy {
    return new Proxy({} as TextContentProxy, {
        get(_, prop) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return textContent.get(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            if (value === undefined || value === null) {
                textContent.delete(eid);
            } else {
                textContent.set(eid, value);
            }
            return true;
        },
    });
}

export const Text: {
    content: TextContentProxy;
    font: FieldProxy;
    fontSize: FieldProxy;
    opacity: FieldProxy;
    visible: FieldProxy;
    anchorX: FieldProxy;
    anchorY: FieldProxy;
    color: FieldProxy;
    colorR: FieldProxy;
    colorG: FieldProxy;
    colorB: FieldProxy;
} = {
    content: contentProxy(),
    font: createFieldProxy(TextFonts, 1, 0),
    fontSize: createFieldProxy(TextData, 12, 0),
    opacity: createFieldProxy(TextData, 12, 1),
    visible: createFieldProxy(TextData, 12, 2),
    anchorX: createFieldProxy(TextData, 12, 3),
    anchorY: createFieldProxy(TextData, 12, 4),
    color: createColorProxy(TextData, 12, 8),
    colorR: createFieldProxy(TextData, 12, 8),
    colorG: createFieldProxy(TextData, 12, 9),
    colorB: createFieldProxy(TextData, 12, 10),
};

traits(Text, {
    requires: [Transform],
    defaults: () => ({
        font: 0,
        fontSize: 1,
        opacity: 1,
        visible: 1,
        anchorX: 0,
        anchorY: 0,
        color: 0xffffff,
    }),
    parse: { font: fontRegistry.getByName },
    format: { color: formatHex },
});

const textShader = /* wgsl */ `
${SCENE_STRUCT_WGSL}

struct GlyphInstance {
    posX: f32,
    posY: f32,
    posZ: f32,
    entityId: u32,
    width: f32,
    height: f32,
    texelWidth: f32,
    texelHeight: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read> glyphs: array<GlyphInstance>;
@group(0) @binding(2) var atlasTexture: texture_2d<f32>;
@group(0) @binding(3) var atlasSampler: sampler;
@group(0) @binding(4) var<storage, read> matrices: array<mat4x4<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) localUV: vec2<f32>,
    @location(3) glyphDimensions: vec2<f32>,
    @location(4) @interpolate(flat) entityId: u32,
}

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let glyphIdx = vid / 6u;
    let cornerIdx = vid % 6u;

    let glyph = glyphs[glyphIdx];

    var localPos: vec2<f32>;
    var uv: vec2<f32>;

    switch cornerIdx {
        case 0u: {
            localPos = vec2(0.0, 0.0);
            uv = vec2(glyph.u0, glyph.v0);
        }
        case 1u: {
            localPos = vec2(1.0, 0.0);
            uv = vec2(glyph.u1, glyph.v0);
        }
        case 2u: {
            localPos = vec2(1.0, 1.0);
            uv = vec2(glyph.u1, glyph.v1);
        }
        case 3u: {
            localPos = vec2(0.0, 0.0);
            uv = vec2(glyph.u0, glyph.v0);
        }
        case 4u: {
            localPos = vec2(1.0, 1.0);
            uv = vec2(glyph.u1, glyph.v1);
        }
        case 5u: {
            localPos = vec2(0.0, 1.0);
            uv = vec2(glyph.u0, glyph.v1);
        }
        default: {
            localPos = vec2(0.0);
            uv = vec2(0.0);
        }
    }

    let localPos3 = vec3(
        glyph.posX + localPos.x * glyph.width,
        glyph.posY + localPos.y * glyph.height,
        glyph.posZ
    );

    let transform = matrices[glyph.entityId];
    let worldPos = transform * vec4(localPos3, 1.0);

    var out: VertexOutput;
    out.position = scene.viewProj * worldPos;
    out.uv = uv;
    out.color = glyph.color;
    out.localUV = localPos;
    out.glyphDimensions = vec2(glyph.width, glyph.height);
    out.entityId = glyph.entityId;
    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) mask: f32,
    @location(2) entityId: u32,
}

const SDF_EXPONENT: f32 = 9.0;

fn sdfToSignedDistance(sdfValue: f32, maxDimension: f32) -> f32 {
    let alpha = select(sdfValue, 1.0 - sdfValue, sdfValue > 0.5);
    let absDist = (1.0 - pow(2.0 * alpha, 1.0 / SDF_EXPONENT)) * maxDimension;
    return absDist * select(1.0, -1.0, sdfValue > 0.5);
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
    let sdfValue = textureSample(atlasTexture, atlasSampler, input.uv).r;

    let maxDimension = max(input.glyphDimensions.x, input.glyphDimensions.y);
    let signedDist = sdfToSignedDistance(sdfValue, maxDimension);

    let aaDist = length(fwidth(input.localUV * input.glyphDimensions)) * 0.5;

    let alpha = smoothstep(aaDist, -aaDist, signedDist);

    let fxaaSpan = aaDist * 8.0;
    let inMaskRegion = signedDist < fxaaSpan;

    if alpha < 0.01 && !inMaskRegion {
        discard;
    }

    var out: FragmentOutput;
    out.color = vec4(input.color.rgb, input.color.a * alpha);
    out.mask = select(0.0, 1.0, inMaskRegion);
    out.entityId = input.entityId;
    return out;
}
`;

function createTextPipeline(
    device: GPUDevice,
    format: GPUTextureFormat,
    maskFormat: GPUTextureFormat,
    eidFormat: GPUTextureFormat,
): GPURenderPipeline {
    const module = device.createShaderModule({ code: textShader });

    return device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module,
            entryPoint: "vs",
        },
        fragment: {
            module,
            entryPoint: "fs",
            targets: [
                {
                    format,
                    blend: {
                        color: {
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                        alpha: {
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                            operation: "add",
                        },
                    },
                },
                {
                    format: maskFormat,
                    writeMask: GPUColorWrite.RED,
                },
                {
                    format: eidFormat,
                },
            ],
        },
        primitive: {
            topology: "triangle-list",
            cullMode: "none",
        },
        depthStencil: {
            format: Z_FORMAT,
            depthCompare: "less",
            depthWriteEnabled: true,
        },
    });
}

function createTextDraw(
    render: { scene: GPUBuffer; matrices: GBuf },
    glyphBuffer: GPUBuffer,
    atlasView: GPUTextureView,
    sampler: GPUSampler,
    fontIndex: number,
    range: FontRange,
): OverlayDraw {
    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let cachedCapacity = capacity();

    return {
        order: 2 + fontIndex,

        draw(pass: GPURenderPassEncoder, ctx: SharedPassContext) {
            if (capacity() !== cachedCapacity) {
                cachedCapacity = capacity();
                bindGroup = null;
            }
            if (range.count === 0) return;

            if (!pipeline) {
                pipeline = createTextPipeline(
                    ctx.device,
                    ctx.format,
                    ctx.maskFormat,
                    ctx.eidFormat,
                );
            }

            if (!bindGroup) {
                bindGroup = ctx.device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: render.scene } },
                        { binding: 1, resource: { buffer: glyphBuffer } },
                        { binding: 2, resource: atlasView },
                        { binding: 3, resource: sampler },
                        { binding: 4, resource: { buffer: render.matrices.buffer } },
                    ],
                });
            }

            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(range.count * 6, 1, range.start * 6, 0);
        },
    };
}

interface FontRange {
    start: number;
    count: number;
}

interface Glyphs {
    atlases: GlyphAtlas[];
    sampler: GPUSampler;
    buffer: GPUBuffer;
    staging: Float32Array;
    stagingU32: Uint32Array;
    ranges: FontRange[];
}

const Glyphs = resource<Glyphs>("glyphs");

interface PendingGlyph {
    eid: number;
    x: number;
    y: number;
    width: number;
    height: number;
    texelWidth: number;
    texelHeight: number;
    u0: number;
    v0: number;
    u1: number;
    v1: number;
    r: number;
    g: number;
    b: number;
    a: number;
}

const glyphsByFont: PendingGlyph[][] = [];

const TextSystem: System = {
    group: "draw",
    annotations: { mode: "always" },

    update(state: State) {
        const compute = Compute.from(state);
        const text = Glyphs.from(state);
        if (!compute || !text) return;

        const { device } = compute;
        const { atlases, staging, stagingU32, ranges } = text;

        while (glyphsByFont.length < atlases.length) glyphsByFont.push([]);
        for (let i = 0; i < atlases.length; i++) glyphsByFont[i].length = 0;

        const tdChunks = TextData.chunks;
        const tfChunks = TextFonts.chunks;
        for (const eid of state.query([Text, Transform])) {
            const td = tdChunks[eid >>> CHUNK_SHIFT];
            const local = eid & CHUNK_MASK;
            if (!td[local * 12 + 2]) continue;

            const content = textContent.get(eid);
            if (!content) continue;

            const fontId = tfChunks[eid >>> CHUNK_SHIFT][eid & CHUNK_MASK];
            const atlas = atlases[fontId] ?? atlases[0];
            const actualFontId = atlases[fontId] ? fontId : 0;
            if (!atlas) continue;

            ensureString(atlas, content);

            const fontSize = td[local * 12];
            const layout = layoutText(content, atlas, fontSize);

            const anchorX = td[local * 12 + 3];
            const anchorY = td[local * 12 + 4];
            const offsetX = -layout.width * anchorX;
            const offsetY = -layout.height * anchorY;

            const o = local * 12 + 8;
            const r = td[o];
            const g = td[o + 1];
            const b = td[o + 2];
            const a = td[local * 12 + 1];

            for (const glyph of layout.glyphs) {
                glyphsByFont[actualFontId].push({
                    eid,
                    x: offsetX + glyph.x,
                    y: offsetY + glyph.y,
                    width: glyph.width,
                    height: glyph.height,
                    texelWidth: glyph.texelWidth,
                    texelHeight: glyph.texelHeight,
                    u0: glyph.u0,
                    v0: glyph.v0,
                    u1: glyph.u1,
                    v1: glyph.v1,
                    r,
                    g,
                    b,
                    a,
                });
            }
        }

        let glyphCount = 0;
        for (let fontIdx = 0; fontIdx < atlases.length; fontIdx++) {
            const fontGlyphs = glyphsByFont[fontIdx];
            ranges[fontIdx].start = glyphCount;
            ranges[fontIdx].count = fontGlyphs.length;

            for (const glyph of fontGlyphs) {
                if (glyphCount >= MAX_GLYPHS) break;

                const offset = glyphCount * GLYPH_FLOATS;

                staging[offset + 0] = glyph.x;
                staging[offset + 1] = glyph.y;
                staging[offset + 2] = 0;
                stagingU32[offset + 3] = glyph.eid;

                staging[offset + 4] = glyph.width;
                staging[offset + 5] = glyph.height;
                staging[offset + 6] = glyph.texelWidth;
                staging[offset + 7] = glyph.texelHeight;

                staging[offset + 8] = glyph.u0;
                staging[offset + 9] = glyph.v0;
                staging[offset + 10] = glyph.u1;
                staging[offset + 11] = glyph.v1;

                staging[offset + 12] = glyph.r;
                staging[offset + 13] = glyph.g;
                staging[offset + 14] = glyph.b;
                staging[offset + 15] = glyph.a;

                glyphCount++;
            }
        }

        if (glyphCount > 0) {
            device.queue.writeBuffer(
                text.buffer,
                0,
                staging.buffer,
                0,
                glyphCount * GLYPH_FLOATS * 4,
            );
        }
    },
};

const ASCII_CACHE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-:;'\"()";

export const TextPlugin: Plugin = {
    name: "Text",
    systems: [TextSystem],
    components: { Text },
    dependencies: [ComputePlugin, RenderPlugin],

    warm(state: State) {
        const text = Glyphs.from(state);
        if (!text) return;
        for (const atlas of text.atlases) ensureString(atlas, ASCII_CACHE);
    },

    async initialize(state: State) {
        fontRegistry.clear();
        loadedFonts.length = 0;
        textContent.clear();

        const compute = Compute.from(state);
        const render = Render.from(state);
        if (!compute || !render) return;

        if (fontRegistry.count() === 0) {
            font(DEFAULT_FONT);
        }

        try {
            await loadFonts();
        } catch (e) {
            console.warn("[TextPlugin] Failed to load fonts:", e);
            return;
        }

        const { device } = compute;

        const atlases: GlyphAtlas[] = [];
        for (const loadedFont of loadedFonts) {
            if (loadedFont) {
                atlases.push(createGlyphAtlas(device, loadedFont));
            }
        }

        if (atlases.length === 0) {
            return;
        }

        const sampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
        });

        const ranges: FontRange[] = atlases.map(() => ({ start: 0, count: 0 }));

        const stagingBuf = new Float32Array(MAX_GLYPHS * GLYPH_FLOATS);
        const textState: Glyphs = {
            atlases,
            sampler,
            buffer: device.createBuffer({
                label: "glyphs",
                size: MAX_GLYPHS * GLYPH_FLOATS * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            staging: stagingBuf,
            stagingU32: new Uint32Array(stagingBuf.buffer),
            ranges,
        };

        state.setResource(Glyphs, textState);

        for (let i = 0; i < atlases.length; i++) {
            render.effects.overlay.push(
                createTextDraw(
                    render,
                    textState.buffer,
                    atlases[i].textureView,
                    sampler,
                    i,
                    ranges[i],
                ),
            );
        }
    },
};
