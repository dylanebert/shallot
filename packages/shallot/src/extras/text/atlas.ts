import type { Font } from "./font";
import { SDFGenerator } from "./sdf";

const SDF_SIZE = 96;
const SDF_EXPONENT = 9;

export interface GlyphMetrics {
    width: number;
    height: number;
    glyphWidth: number;
    glyphHeight: number;
    glyphTop: number;
    glyphLeft: number;
    advance: number;
    u0: number;
    v0: number;
    u1: number;
    v1: number;
}

export interface GlyphAtlas {
    texture: GPUTexture;
    textureView: GPUTextureView;
    width: number;
    height: number;
    glyphs: Map<string, GlyphMetrics>;
    rowHeight: number;
    cursorX: number;
    cursorY: number;
    font: Font;
    sdfGenerator: SDFGenerator;
}

export function createGlyphAtlas(device: GPUDevice, font: Font): GlyphAtlas {
    const width = 2048;
    const height = 2048;

    const texture = device.createTexture({
        size: { width, height },
        format: "r8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        label: "glyphAtlas",
    });

    const sdfGenerator = new SDFGenerator({
        device,
        sdfSize: SDF_SIZE,
        exponent: SDF_EXPONENT,
        curveSubdivisions: 24,
    });

    return {
        texture,
        textureView: texture.createView(),
        width,
        height,
        glyphs: new Map(),
        rowHeight: 0,
        cursorX: 0,
        cursorY: 0,
        font,
        sdfGenerator,
    };
}

interface PendingGlyphEntry {
    path: string;
    paddedBounds: [number, number, number, number];
    atlasX: number;
    atlasY: number;
}

function computeGlyphMetrics(atlas: GlyphAtlas, char: string): PendingGlyphEntry | null {
    const path = atlas.font.glyphPath(char);
    const bounds = atlas.font.glyphBounds(char);
    const advance = atlas.font.advance(char);

    if (!path || !bounds) return null;

    const [xMin, yMin, xMax, yMax] = bounds;
    const unitsPerEm = atlas.font.unitsPerEm;

    const padding = unitsPerEm * 0.1;
    const paddedBounds: [number, number, number, number] = [
        xMin - padding,
        yMin - padding,
        xMax + padding,
        yMax + padding,
    ];

    const glyphWidth = paddedBounds[2] - paddedBounds[0];
    const glyphHeight = paddedBounds[3] - paddedBounds[1];

    if (atlas.cursorX + SDF_SIZE > atlas.width) {
        atlas.cursorX = 0;
        atlas.cursorY += atlas.rowHeight;
        atlas.rowHeight = 0;
    }

    if (atlas.cursorY + SDF_SIZE > atlas.height) {
        throw new Error("Glyph atlas full");
    }

    const metrics: GlyphMetrics = {
        width: SDF_SIZE,
        height: SDF_SIZE,
        glyphWidth: glyphWidth / unitsPerEm,
        glyphHeight: glyphHeight / unitsPerEm,
        glyphTop: paddedBounds[3] / unitsPerEm,
        glyphLeft: paddedBounds[0] / unitsPerEm,
        advance: advance / unitsPerEm,
        u0: atlas.cursorX / atlas.width,
        v0: atlas.cursorY / atlas.height,
        u1: (atlas.cursorX + SDF_SIZE) / atlas.width,
        v1: (atlas.cursorY + SDF_SIZE) / atlas.height,
    };

    const atlasX = atlas.cursorX;
    const atlasY = atlas.cursorY;

    atlas.glyphs.set(char, metrics);
    atlas.cursorX += SDF_SIZE;
    atlas.rowHeight = Math.max(atlas.rowHeight, SDF_SIZE);

    return { path, paddedBounds, atlasX, atlasY };
}

export function ensureString(atlas: GlyphAtlas, text: string): void {
    const pending: PendingGlyphEntry[] = [];

    for (const char of text) {
        if (atlas.glyphs.has(char)) continue;
        const entry = computeGlyphMetrics(atlas, char);
        if (entry) pending.push(entry);
    }

    if (pending.length === 0) return;

    atlas.sdfGenerator.begin();
    for (const entry of pending) {
        atlas.sdfGenerator.add(
            entry.path,
            entry.paddedBounds,
            atlas.texture,
            entry.atlasX,
            entry.atlasY,
        );
    }
    atlas.sdfGenerator.flush();
}

export interface LayoutGlyph {
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
}

export interface LayoutResult {
    glyphs: LayoutGlyph[];
    width: number;
    height: number;
}

export function layoutText(text: string, atlas: GlyphAtlas, fontSize: number): LayoutResult {
    const glyphs: LayoutGlyph[] = [];
    const scale = fontSize;

    let cursorX = 0;
    let maxHeight = 0;
    let prevChar: string | null = null;

    for (const char of text) {
        const metrics = atlas.glyphs.get(char);
        if (!metrics) continue;

        if (prevChar) {
            cursorX += (atlas.font.kerning(prevChar, char) / atlas.font.unitsPerEm) * scale;
        }

        const glyphW = metrics.glyphWidth * scale;
        const glyphH = metrics.glyphHeight * scale;
        const advance = metrics.advance * scale;

        const x = cursorX + metrics.glyphLeft * scale;
        const y = (metrics.glyphTop - metrics.glyphHeight) * scale;

        glyphs.push({
            x,
            y,
            width: glyphW,
            height: glyphH,
            texelWidth: metrics.width,
            texelHeight: metrics.height,
            u0: metrics.u0,
            v0: metrics.v0,
            u1: metrics.u1,
            v1: metrics.v1,
        });

        cursorX += advance;
        maxHeight = Math.max(maxHeight, glyphH);
        prevChar = char;
    }

    return {
        glyphs,
        width: cursorX,
        height: maxHeight,
    };
}
