import type { Font } from "./font";
import { Justify } from "./justify";
import { SDF_EXPONENT, SDFGenerator } from "./sdf";

const SDF_SIZE = 96;

export interface GlyphMetrics {
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
    width: number;
    height: number;
    glyphs: Map<string, GlyphMetrics>;
    rowHeight: number;
    cursorX: number;
    cursorY: number;
    font: Font;
    sdfGenerator: SDFGenerator;
}

/** Releases both the atlas texture and its generator's intermediate texture. @internal */
export function disposeAtlases(atlases: readonly (GlyphAtlas | null | undefined)[]): void {
    for (const atlas of atlases) {
        atlas?.sdfGenerator.destroy();
        atlas?.texture.destroy();
    }
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

/** Records UVs and em-normalized metrics in atlas.glyphs; returns a request for batched rendering.
 * Returns null without an outline or bounds; throws when the atlas is full. */
function computeGlyphMetrics(atlas: GlyphAtlas, char: string): PendingGlyphEntry | null {
    const path = atlas.font.glyphPath(char);
    const bounds = atlas.font.glyphBounds(char);
    const advance = atlas.font.advance(char);

    const unitsPerEm = atlas.font.unitsPerEm;
    if (!path || !bounds) {
        atlas.glyphs.set(char, {
            glyphWidth: 0,
            glyphHeight: 0,
            glyphTop: 0,
            glyphLeft: 0,
            advance: advance / unitsPerEm,
            u0: 0,
            v0: 0,
            u1: 0,
            v1: 0,
        });
        return null;
    }

    const [xMin, yMin, xMax, yMax] = bounds;

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

export function ensureString(
    world: import("../../engine").World,
    atlas: GlyphAtlas,
    text: string,
): void {
    const pending: PendingGlyphEntry[] = [];

    for (const char of text) {
        if (atlas.glyphs.has(char)) continue;
        const entry = computeGlyphMetrics(atlas, char);
        if (entry) pending.push(entry);
    }

    if (pending.length === 0) return;

    atlas.sdfGenerator.begin(world);
    for (const entry of pending) {
        atlas.sdfGenerator.add(
            entry.path,
            entry.paddedBounds,
            atlas.texture,
            entry.atlasX,
            entry.atlasY,
        );
    }
    atlas.sdfGenerator.flush(world);
}

export interface LayoutGlyph {
    x: number;
    y: number;
    width: number;
    height: number;
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

export function layoutText(
    text: string,
    atlas: GlyphAtlas,
    fontSize: number,
    maxWidth = Infinity,
    justify: number = Justify.Left,
): LayoutResult {
    if (!Number.isFinite(maxWidth) && !text.includes("\n"))
        return layoutLine(text, atlas, fontSize);
    const lines: LayoutResult[] = [];
    for (const paragraph of text.split("\n")) {
        let line = "";
        for (const part of paragraph.match(/[^\S\n]*\S+|[^\S\n]+$/gu) ?? []) {
            const candidate = line + part;
            if (line && layoutLine(candidate, atlas, fontSize).width > maxWidth) {
                lines.push(layoutLine(line, atlas, fontSize));
                line = part.trimStart();
            } else line = candidate;
        }
        lines.push(layoutLine(line, atlas, fontSize));
    }
    const width = Math.max(
        Number.isFinite(maxWidth) ? maxWidth : 0,
        ...lines.map((line) => line.width),
    );
    const advance = (atlas.font.lineHeight / atlas.font.unitsPerEm) * fontSize;
    const glyphs: LayoutGlyph[] = [];
    for (const [i, line] of lines.entries()) {
        const x =
            (width - line.width) *
            (justify === Justify.Center ? 0.5 : justify === Justify.Right ? 1 : 0);
        const y = (lines.length - 1 - i) * advance;
        for (const glyph of line.glyphs) glyphs.push({ ...glyph, x: glyph.x + x, y: glyph.y + y });
    }
    return {
        glyphs,
        width,
        height: Math.max(...lines.map((line) => line.height)) + (lines.length - 1) * advance,
    };
}

function layoutLine(text: string, atlas: GlyphAtlas, fontSize: number): LayoutResult {
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

        if (glyphW > 0 && glyphH > 0) {
            glyphs.push({
                x,
                y,
                width: glyphW,
                height: glyphH,
                u0: metrics.u0,
                v0: metrics.v0,
                u1: metrics.u1,
                v1: metrics.v1,
            });
        }

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
