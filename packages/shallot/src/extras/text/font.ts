export interface Font {
    unitsPerEm: number;
    ascender: number;
    descender: number;
    lineGap: number;
    glyphPath(char: string): string | null;
    glyphBounds(char: string): [number, number, number, number] | null;
    advance(char: string): number;
    kerning(left: string, right: string): number;
}

interface TableEntry {
    offset: number;
    length: number;
}

interface Reader {
    data: DataView;
    offset: number;
}

function u8(r: Reader): number {
    return r.data.getUint8(r.offset++);
}

function i16(r: Reader): number {
    const v = r.data.getInt16(r.offset);
    r.offset += 2;
    return v;
}

function u16(r: Reader): number {
    const v = r.data.getUint16(r.offset);
    r.offset += 2;
    return v;
}

function u32(r: Reader): number {
    const v = r.data.getUint32(r.offset);
    r.offset += 4;
    return v;
}

function tag(r: Reader): string {
    return String.fromCharCode(u8(r), u8(r), u8(r), u8(r));
}

function seek(r: Reader, offset: number): void {
    r.offset = offset;
}

function parseTables(r: Reader): Map<string, TableEntry> {
    const tables = new Map<string, TableEntry>();
    seek(r, 0);

    const sfntVersion = tag(r);
    if (sfntVersion !== "\x00\x01\x00\x00" && sfntVersion !== "OTTO" && sfntVersion !== "true") {
        throw new Error("Not a valid TTF/OTF font");
    }

    const numTables = u16(r);
    u16(r);
    u16(r);
    u16(r);

    for (let i = 0; i < numTables; i++) {
        const name = tag(r);
        u32(r);
        const offset = u32(r);
        const length = u32(r);
        tables.set(name, { offset, length });
    }

    return tables;
}

function parseHead(r: Reader, table: TableEntry): { unitsPerEm: number; indexToLocFormat: number } {
    seek(r, table.offset + 18);
    const unitsPerEm = u16(r);
    seek(r, table.offset + 50);
    const indexToLocFormat = i16(r);
    return { unitsPerEm, indexToLocFormat };
}

function parseHhea(
    r: Reader,
    table: TableEntry,
): { ascender: number; descender: number; lineGap: number; numHMetrics: number } {
    seek(r, table.offset + 4);
    const ascender = i16(r);
    const descender = i16(r);
    const lineGap = i16(r);
    seek(r, table.offset + 34);
    const numHMetrics = u16(r);
    return { ascender, descender, lineGap, numHMetrics };
}

function parseHmtx(
    r: Reader,
    table: TableEntry,
    numHMetrics: number,
    numGlyphs: number,
): { advances: Uint16Array } {
    const advances = new Uint16Array(numGlyphs);
    seek(r, table.offset);

    let lastAdvance = 0;
    for (let i = 0; i < numHMetrics; i++) {
        lastAdvance = u16(r);
        advances[i] = lastAdvance;
        i16(r);
    }
    for (let i = numHMetrics; i < numGlyphs; i++) {
        advances[i] = lastAdvance;
    }

    return { advances };
}

function parseMaxp(r: Reader, table: TableEntry): number {
    seek(r, table.offset + 4);
    return u16(r);
}

function parseLoca(
    r: Reader,
    table: TableEntry,
    numGlyphs: number,
    indexToLocFormat: number,
): Uint32Array {
    const offsets = new Uint32Array(numGlyphs + 1);
    seek(r, table.offset);

    if (indexToLocFormat === 0) {
        for (let i = 0; i <= numGlyphs; i++) {
            offsets[i] = u16(r) * 2;
        }
    } else {
        for (let i = 0; i <= numGlyphs; i++) {
            offsets[i] = u32(r);
        }
    }

    return offsets;
}

function parseCmap(r: Reader, table: TableEntry): Map<number, number> {
    const charToGlyph = new Map<number, number>();
    seek(r, table.offset);

    u16(r);
    const numSubtables = u16(r);

    let format4Offset = -1;
    let format12Offset = -1;

    for (let i = 0; i < numSubtables; i++) {
        const platformId = u16(r);
        const encodingId = u16(r);
        const offset = u32(r);

        if (platformId === 3 && encodingId === 1) format4Offset = table.offset + offset;
        if (platformId === 3 && encodingId === 10) format12Offset = table.offset + offset;
        if (platformId === 0 && encodingId === 3) format4Offset = table.offset + offset;
        if (platformId === 0 && encodingId === 4) format12Offset = table.offset + offset;
    }

    if (format12Offset !== -1) {
        seek(r, format12Offset);
        const format = u16(r);
        if (format === 12) {
            u16(r);
            u32(r);
            u32(r);
            const numGroups = u32(r);
            for (let i = 0; i < numGroups; i++) {
                const startCode = u32(r);
                const endCode = u32(r);
                const startGlyph = u32(r);
                for (let c = startCode; c <= endCode; c++) {
                    charToGlyph.set(c, startGlyph + (c - startCode));
                }
            }
            return charToGlyph;
        }
    }

    if (format4Offset !== -1) {
        seek(r, format4Offset);
        const format = u16(r);
        if (format === 4) {
            u16(r);
            u16(r);
            const segCount = u16(r) / 2;
            u16(r);
            u16(r);
            u16(r);

            const endCodes: number[] = [];
            for (let i = 0; i < segCount; i++) endCodes.push(u16(r));
            u16(r);

            const startCodes: number[] = [];
            for (let i = 0; i < segCount; i++) startCodes.push(u16(r));

            const idDeltas: number[] = [];
            for (let i = 0; i < segCount; i++) idDeltas.push(i16(r));

            const idRangeOffsetPos = r.offset;
            const idRangeOffsets: number[] = [];
            for (let i = 0; i < segCount; i++) idRangeOffsets.push(u16(r));

            for (let i = 0; i < segCount; i++) {
                const start = startCodes[i];
                const end = endCodes[i];
                const delta = idDeltas[i];
                const rangeOffset = idRangeOffsets[i];

                if (end === 0xffff) continue;

                for (let c = start; c <= end; c++) {
                    let glyphId: number;
                    if (rangeOffset === 0) {
                        glyphId = (c + delta) & 0xffff;
                    } else {
                        const glyphIdOffset =
                            idRangeOffsetPos + i * 2 + rangeOffset + (c - start) * 2;
                        seek(r, glyphIdOffset);
                        glyphId = u16(r);
                        if (glyphId !== 0) {
                            glyphId = (glyphId + delta) & 0xffff;
                        }
                    }
                    if (glyphId !== 0) {
                        charToGlyph.set(c, glyphId);
                    }
                }
            }
        }
    }

    return charToGlyph;
}

function parseKern(r: Reader, table: TableEntry): Map<number, number> {
    const kerning = new Map<number, number>();
    seek(r, table.offset);

    const version = u16(r);
    if (version === 0) {
        const numSubtables = u16(r);
        for (let t = 0; t < numSubtables; t++) {
            u16(r);
            u16(r);
            const coverage = u16(r);
            const format = coverage >> 8;

            if (format === 0) {
                const numPairs = u16(r);
                u16(r);
                u16(r);
                u16(r);

                for (let i = 0; i < numPairs; i++) {
                    const left = u16(r);
                    const right = u16(r);
                    const value = i16(r);
                    kerning.set((left << 16) | right, value);
                }
            }
        }
    } else if (version === 1) {
        u16(r);
        const numSubtables = u32(r);
        for (let t = 0; t < numSubtables; t++) {
            const subtableLength = u32(r);
            const coverage = u16(r);
            const format = coverage & 0xff;

            if (format === 0) {
                const numPairs = u16(r);
                u16(r);
                u16(r);
                u16(r);

                for (let i = 0; i < numPairs; i++) {
                    const left = u16(r);
                    const right = u16(r);
                    const value = i16(r);
                    kerning.set((left << 16) | right, value);
                }
            } else {
                seek(r, r.offset + subtableLength - 8);
            }
        }
    }

    return kerning;
}

const ON_CURVE = 1;
const X_SHORT = 2;
const Y_SHORT = 4;
const REPEAT = 8;
const X_SAME = 16;
const Y_SAME = 32;

function parseGlyph(
    r: Reader,
    glyfOffset: number,
    loca: Uint32Array,
    glyphId: number,
): { path: string; bounds: [number, number, number, number] } | null {
    const start = loca[glyphId];
    const end = loca[glyphId + 1];
    if (start === end) return null;

    seek(r, glyfOffset + start);
    const numContours = i16(r);
    const xMin = i16(r);
    const yMin = i16(r);
    const xMax = i16(r);
    const yMax = i16(r);

    if (numContours < 0) {
        return parseCompositeGlyph(r, glyfOffset, loca);
    }

    const endPts: number[] = [];
    for (let i = 0; i < numContours; i++) {
        endPts.push(u16(r));
    }

    const numPoints = endPts.length > 0 ? endPts[endPts.length - 1] + 1 : 0;
    const instructionLength = u16(r);
    seek(r, r.offset + instructionLength);

    const flags: number[] = [];
    while (flags.length < numPoints) {
        const flag = u8(r);
        flags.push(flag);
        if (flag & REPEAT) {
            const repeat = u8(r);
            for (let i = 0; i < repeat; i++) flags.push(flag);
        }
    }

    const xs: number[] = [];
    let x = 0;
    for (let i = 0; i < numPoints; i++) {
        const flag = flags[i];
        if (flag & X_SHORT) {
            const dx = u8(r);
            x += flag & X_SAME ? dx : -dx;
        } else if (!(flag & X_SAME)) {
            x += i16(r);
        }
        xs.push(x);
    }

    const ys: number[] = [];
    let y = 0;
    for (let i = 0; i < numPoints; i++) {
        const flag = flags[i];
        if (flag & Y_SHORT) {
            const dy = u8(r);
            y += flag & Y_SAME ? dy : -dy;
        } else if (!(flag & Y_SAME)) {
            y += i16(r);
        }
        ys.push(y);
    }

    let path = "";
    let contourStart = 0;

    for (let c = 0; c < numContours; c++) {
        const contourEnd = endPts[c];
        const points: { x: number; y: number; on: boolean }[] = [];

        for (let i = contourStart; i <= contourEnd; i++) {
            points.push({ x: xs[i], y: ys[i], on: !!(flags[i] & ON_CURVE) });
        }

        if (points.length === 0) {
            contourStart = contourEnd + 1;
            continue;
        }

        let firstOn = 0;
        while (firstOn < points.length && !points[firstOn].on) firstOn++;

        if (firstOn === points.length) {
            const mid = {
                x: (points[0].x + points[1].x) / 2,
                y: (points[0].y + points[1].y) / 2,
                on: true,
            };
            points.unshift(mid);
            firstOn = 0;
        }

        const reordered = [...points.slice(firstOn), ...points.slice(0, firstOn)];
        path += `M${reordered[0].x},${reordered[0].y}`;

        let i = 1;
        while (i < reordered.length) {
            const p = reordered[i];
            if (p.on) {
                path += `L${p.x},${p.y}`;
                i++;
            } else {
                const next = reordered[(i + 1) % reordered.length];
                if (next.on) {
                    path += `Q${p.x},${p.y},${next.x},${next.y}`;
                    i += 2;
                } else {
                    const midX = (p.x + next.x) / 2;
                    const midY = (p.y + next.y) / 2;
                    path += `Q${p.x},${p.y},${midX},${midY}`;
                    i++;
                }
            }
        }

        if (!reordered[reordered.length - 1].on) {
            const last = reordered[reordered.length - 1];
            path += `Q${last.x},${last.y},${reordered[0].x},${reordered[0].y}`;
        }

        path += "Z";
        contourStart = contourEnd + 1;
    }

    return { path, bounds: [xMin, yMin, xMax, yMax] };
}

function parseCompositeGlyph(
    r: Reader,
    glyfOffset: number,
    loca: Uint32Array,
): { path: string; bounds: [number, number, number, number] } | null {
    let path = "";
    let xMin = Infinity,
        yMin = Infinity,
        xMax = -Infinity,
        yMax = -Infinity;
    let hasMore = true;

    while (hasMore) {
        const flags = u16(r);
        const glyphIndex = u16(r);

        let dx = 0,
            dy = 0;
        let a = 1,
            b = 0,
            c = 0,
            d = 1;

        if (flags & 1) {
            dx = i16(r);
            dy = i16(r);
        } else {
            dx = (r.data.getInt8(r.offset++) + r.data.getInt8(r.offset++)) / 2;
            dy = 0;
        }

        if (flags & 8) {
            a = d = i16(r) / 16384;
        } else if (flags & 64) {
            a = i16(r) / 16384;
            d = i16(r) / 16384;
        } else if (flags & 128) {
            a = i16(r) / 16384;
            b = i16(r) / 16384;
            c = i16(r) / 16384;
            d = i16(r) / 16384;
        }

        const savedOffset = r.offset;
        const component = parseGlyph(r, glyfOffset, loca, glyphIndex);
        r.offset = savedOffset;

        if (component) {
            const transformed = transformPath(component.path, a, b, c, d, dx, dy);
            path += transformed;
            xMin = Math.min(xMin, component.bounds[0] * a + component.bounds[1] * b + dx);
            yMin = Math.min(yMin, component.bounds[0] * c + component.bounds[1] * d + dy);
            xMax = Math.max(xMax, component.bounds[2] * a + component.bounds[3] * b + dx);
            yMax = Math.max(yMax, component.bounds[2] * c + component.bounds[3] * d + dy);
        }

        hasMore = !!(flags & 32);
    }

    if (path === "") return null;
    return { path, bounds: [xMin, yMin, xMax, yMax] };
}

function transformPath(
    path: string,
    a: number,
    b: number,
    c: number,
    d: number,
    dx: number,
    dy: number,
): string {
    return path.replace(/(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/g, (_, x, y) => {
        const nx = parseFloat(x) * a + parseFloat(y) * b + dx;
        const ny = parseFloat(x) * c + parseFloat(y) * d + dy;
        return `${nx},${ny}`;
    });
}

export function parseFont(buffer: ArrayBuffer): Font {
    const r: Reader = { data: new DataView(buffer), offset: 0 };
    const tables = parseTables(r);

    const headTable = tables.get("head");
    const hheaTable = tables.get("hhea");
    const hmtxTable = tables.get("hmtx");
    const maxpTable = tables.get("maxp");
    const cmapTable = tables.get("cmap");
    const locaTable = tables.get("loca");
    const glyfTable = tables.get("glyf");
    const kernTable = tables.get("kern");

    if (
        !headTable ||
        !hheaTable ||
        !hmtxTable ||
        !maxpTable ||
        !cmapTable ||
        !locaTable ||
        !glyfTable
    ) {
        throw new Error("Missing required font tables");
    }

    const head = parseHead(r, headTable);
    const hhea = parseHhea(r, hheaTable);
    const numGlyphs = parseMaxp(r, maxpTable);
    const hmtx = parseHmtx(r, hmtxTable, hhea.numHMetrics, numGlyphs);
    const loca = parseLoca(r, locaTable, numGlyphs, head.indexToLocFormat);
    const cmap = parseCmap(r, cmapTable);
    const kern = kernTable ? parseKern(r, kernTable) : new Map<number, number>();

    const glyphCache = new Map<
        number,
        { path: string; bounds: [number, number, number, number] } | null
    >();
    const glyfOffset = glyfTable.offset;

    function getGlyphId(char: string): number {
        return cmap.get(char.codePointAt(0) ?? 0) ?? 0;
    }

    function getGlyph(
        glyphId: number,
    ): { path: string; bounds: [number, number, number, number] } | null {
        if (glyphCache.has(glyphId)) return glyphCache.get(glyphId)!;
        const glyph = parseGlyph(r, glyfOffset, loca, glyphId);
        glyphCache.set(glyphId, glyph);
        return glyph;
    }

    return {
        unitsPerEm: head.unitsPerEm,
        ascender: hhea.ascender,
        descender: hhea.descender,
        lineGap: hhea.lineGap,

        glyphPath(char: string): string | null {
            const glyph = getGlyph(getGlyphId(char));
            return glyph?.path ?? null;
        },

        glyphBounds(char: string): [number, number, number, number] | null {
            const glyph = getGlyph(getGlyphId(char));
            return glyph?.bounds ?? null;
        },

        advance(char: string): number {
            return hmtx.advances[getGlyphId(char)] ?? 0;
        },

        kerning(left: string, right: string): number {
            const l = getGlyphId(left);
            const r = getGlyphId(right);
            return kern.get((l << 16) | r) ?? 0;
        },
    };
}

export async function loadFont(url: string): Promise<Font> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load font: ${response.statusText}`);
    return parseFont(await response.arrayBuffer());
}
