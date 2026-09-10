import { deflateSync } from "node:zlib";
import type { Grid, Palette } from "../../src/standard/loading/mark";

// Minimal PNG writer for pixel-art exports: 8-bit RGBA, no filtering, one IDAT. Enough for a
// few-kilobyte mark; a general encoder would be a dependency for nothing.

const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
    }
    return table;
})();

function crc32(bytes: Uint8Array): number {
    let c = 0xffffffff;
    for (const b of bytes) c = (CRC_TABLE[(c ^ b) & 0xff] as number) ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length);
    out.set(
        [...type].map((ch) => ch.charCodeAt(0)),
        4,
    );
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
    return out;
}

const hexToRgb = (hex: string): [number, number, number] => [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
];

/** Pixel grid → PNG bytes at `scale` pixels per cell. Transparent where the grid is null unless `ground` is set. */
export function toPng(grid: Grid, palette: Palette, scale: number, ground?: string): Uint8Array {
    const height = grid.length * scale;
    const width = (grid[0]?.length ?? 0) * scale;
    const raw = new Uint8Array(height * (1 + width * 4));
    const bg = ground ? hexToRgb(ground) : null;
    for (let y = 0; y < height; y++) {
        const row = grid[Math.floor(y / scale)] ?? [];
        const offset = y * (1 + width * 4);
        raw[offset] = 0;
        for (let x = 0; x < width; x++) {
            const tone = row[Math.floor(x / scale)] ?? null;
            const i = offset + 1 + x * 4;
            const rgb = tone ? hexToRgb(palette[tone]) : bg;
            if (!rgb) continue;
            raw[i] = rgb[0];
            raw[i + 1] = rgb[1];
            raw[i + 2] = rgb[2];
            raw[i + 3] = 255;
        }
    }
    const ihdr = new Uint8Array(13);
    const view = new DataView(ihdr.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    ihdr.set([8, 6, 0, 0, 0], 8);
    const parts = [
        new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
        chunk("IHDR", ihdr),
        chunk("IDAT", new Uint8Array(deflateSync(raw))),
        chunk("IEND", new Uint8Array(0)),
    ];
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
        out.set(p, at);
        at += p.length;
    }
    return out;
}
