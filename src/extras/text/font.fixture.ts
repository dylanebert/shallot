/** A two-glyph SFNT: empty .notdef and one rectangle, mapped to the letters in "isolation".
 * Generated test data keeps GPU ownership checks independent of network/font-download latency. */
export function isolationFont(): ArrayBuffer {
    const tables = [
        ["head", 54],
        ["hhea", 36],
        ["maxp", 32],
        ["hmtx", 8],
        ["loca", 6],
        ["glyf", 34],
        ["cmap", 112],
    ] as const;
    const offsets = new Map<string, number>();
    let size = 12 + tables.length * 16;
    for (const [name, length] of tables) {
        offsets.set(name, size);
        size += (length + 3) & ~3;
    }
    const buffer = new ArrayBuffer(size);
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    view.setUint32(0, 0x00010000);
    view.setUint16(4, tables.length);
    view.setUint16(6, 64);
    view.setUint16(8, 2);
    view.setUint16(10, 48);
    for (let i = 0; i < tables.length; i++) {
        const [name, length] = tables[i];
        const at = 12 + i * 16;
        for (let j = 0; j < 4; j++) bytes[at + j] = name.charCodeAt(j);
        view.setUint32(at + 8, offsets.get(name)!);
        view.setUint32(at + 12, length);
    }
    const head = offsets.get("head")!;
    view.setUint32(head, 0x00010000);
    view.setUint32(head + 12, 0x5f0f3cf5);
    view.setUint16(head + 18, 1000);
    const hhea = offsets.get("hhea")!;
    view.setUint32(hhea, 0x00010000);
    view.setInt16(hhea + 4, 800);
    view.setInt16(hhea + 6, -200);
    view.setUint16(hhea + 34, 2);
    const maxp = offsets.get("maxp")!;
    view.setUint32(maxp, 0x00010000);
    view.setUint16(maxp + 4, 2);
    view.setUint16(maxp + 6, 4);
    view.setUint16(maxp + 8, 1);
    const hmtx = offsets.get("hmtx")!;
    view.setUint16(hmtx, 600);
    view.setUint16(hmtx + 4, 600);
    view.setUint16(offsets.get("loca")! + 4, 17);
    const glyph = offsets.get("glyf")!;
    view.setInt16(glyph, 1);
    view.setInt16(glyph + 6, 600);
    view.setInt16(glyph + 8, 800);
    view.setUint16(glyph + 10, 3);
    bytes.fill(1, glyph + 14, glyph + 18);
    for (const [i, delta] of [0, 600, 0, -600, 0, 0, 800, 0].entries())
        view.setInt16(glyph + 18 + i * 2, delta);
    const cmap = offsets.get("cmap")!;
    view.setUint16(cmap + 2, 1);
    view.setUint16(cmap + 4, 3);
    view.setUint16(cmap + 6, 10);
    view.setUint32(cmap + 8, 12);
    const sub = cmap + 12;
    view.setUint16(sub, 12);
    view.setUint32(sub + 4, 100);
    view.setUint32(sub + 12, 7);
    for (const [i, char] of [...new Set("isolation")].sort().entries()) {
        const at = sub + 16 + i * 12;
        view.setUint32(at, char.charCodeAt(0));
        view.setUint32(at + 4, char.charCodeAt(0));
        view.setUint32(at + 8, 1);
    }
    // Table checksums and the head adjustment make the fixture a complete SFNT checksum as well.
    for (let i = 0; i < tables.length; i++) {
        const [name, length] = tables[i];
        let sum = 0;
        for (let j = 0; j < length; j += 4)
            sum = (sum + view.getUint32(offsets.get(name)! + j)) >>> 0;
        view.setUint32(12 + i * 16 + 4, sum);
    }
    let sum = 0;
    for (let j = 0; j < size; j += 4) sum = (sum + view.getUint32(j)) >>> 0;
    view.setUint32(head + 8, (0xb1b0afba - sum) >>> 0);
    return buffer;
}
