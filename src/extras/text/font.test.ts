import { expect, test } from "bun:test";
import { parseFont } from "./font";
import { isolationFont } from "./font.fixture";

test("the generated isolation font has a nonempty rectangle glyph for every letter the GPU sweep renders", () => {
    const bytes = isolationFont();
    const font = parseFont(bytes);
    for (const char of "isolation") {
        expect(font.glyphPath(char)).toContain("L");
        expect(font.glyphBounds(char)).toEqual([0, 0, 600, 800]);
        expect(font.advance(char)).toBe(600);
    }
    let checksum = 0;
    const view = new DataView(bytes);
    for (let i = 0; i < bytes.byteLength; i += 4) checksum = (checksum + view.getUint32(i)) >>> 0;
    expect(checksum).toBe(0xb1b0afba);
});
