import { expect, test } from "bun:test";
import { admitsAlpha, certifyFill, packingInputs, packingWords } from "./cells-oracle";

test("the finite rational certificate proves all 32 RGB constructions and ten minimal alpha sets", () => {
    const certificate = certifyFill(10, 6);
    expect(certificate.population).toEqual({ cells: 60, alpha: 120, rgb: 360 });
    expect(certificate.rgb).toHaveLength(32);
    for (const [denominator, complement, expected] of [
        [10, false, [0, 89, 124, 149, 170, 188, 203, 218, 231, 243]],
        [10, true, [255, 243, 231, 218, 203, 188, 170, 149, 124, 89]],
        [6, false, [0, 113, 156, 188, 213, 235]],
        [6, true, [255, 235, 213, 188, 156, 113]],
    ] as const) {
        expect(
            certificate.rgb
                .filter((r) => r.denominator === denominator && r.complement === complement)
                .map((r) => r.byte),
        ).toEqual([...expected]);
    }
    expect(certificate.alpha).toHaveLength(10);
    for (let i = 0; i < 10; i++) {
        const set = certificate.alpha[i];
        expect(set).toEqual([254 - i, 255 - i]);
        for (let byte = 0; byte < 256; byte++) {
            expect(admitsAlpha(set, byte)).toBe(byte === 254 - i || byte === 255 - i);
        }
        for (const invalid of [NaN, Infinity, -Infinity, set[0] - 1, set[1] + 1, set[0] + 0.5]) {
            expect(admitsAlpha(set, invalid)).toBe(false);
        }
    }
});

test("changed dimensions refuse rather than reuse a finite expression certificate", () => {
    for (const [cols, rows] of [
        [0, 0],
        [9, 6],
        [10, 5],
        [6, 10],
        [11, 6],
        [10, 7],
        [NaN, 6],
        [10, Infinity],
    ]) {
        expect(() => certifyFill(cols, rows)).toThrow("only the 10×6");
    }
});

test("exact dyadic GPU inputs include independent fg/bg routes and below/at/above controls", () => {
    expect(packingInputs).toEqual([
        [0.25, 0.75],
        [0, 1],
        [511 / 1024, 513 / 1024],
        [0.5, 0.25],
    ]);
    const words = packingWords();
    expect(Array.from(words)).toEqual([
        17, 0x4000ff00, 0xbfff00ff, 17, 0x0000ff00, 0xffff00ff, 17, 0x7f00ff00, 0x80ff00ff, 17,
        0x8000ff00, 0x40ff00ff,
    ]);
    // Exact controls cannot absorb the shared packer's +/- one-byte alpha bias.
    for (const index of [1, 2]) {
        expect((words[index] + 0x01000000) >>> 0).not.toBe(words[index]);
        expect((words[index] - 0x01000000) >>> 0).not.toBe(words[index]);
    }
});
