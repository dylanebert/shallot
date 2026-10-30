import { expect, test } from "bun:test";
import { decodedArray } from "./arrays";
import { recordArray } from "./records";

test("authored signed zero remains exact in small values and large byte digests", () => {
    decodedArray(new Float32Array([-0]), [-0], 1, "Model primitive 0 positions");
    expect(() =>
        decodedArray(new Float32Array([-0]), [0], 1, "Model primitive 0 positions"),
    ).toThrow();
    const positions = Float32Array.from({ length: 135 }, (_, i) => (i < 3 ? -1 : i < 6 ? 1 : 0));
    const expected = recordArray(positions, 3);
    new Uint32Array(positions.buffer)[65] = 0x80000000;
    expect(() => decodedArray(positions, expected, 3, "Model primitive 0 positions")).toThrow();
});

test("a one-ulp interior position change fails the digest without changing counts or bounds", () => {
    const positions = Float32Array.from({ length: 135 }, (_, i) => (i < 3 ? 0 : i < 6 ? 1 : 0.5));
    const expected = recordArray(positions, 3);
    new Uint32Array(positions.buffer)[65]++;
    const actual = recordArray(positions, 3);
    expect(Array.isArray(expected)).toBe(false);
    expect(Array.isArray(actual)).toBe(false);
    if (Array.isArray(expected) || Array.isArray(actual))
        throw new Error("large position fixture must use a digest");
    expect(actual.count).toBe(expected.count);
    expect(actual.min).toEqual(expected.min);
    expect(actual.max).toEqual(expected.max);
    expect(actual.sha256).not.toBe(expected.sha256);
    expect(() => decodedArray(positions, expected, 3, "Model primitive 0 positions")).toThrow(
        "Model primitive 0 positions",
    );
});
