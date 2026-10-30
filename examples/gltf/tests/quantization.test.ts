import { expect, test } from "bun:test";
import { decodeQuantized, float32Ulp } from "./quantization";

test("the decode bound includes division rounding amplified by the extent", () => {
    const offset = Math.fround(-0.014622720889747143);
    const extent = Math.fround(3.7);
    // A float32 source just inside code 259's lower cell boundary; offset cancels
    // the product, making division's rounding larger than addition's rounding.
    const expected = Math.fround(-0.000028229208564476796 + 2e-12);
    const code = Math.round(((expected - offset) / extent) * 65535);
    expect(code).toBe(259);
    const { value, bound } = decodeQuantized(code, offset, extent);
    const error = Math.abs(value - expected);
    const product = Math.fround(Math.fround(code / 65535) * extent);
    const withoutDivision = extent / 131070 + float32Ulp(product) / 2 + float32Ulp(value) / 2;
    expect(error).toBeGreaterThan(withoutDivision);
    expect(error).toBeLessThanOrEqual(bound);
});

test("one extra x-axis code exceeds the derived float32 decode bound", () => {
    const expected = -0.5;
    const correct = decodeQuantized(0, -0.5, 1);
    expect(Math.abs(correct.value - expected)).toBeLessThanOrEqual(correct.bound);
    const perturbed = decodeQuantized(1, -0.5, 1);
    expect(Math.abs(perturbed.value - expected)).toBeGreaterThan(perturbed.bound);
});
