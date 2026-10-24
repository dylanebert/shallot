import { expect, test } from "bun:test";
import { aoDifference } from "./compare-ao";

test("AO difference reports every pixel, including unchanged pixels, in visibility units", () => {
    expect(aoDifference(new Float32Array([0, 1, 1, 1]), new Float32Array([1, 1, 1, 1]))).toEqual({
        pixels: 4,
        max: 1,
        mean: 0.25,
    });
});

test("incomplete and non-finite AO captures fail instead of reporting a look difference", () => {
    expect(() => aoDifference(new Float32Array(), new Float32Array())).toThrow();
    expect(() => aoDifference(new Float32Array([1]), new Float32Array([1, 1]))).toThrow();
    expect(() => aoDifference(new Float32Array([NaN]), new Float32Array([1]))).toThrow();
});
