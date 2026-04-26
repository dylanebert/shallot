import { describe, test, expect } from "bun:test";
import { invert, multiply, orthographic, perspective, unpackColor } from "../src/engine/utils/math";

describe("Camera", () => {
    describe("unpackColor", () => {
        test("unpacks black (0x000000)", () => {
            const color = unpackColor(0x000000);
            expect(color.r).toBe(0);
            expect(color.g).toBe(0);
            expect(color.b).toBe(0);
        });

        test("unpacks white (0xffffff)", () => {
            const color = unpackColor(0xffffff);
            expect(color.r).toBe(1);
            expect(color.g).toBe(1);
            expect(color.b).toBe(1);
        });

        test("unpacks red (0xff0000)", () => {
            const color = unpackColor(0xff0000);
            expect(color.r).toBe(1);
            expect(color.g).toBe(0);
            expect(color.b).toBe(0);
        });

        test("unpacks green (0x00ff00)", () => {
            const color = unpackColor(0x00ff00);
            expect(color.r).toBe(0);
            expect(color.g).toBe(1);
            expect(color.b).toBe(0);
        });

        test("unpacks blue (0x0000ff)", () => {
            const color = unpackColor(0x0000ff);
            expect(color.r).toBe(0);
            expect(color.g).toBe(0);
            expect(color.b).toBe(1);
        });

        test("unpacks mid-gray (0x808080) as linear", () => {
            const color = unpackColor(0x808080);
            expect(color.r).toBeCloseTo(0.2159, 3);
            expect(color.g).toBeCloseTo(0.2159, 3);
            expect(color.b).toBeCloseTo(0.2159, 3);
        });
    });

    describe("invert", () => {
        test("inverts identity matrix to identity", () => {
            const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            const result = invert(identity);
            expect(result[0]).toBeCloseTo(1);
            expect(result[5]).toBeCloseTo(1);
            expect(result[10]).toBeCloseTo(1);
            expect(result[15]).toBeCloseTo(1);
        });

        test("inverts translation", () => {
            const translated = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 3, 2, 1]);
            const result = invert(translated);
            expect(result[12]).toBeCloseTo(-5);
            expect(result[13]).toBeCloseTo(-3);
            expect(result[14]).toBeCloseTo(-2);
        });
    });

    describe("multiply", () => {
        test("identity × identity = identity", () => {
            const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            const result = multiply(identity, identity);
            expect(result[0]).toBeCloseTo(1);
            expect(result[5]).toBeCloseTo(1);
            expect(result[10]).toBeCloseTo(1);
            expect(result[15]).toBeCloseTo(1);
        });

        test("identity × A = A", () => {
            const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
            const a = new Float32Array([2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 1, 2, 3, 1]);
            const result = multiply(identity, a);
            for (let i = 0; i < 16; i++) {
                expect(result[i]).toBeCloseTo(a[i]);
            }
        });
    });

    describe("orthographic", () => {
        test("produces correct matrix for square aspect", () => {
            const result = orthographic(5, 1, 0.1, 100);
            expect(result[0]).toBeCloseTo(0.2);
            expect(result[5]).toBeCloseTo(0.2);
            expect(result[10]).toBeCloseTo(-0.01001);
            expect(result[14]).toBeCloseTo(-0.001001);
            expect(result[15]).toBeCloseTo(1);
        });

        test("scales x by aspect ratio", () => {
            const result = orthographic(5, 2, 0.1, 100);
            expect(result[0]).toBeCloseTo(0.1);
            expect(result[5]).toBeCloseTo(0.2);
        });

        test("w component is 1 (no perspective divide)", () => {
            const result = orthographic(10, 1.5, 1, 1000);
            expect(result[3]).toBe(0);
            expect(result[7]).toBe(0);
            expect(result[11]).toBe(0);
            expect(result[15]).toBe(1);
        });

        test("maps z correctly for WebGPU clip space [0, 1]", () => {
            const near = 0.1;
            const far = 100;
            const result = orthographic(5, 1, near, far);
            const nf = 1 / (near - far);
            expect(result[10]).toBeCloseTo(nf);
            expect(result[14]).toBeCloseTo(near * nf);
        });

        test("throws on size = 0", () => {
            expect(() => orthographic(0, 1, 0.1, 100)).toThrow("Invalid orthographic size");
        });

        test("throws on negative size", () => {
            expect(() => orthographic(-5, 1, 0.1, 100)).toThrow("Invalid orthographic size");
        });

        test("throws on aspect = 0", () => {
            expect(() => orthographic(5, 0, 0.1, 100)).toThrow("Invalid aspect ratio");
        });

        test("throws on near === far", () => {
            expect(() => orthographic(5, 1, 100, 100)).toThrow("Invalid depth planes");
        });
    });

    describe("perspective", () => {
        test("produces finite values for valid inputs", () => {
            const result = perspective(60, 16 / 9, 0.1, 1000);
            for (let i = 0; i < 16; i++) {
                expect(Number.isFinite(result[i])).toBe(true);
            }
        });

        test("throws on FOV = 0", () => {
            expect(() => perspective(0, 1, 0.1, 100)).toThrow("Invalid FOV");
        });

        test("throws on negative FOV", () => {
            expect(() => perspective(-60, 1, 0.1, 100)).toThrow("Invalid FOV");
        });

        test("throws on aspect = 0", () => {
            expect(() => perspective(60, 0, 0.1, 100)).toThrow("Invalid aspect ratio");
        });

        test("throws on near === far", () => {
            expect(() => perspective(60, 1, 100, 100)).toThrow("Invalid depth planes");
        });
    });
});
