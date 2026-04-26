import { test, expect, describe } from "bun:test";
import { normalizeDirection } from "../src/engine/utils/math";
import { packLightUniforms } from "../src/standard/render/light";

describe("light", () => {
    describe("normalizeDirection", () => {
        test("normalizes unit x vector", () => {
            const [x, y, z] = normalizeDirection(1, 0, 0);
            expect(x).toBeCloseTo(1.0);
            expect(y).toBeCloseTo(0.0);
            expect(z).toBeCloseTo(0.0);
        });

        test("normalizes arbitrary vector", () => {
            const [x, y, z] = normalizeDirection(3, 4, 0);
            expect(x).toBeCloseTo(0.6);
            expect(y).toBeCloseTo(0.8);
            expect(z).toBeCloseTo(0.0);
        });

        test("normalizes negative vector", () => {
            const [x, y, z] = normalizeDirection(-1, -1, -1);
            const len = Math.sqrt(3);
            expect(x).toBeCloseTo(-1 / len);
            expect(y).toBeCloseTo(-1 / len);
            expect(z).toBeCloseTo(-1 / len);
        });

        test("handles zero vector with default down", () => {
            const [x, y, z] = normalizeDirection(0, 0, 0);
            expect(x).toBeCloseTo(0.0);
            expect(y).toBeCloseTo(-1.0);
            expect(z).toBeCloseTo(0.0);
        });
    });

    describe("packLightUniforms", () => {
        test("creates Float32Array with correct size", () => {
            const data = packLightUniforms(
                { color: 0xffffff, intensity: 1.0 },
                { color: 0xffffff, intensity: 1.0, directionX: 0, directionY: -1, directionZ: 0 },
            );
            expect(data).toBeInstanceOf(Float32Array);
            expect(data.length).toBe(12);
        });

        test("packs ambient color with intensity in alpha", () => {
            const data = packLightUniforms(
                { color: 0xff0000, intensity: 0.5 },
                { color: 0xffffff, intensity: 1.0, directionX: 0, directionY: -1, directionZ: 0 },
            );
            expect(data[0]).toBeCloseTo(1.0);
            expect(data[1]).toBeCloseTo(0.0);
            expect(data[2]).toBeCloseTo(0.0);
            expect(data[3]).toBeCloseTo(0.5);
        });

        test("packs normalized sun direction", () => {
            const data = packLightUniforms(
                { color: 0xffffff, intensity: 1.0 },
                { color: 0xffffff, intensity: 1.0, directionX: 1, directionY: 1, directionZ: 0 },
            );
            const len = Math.sqrt(2);
            expect(data[4]).toBeCloseTo(1 / len);
            expect(data[5]).toBeCloseTo(1 / len);
            expect(data[6]).toBeCloseTo(0.0);
            expect(data[7]).toBeCloseTo(0.0);
        });

        test("packs sun color multiplied by intensity", () => {
            const data = packLightUniforms(
                { color: 0xffffff, intensity: 1.0 },
                { color: 0xff8000, intensity: 2.0, directionX: 0, directionY: -1, directionZ: 0 },
            );
            expect(data[8]).toBeCloseTo(2.0);
            expect(data[9]).toBeCloseTo(0.2159 * 2, 2);
            expect(data[10]).toBeCloseTo(0.0);
            expect(data[11]).toBeCloseTo(0.0);
        });
    });
});
