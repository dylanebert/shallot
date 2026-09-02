import { describe, expect, test } from "bun:test";
import { CAPTURE, sunDirection } from "../src/conditions";

describe("ocean showcase capture", () => {
    test("pins the matched frame conditions", () => {
        expect(CAPTURE).toEqual({
            width: 1280,
            height: 720,
            time: 6,
            camera: { distance: 18, yaw: 0, pitch: 0.24 },
            sunAzimuthOffset: 0.35,
            sunElevation: 0.32,
        });
        expect(
            Math.hypot(...sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation)),
        ).toBeCloseTo(1, 12);
    });
});
