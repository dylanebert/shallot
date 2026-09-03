import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CAPTURE, SUN_FACING, sunDirection } from "../src/conditions";

const scene = readFileSync(join(import.meta.dir, "../public/scenes/ocean.scene"), "utf8");

function declaredFrame(source: string) {
    const direction = source
        .match(/direction: ([^;"]+)/)?.[1]
        .split(" ")
        .map(Number);
    const orbit = source.match(/orbit="distance: ([^;]+); yaw: ([^;]+); pitch: ([^"]+)"/);
    if (!direction || !orbit) throw new Error("ocean scene is missing its declared frame");
    return {
        direction,
        camera: { distance: Number(orbit[1]), yaw: Number(orbit[2]), pitch: Number(orbit[3]) },
    };
}

function declaredSky(source: string) {
    const declaration = source.match(/sky="([^"]+)"/)?.[1];
    if (!declaration) throw new Error("ocean scene is missing its declared sky");
    return Object.fromEntries(
        declaration.split(";").map((entry) => {
            const [name, raw] = entry.trim().split(": ");
            const key = name.replace(/-([a-z])/g, (_, letter: string) => letter.toUpperCase());
            return [key, Number(raw)];
        }),
    );
}

function expectDeclaredConditions(source: string) {
    const frame = declaredFrame(source);
    const sun = sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation);
    for (let i = 0; i < sun.length; i++) expect(frame.direction[i]).toBeCloseTo(sun[i], 6);
    expect(frame.camera.distance).toBeCloseTo(CAPTURE.camera.distance, 6);
    expect(frame.camera.yaw).toBeCloseTo(CAPTURE.camera.yaw, 6);
    expect(frame.camera.pitch).toBeCloseTo(CAPTURE.camera.pitch, 6);
    expect(declaredSky(source)).toEqual(CAPTURE.sky);
}

describe("ocean showcase capture", () => {
    test("pins the matched frame conditions", () => {
        expect(CAPTURE).toEqual({
            width: 1280,
            height: 720,
            time: 6,
            camera: { distance: 18, yaw: 0, pitch: 0.24 },
            sunAzimuthOffset: 0.35,
            sunElevation: 0.32,
            sky: {
                zenith: 0x31577e,
                horizon: 0xd5a6a0,
                haze: 0xe8b5a4,
                hazeStrength: 0.16,
                cloud: 0x715f78,
                cloudStrength: 0.18,
                sun: 0xffd5a0,
                sunStrength: 0.42,
                exposure: 1.05,
            },
        });
        expect(
            Math.hypot(...sunDirection(CAPTURE.sunAzimuthOffset, CAPTURE.sunElevation)),
        ).toBeCloseTo(1, 12);
        expect(SUN_FACING).toEqual({
            name: "sun-facing",
            camera: { yaw: Math.PI + CAPTURE.sunAzimuthOffset, pitch: 0 },
        });
    });

    test("scene declares the capture light, camera, and sky", () => {
        expectDeclaredConditions(scene);
    });

    test("scene oracle rejects perturbed camera and sky parameters", () => {
        expect(() =>
            expectDeclaredConditions(scene.replace("pitch: 0.24", "pitch: 0.25")),
        ).toThrow();
        expect(() =>
            expectDeclaredConditions(scene.replace("exposure: 1.05", "exposure: 1.06")),
        ).toThrow();
    });
});
