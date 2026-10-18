import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    Materials,
    MeshMaterial,
    StandardMaterial,
    StandardRenderer,
} from "../../standard/rendering";
import { Fog, FogPlugin, NoFog } from ".";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);
const decode = (v: number) => {
    const x = v / 255;
    return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};
const encode = (x: number) => 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
test("MSAA opaque edges blend fog at each covered surface depth", async () => {
    const { world } = subjects()[0]!;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, {
        mode: CameraMode.Orthographic,
        size: 10,
        near: 0.1,
        far: 30,
        antialias: 1,
    });
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    const width = 160,
        height = 90;
    attachTexture(world, camera, { width, height });
    const mats = world.resource(Materials);
    const farRow = mats.add(StandardMaterial({ baseColor: [0, 0, 0, 1], unlit: true }));
    const nearRow = mats.add(StandardMaterial({ baseColor: [0, 0, 0, 1], unlit: true }));
    const wall = world.create();
    world.add(wall, Transform, { translation: [0, 0, -20.5, 0], scale: [80, 80, 1, 0] });
    world.add(wall, MeshInstance);
    world.add(wall, MeshMaterial, farRow);
    const near = world.create();
    world.add(near, Transform, {
        translation: [-2.7, 0, -4.5, 0],
        scale: [5.4, 40, 1, 0],
        rotation: [0, 0, Math.sin(0.075), Math.cos(0.075)],
    });
    world.add(near, MeshInstance);
    world.add(near, MeshMaterial, nearRow);
    const fog = world.create();
    world.add(fog, Fog, {
        color: 0xffffff,
        density: 0.08,
        absorption: 0,
        scattering: 1,
        // Only the medium's own light: its haze is color · (1 − T).
        lightIntensity: 0,
        viewDistance: 32,
    });
    const capture = async () => {
        for (let i = 0; i < 3; i++) world.step(0);
        return (await captureTexture(world, camera)).rgba;
    };
    const reports = [];
    for (const aa of [1, 0]) {
        world.storage(Camera).antialias.set(camera, aa);
        const actual = await capture();
        world.add(camera, NoFog);
        mats.update(nearRow, { baseColor: [1, 1, 1, 1] });
        const coverage = await capture();
        mats.update(nearRow, { baseColor: [0, 0, 0, 1] });
        world.remove(camera, NoFog);
        const rows = [];
        for (let y = 8; y < height - 8; y++) {
            const n = actual[(y * width + 68) * 4]!;
            const f = actual[(y * width + 140) * 4]!;
            for (let x = 60; x < 100; x++) {
                const p = (y * width + x) * 4;
                const c = decode(coverage[p]!);
                if ((c > 0.01 && c < 0.99) || (aa === 0 && coverage[p] !== coverage[p + 4])) {
                    const expected = encode(c * decode(n) + (1 - c) * decode(f));
                    rows.push({
                        x,
                        y,
                        coverage: c,
                        actual: actual[p],
                        expected,
                        near: n,
                        far: f,
                        error: Math.abs(actual[p]! - expected),
                    });
                }
            }
        }
        reports.push({
            aa,
            width,
            height,
            edgePixels: rows.length,
            maxError: Math.max(0, ...rows.map((r) => r.error)),
            meanError: rows.length ? rows.reduce((a, r) => a + r.error, 0) / rows.length : 0,
            rows,
        });
    }
    for (const report of reports) {
        console.log({
            aa: report.aa,
            edgePixels: report.edgePixels,
            maxError: report.maxError,
            meanError: report.meanError,
        });
        if (report.aa === 1) expect(report.edgePixels).toBeGreaterThan(20);
        for (const pixel of report.rows) {
            expect(pixel.actual!).toBeGreaterThanOrEqual(pixel.near - 1);
            expect(pixel.actual!).toBeLessThanOrEqual(pixel.far + 1);
            // Two encoded levels cover 8-bit endpoint/coverage quantization and the HDR resolve.
            // Sample-zero fog instead misses the coverage blend by tens of levels.
            expect(pixel.error).toBeLessThanOrEqual(2);
        }
    }
});
