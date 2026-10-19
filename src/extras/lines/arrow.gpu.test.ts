import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    BeginFrameSystem,
    Camera,
    CameraMode,
    captureTexture,
    computeViewProj,
    DebandDither,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { lookAtRotation } from "../../engine/utils";
import { Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";
import { SEGMENT_FLOATS } from "./pipeline";
import {
    drawArrow,
    drawLine,
    drawWireBox,
    lineCount,
    lineStableIds,
    lineStaging,
} from "./segments";

setDefaultTimeout(TIMEOUT.gpu);
const Show = { create: () => ({ value: false }) };
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            {
                name: "ArrowCheck",
                dependencies: [LinesPlugin],
                systems: [
                    {
                        name: "immediate",
                        group: "draw",
                        after: [BeginFrameSystem],
                        before: LinesPlugin.systems,
                        update(world) {
                            if (world.resource(Show).value)
                                drawArrow(
                                    world,
                                    [-5, -1.5, 0],
                                    [1, -1.5, 0],
                                    0xffffff,
                                    { width: 3 },
                                    1.8,
                                );
                        },
                    },
                ],
            },
        ],
    },
    { defaults: false, plugins: [LinesPlugin] },
]);

test("different-length retained and immediate arrows given the same tip length fill the same head from two views", async () => {
    const { world } = subjects()[0];
    console.info("arrow adapter:", world.gpu.adapter);
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 5, clearColor: 0 });
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: 128, height: 128 });
    const arrow = world.create();
    world.add(arrow, Transform, { translation: [-2, 0, 0] });
    const asset = new GizmoAsset();
    drawArrow(asset, [0, 0, 0], [3, 0, 0], 0xffffff, undefined, 1.8);
    const handle = registerGizmo(world, asset);
    world.add(arrow, Gizmo, { handle, width: 3 });
    const frame = async () => {
        world.step(1 / 60);
        world.step(1 / 60);
        return (await captureTexture(world, camera)).rgba;
    };
    for (const eye of [d.vec3f(3, 4, 6), d.vec3f(3, 1, -8)]) {
        world.storage(Transform).translation.set(camera, eye.x, eye.y, eye.z);
        world
            .storage(Transform)
            .rotation.set(camera, ...lookAtRotation(eye.x, eye.y, eye.z, 0, -0.75, 0));
        world.remove(arrow, Gizmo);
        world.resource(Show).value = false;
        const before = await frame();
        world.add(arrow, Gizmo, { handle, width: 3 });
        world.resource(Show).value = true;
        const after = await frame();
        const matrix = new Float32Array(16);
        computeViewProj(world, camera, 1, matrix);
        for (const y of [0, -1.5]) {
            const tip = d.vec3f(1, y, 0);
            const left = d.vec3f(-0.8, y, -0.9),
                right = d.vec3f(-0.8, y, 0.9);
            for (const side of [-1, 1]) {
                const p = [0, 1, 2].map(
                    (i) =>
                        tip[i] * 0.2 + left[i] * (0.4 + side * 0.2) + right[i] * (0.4 - side * 0.2),
                );
                const clip = [0, 1, 3].map(
                    (row) =>
                        matrix[row] * p[0] +
                        matrix[row + 4] * p[1] +
                        matrix[row + 8] * p[2] +
                        matrix[row + 12],
                );
                const x = Math.floor((clip[0] / clip[2] + 1) * 64),
                    pixelY = Math.floor((1 - clip[1] / clip[2]) * 64);
                const at = (pixelY * 128 + x) * 4;
                expect(after[at]).toBeGreaterThan(before[at] + 100);
            }
        }
    }
});

test("one asset's line, arrow and box move all fifteen primitives with its entity, leaving the asset unchanged", async () => {
    const { world } = subjects()[1];
    const camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5] });
    world.add(camera, Camera, { mode: CameraMode.Orthographic, size: 5, clearColor: 0 });
    world.add(camera, DebandDither, { enabled: 0 });
    attachTexture(world, camera, { width: 128, height: 128 });
    const asset = new GizmoAsset();
    drawLine(asset, [-0.6, 0.6, 0], [0.6, 0.6, 0], 0xffaa55);
    drawArrow(asset, [-0.6, -0.6, 0], [0.6, -0.6, 0], 0x55aaff, undefined, 0.3, [0, 0, 1]);
    drawWireBox(asset, [-0.4, -0.3, -0.2], [0.4, 0.3, 0.2], 0xffffff);
    const source = asset.segments.f32;
    const original = new Uint32Array(source.buffer).slice();
    expect(asset.segments.count).toBe(15);
    const entity = world.create();
    world.add(entity, Transform, { translation: [-1, 0, 0] });
    world.add(entity, Gizmo, { handle: registerGizmo(world, asset) });
    const frame = async () => {
        world.step(1 / 60);
        world.step(1 / 60);
        return (await captureTexture(world, camera)).rgba;
    };
    const before = await frame();
    world.storage(Transform).translation.set(entity, 1, 0, 0);
    const after = await frame();
    expect(asset.segments.f32).toBe(source);
    expect(new Uint32Array(source.buffer)).toEqual(original);
    expect(lineCount(world)).toBe(15);
    const staged = lineStaging(world);
    for (let i = 0; i < 15; i++) {
        expect(lineStableIds(world)[i]).toBe(entity);
        const at = i * SEGMENT_FLOATS;
        for (const point of [0, 4, ...(asset.segments.u32[at + 11] ? [8] : [])]) {
            expect(staged[at + point]).toBeCloseTo(source[at + point] + 1, 6);
            expect(staged[at + point + 1]).toBeCloseTo(source[at + point + 1], 6);
            expect(staged[at + point + 2]).toBeCloseTo(source[at + point + 2], 6);
        }
        expect(staged[at + 3]).toBe(2);
    }
    const centroid = (pixels: Uint8ClampedArray) => {
        let count = 0,
            x = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i] + pixels[i + 1] + pixels[i + 2] < 100) continue;
            count++;
            x += (i / 4) % 128;
        }
        expect(count).toBeGreaterThan(100);
        return x / count;
    };
    expect(centroid(after) - centroid(before)).toBeGreaterThan(20);
});
