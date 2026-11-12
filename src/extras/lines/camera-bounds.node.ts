import { expect, setDefaultTimeout, test } from "bun:test";
import { StandardRenderingPlugin, Transform } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuScene } from "../../../scripts/gpu.fixture";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    DebandDither,
} from "../../core/rendering";
import { drawArrow, drawLine, Gizmo, GizmoAsset, LinesPlugin, registerGizmo } from "./index";

import { SEGMENT_FLOATS } from "./pipeline";
import { segmentState } from "./segments";

setDefaultTimeout(TIMEOUT.node);
const subject = gpuScene(
    import.meta.path,
    { defaults: false, plugins: [StandardRenderingPlugin, LinesPlugin] },
    (world) => {
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera, { near: 1, far: 10, clearColor: 0, antialias: 0 });
        world.add(camera, DebandDither, { enabled: 0 });
        attachTexture(world, camera, { width: 128, height: 128 });
        const line = world.create();
        world.add(line, Transform);
        return { camera, line };
    },
);

test("a composed perspective Camera excludes lines wholly beyond its authored far bound", async () => {
    const {
        world,
        scene: { camera, line },
    } = subject();
    console.info("camera-bounds adapter:", world.gpu.adapter);
    const asset = new GizmoAsset();
    drawLine(asset, [-2, 0, -12], [2, 0, -12], 0xffffff);
    world.add(line, Gizmo, { handle: registerGizmo(world, asset), width: 3 });
    world.step(1 / 60);
    world.step(1 / 60);
    const { rgba } = await captureTexture(world, camera);
    let lit = 0;
    for (let i = 0; i < rgba.length; i += 4) if (rgba[i]! > 40) lit++;
    expect(lit).toBe(0);
    const distant = world.create();
    world.add(distant, Transform);
    world.add(distant, Camera, { near: 1, far: 20, clearColor: 0, antialias: 0 });
    world.add(distant, DebandDither, { enabled: 0 });
    attachTexture(world, distant, { width: 128, height: 128 });
    world.step(1 / 60);
    world.step(1 / 60);
    expect(
        mask((await captureTexture(world, distant)).rgba).filter(Boolean).length,
    ).toBeGreaterThan(20);
    expect(mask((await captureTexture(world, camera)).rgba).filter(Boolean).length).toBe(0);
});

type Point = [number, number, number];

// Independent world-space polygon clipping makes a reference that never crosses either bound.
function clippedArrow(source: GizmoAsset): GizmoAsset {
    const result = new GizmoAsset();
    const input = source.segments;
    const output = result.segments;
    for (let row = 0; row < input.count; row++) {
        const offset = row * SEGMENT_FLOATS;
        const triangle = input.u32[offset + 11] !== 0;
        let points: Point[] = Array.from(
            { length: triangle ? 3 : 2 },
            (_, i) => Array.from(input.f32.subarray(offset + i * 4, offset + i * 4 + 3)) as Point,
        );
        for (const [bound, sign] of [
            [-1, -1],
            [-10, 1],
        ]) {
            const clipped: Point[] = [];
            const edges = triangle ? points.length : Math.max(0, points.length - 1);
            for (let i = 0; i < edges; i++) {
                const a = points[i]!,
                    b = points[(i + 1) % points.length]!;
                const da = (a[2] - bound!) * sign!,
                    db = (b[2] - bound!) * sign!;
                if (da >= 0) clipped.push(a);
                if (da < 0 !== db < 0) {
                    const t = da / (da - db);
                    clipped.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, bound!]);
                }
                if (!triangle && db >= 0) clipped.push(b);
            }
            points = clipped;
        }
        const faces = triangle ? Math.max(0, points.length - 2) : Number(points.length === 2);
        for (let face = 0; face < faces; face++) {
            const at = output.count++ * SEGMENT_FLOATS;
            output.u32.set(input.u32.subarray(offset, offset + SEGMENT_FLOATS), at);
            const vertices = triangle ? [points[0]!, points[face + 1]!, points[face + 2]!] : points;
            vertices.forEach((point, i) => {
                output.f32.set(point, at + i * 4);
            });
        }
    }
    return result;
}

for (const retained of [false, true]) {
    test(`${retained ? "retained" : "immediate"} arrow triangles match independently clipped near/far polygons in both orders`, async () => {
        const {
            world,
            scene: { camera, line },
        } = subject();
        world.step(1 / 60);
        world.step(1 / 60);
        async function render(asset: GizmoAsset) {
            world.remove(line, Gizmo);
            if (retained)
                world.add(line, Gizmo, {
                    handle: registerGizmo(world, asset),
                    width: 3,
                    depthBias: -1,
                });
            else {
                const output = segmentState(world);
                output.u32.set(
                    asset.segments.u32.subarray(0, asset.segments.count * SEGMENT_FLOATS),
                );
                output.count = asset.segments.count;
                for (let i = 0; i < output.count; i++) {
                    output.f32[i * SEGMENT_FLOATS + 3] = 3;
                    output.f32[i * SEGMENT_FLOATS + 12] = -1;
                    output.layers[i] = 1;
                }
            }
            world.step(0);
            return mask((await captureTexture(world, camera)).rgba);
        }
        for (const reverse of [false, true]) {
            for (const [a, b] of [
                [
                    [-0.4, 0, -0.5],
                    [2, 0, -5],
                ],
                [
                    [8, 0, -15],
                    [-2, 0, -5],
                ],
            ] as [Point, Point][]) {
                const asset = new GizmoAsset();
                drawArrow(asset, reverse ? b : a, reverse ? a : b, 0xffffff, undefined, 8, [
                    b[2] - a[2],
                    0,
                    a[0] - b[0],
                ]);
                const actual = await render(asset);
                const expected = await render(clippedArrow(asset));
                expect(expected.filter(Boolean).length).toBeGreaterThan(10);
                expect(
                    actual.filter((value, i) => value !== expected[i]).length,
                ).toBeLessThanOrEqual(3);
            }
        }
    });
}

function mask(rgba: Uint8ClampedArray) {
    return Array.from({ length: rgba.length / 4 }, (_, i) => rgba[i * 4]! > 40);
}

for (const retained of [false, true]) {
    for (const mode of [CameraMode.Perspective, CameraMode.Orthographic]) {
        test(`${retained ? "retained" : "immediate"} camera bounds clip crossings in either order (${mode}) before bias`, async () => {
            const {
                world,
                scene: { camera, line },
            } = subject();
            world.storage(Camera).mode.set(camera, mode);
            world.step(1 / 60);
            world.step(1 / 60);
            async function render(a: Point, b: Point, bias: number, arrow = false) {
                world.remove(line, Gizmo);
                const asset = new GizmoAsset();
                if (retained) {
                    if (arrow) drawArrow(asset, a, b, 0xffffff, undefined, 1, [0, 0, 1]);
                    else drawLine(asset, a, b, 0xffffff);
                    world.add(line, Gizmo, {
                        handle: registerGizmo(world, asset),
                        width: 3,
                        depthBias: bias,
                    });
                } else if (arrow)
                    drawArrow(world, a, b, 0xffffff, { width: 3, depthBias: bias }, 1, [0, 0, 1]);
                else drawLine(world, a, b, 0xffffff, { width: 3, depthBias: bias });
                world.step(0);
                return mask((await captureTexture(world, camera)).rgba);
            }
            for (const bias of [-1, 0, 1]) {
                for (const reverse of [false, true]) {
                    for (const z of [-0.5, -12]) {
                        const a: Point = [-0.2, 0, z],
                            b: Point = [0.2, 0, z];
                        for (const arrow of [false, true]) {
                            const pixels = await render(
                                reverse ? b : a,
                                reverse ? a : b,
                                bias,
                                arrow,
                            );
                            expect(pixels.filter(Boolean).length).toBe(0);
                        }
                    }
                    for (const [a, b, clipped] of [
                        [
                            [-0.4, 0, -0.5],
                            [2, 0, -5],
                            [-0.4 + 2.4 / 9, 0, -1],
                        ],
                        [
                            [8, 0, -15],
                            [-2, 0, -5],
                            [3, 0, -10],
                        ],
                    ] as [Point, Point, Point][]) {
                        const actual = await render(reverse ? b : a, reverse ? a : b, bias);
                        const expected = await render(
                            reverse ? b : clipped,
                            reverse ? clipped : b,
                            bias,
                        );
                        expect(expected.filter(Boolean).length).toBeGreaterThan(5);
                        expect(
                            actual.filter((value, i) => value !== expected[i]).length,
                        ).toBeLessThanOrEqual(2);
                    }
                }
            }
        });
    }
}
