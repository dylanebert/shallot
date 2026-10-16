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
import { Arrow, Line, LinesPlugin } from "./index";
import { drawArrow } from "./segments";

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
                                drawArrow(world, [-5, -1.5, 0], [1, -1.5, 0], 0xffffff, 3, 1.8);
                        },
                    },
                ],
            },
        ],
    },
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
    world.add(arrow, Line, { offset: [3, 0, 0], color: 0xffffff, thickness: 3 });
    world.add(arrow, Arrow, { tipLength: 1.8 });
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
        world.remove(arrow, Line);
        world.resource(Show).value = false;
        const before = await frame();
        world.add(arrow, Line, { offset: [3, 0, 0], color: 0xffffff, thickness: 3 });
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
