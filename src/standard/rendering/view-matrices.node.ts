import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    CameraMode,
    captureTexture,
    computeViewFromWorld,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import { orthographic, perspective } from "../../engine";
import {
    engineLayout,
    MaterialPlugin,
    MeshMaterial,
    MeshRenderPlugin,
    materialType,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.node);

const ExpectedView = d.struct({ viewFromWorld: d.mat4x4f, clipFromView: d.mat4x4f });
// Green where the bound ViewUniforms carries exactly the expected matrices, red elsewhere.
// Multiplying by a basis vector reads one column exactly.
const ViewMatrixMaterial = materialType({
    name: "ViewMatrixMaterial",
    parameters: ExpectedView,
    vertex: (input) => {
        "use gpu";
        return { world: input.world, worldNormal: input.worldNormal };
    },
    fragment: (ctx) => {
        "use gpu";
        const view = engineLayout.$.view;
        let same = true;
        for (let i = 0; i < 4; i++) {
            const basis = d.vec4f(
                std.select(0, 1, i === 0),
                std.select(0, 1, i === 1),
                std.select(0, 1, i === 2),
                std.select(0, 1, i === 3),
            );
            same =
                same &&
                std.all(
                    std.eq(
                        std.mul(view.viewFromWorld, basis),
                        std.mul(ctx.parameters.viewFromWorld, basis),
                    ),
                ) &&
                std.all(
                    std.eq(
                        std.mul(view.clipFromView, basis),
                        std.mul(ctx.parameters.clipFromView, basis),
                    ),
                );
        }
        return std.select(d.vec4f(1, 0, 0, 1), d.vec4f(0, 1, 0, 1), same);
    },
    defaults: { viewFromWorld: d.mat4x4f.identity(), clipFromView: d.mat4x4f.identity() },
});

const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [StandardRenderingPlugin, MeshRenderPlugin, MaterialPlugin(ViewMatrixMaterial)],
    },
]);

const mat4 = (m: Float32Array) =>
    d.mat4x4f(
        d.vec4f(m[0]!, m[1]!, m[2]!, m[3]!),
        d.vec4f(m[4]!, m[5]!, m[6]!, m[7]!),
        d.vec4f(m[8]!, m[9]!, m[10]!, m[11]!),
        d.vec4f(m[12]!, m[13]!, m[14]!, m[15]!),
    );

test("a material reads its view's viewFromWorld and clipFromView as the CPU computes them, perspective and orthographic", async () => {
    const { world } = subjects()[0];
    const yaw = 0.3;
    const camera = world.create();
    world.add(camera, Transform, {
        translation: [1, 0.5, 5, 0],
        rotation: [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
    });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 64, height: 32 });
    const materials = world.resource(ViewMatrixMaterial);
    const handle = materials.add({
        viewFromWorld: d.mat4x4f.identity(),
        clipFromView: d.mat4x4f.identity(),
    });
    const cube = world.create();
    world.add(cube, Transform, {
        translation: [1 - 4 * Math.sin(yaw), 0.5, 5 - 4 * Math.cos(yaw), 0],
    });
    world.add(cube, MeshInstance);
    world.add(cube, MeshMaterial, handle);

    const viewFromWorld = new Float32Array(16);
    const clipFromView = new Float32Array(16);
    const center = async (expected: {
        viewFromWorld: Float32Array;
        clipFromView: Float32Array;
    }) => {
        materials.update(handle, {
            viewFromWorld: mat4(expected.viewFromWorld),
            clipFromView: mat4(expected.clipFromView),
        });
        world.step(0);
        world.step(0);
        const frame = await captureTexture(world, camera);
        const at = (16 * 64 + 32) * 4;
        return Array.from(frame.rgba.subarray(at, at + 4));
    };
    const cameras = world.storage(Camera);
    const aspect = 64 / 32;

    world.step(0);
    computeViewFromWorld(world, camera, viewFromWorld);
    perspective(cameras.fov.get(camera), aspect, cameras.near.get(camera), clipFromView);
    expect(await center({ viewFromWorld, clipFromView })).toEqual([0, 255, 0, 255]);
    const perturbed = clipFromView.slice();
    perturbed[0]! += 1e-3;
    expect(await center({ viewFromWorld, clipFromView: perturbed })).toEqual([255, 0, 0, 255]);

    cameras.mode.set(camera, CameraMode.Orthographic);
    world.step(0);
    orthographic(
        cameras.size.get(camera),
        aspect,
        cameras.near.get(camera),
        cameras.far.get(camera),
        clipFromView,
    );
    expect(await center({ viewFromWorld, clipFromView })).toEqual([0, 255, 0, 255]);
});
