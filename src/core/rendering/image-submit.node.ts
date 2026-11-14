import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { createApp, type World } from "../../engine";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    engineLayout,
    MaterialPlugin,
    MeshMaterial,
    materialLayout,
    materialType,
    StandardRenderer,
} from "../../standard/rendering";
import { Meshes, MeshInstance } from "../mesh";
import { Transform } from "../transform";
import {
    attachTexture,
    Billboard,
    billboardCorner,
    Camera,
    CameraMode,
    captureTexture,
    type ImageSource,
    prepareImages,
    registerImage,
    replaceImage,
    Tonemapping,
    TonemappingMethod,
    xformMat,
} from "./index";
import { PresentationSystem } from "./substrate";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();

// Replica's ChipMaterial verbatim.
const chipLayout = materialLayout(d.struct({ image: d.u32 }), {
    image: { type: "image", field: "image", isSrgb: false },
});
const ChipMaterial = materialType({
    name: "OfflineCanvasChip",
    layout: chipLayout,
    defaults: { image: 0 },
    fragmentInputs: { uv: true },
    vertex: (input) => {
        "use gpu";
        const world = billboardCorner(
            Billboard.Screen,
            xformMat(input.xform),
            engineLayout.$.view.right.xyz,
            engineLayout.$.view.up.xyz,
            input.localPos.x,
            input.localPos.y,
        );
        return { world: d.vec4f(world, 1), worldNormal: input.worldNormal };
    },
    fragment: (ctx) => {
        "use gpu";
        return std.textureSample(
            chipLayout.$.image,
            chipLayout.$.imageSampler,
            d.vec2f(ctx.uv.x, 1 - ctx.uv.y),
        );
    },
});

const ChipPlugin = MaterialPlugin(ChipMaterial);
const pixel =
    (v: number): ImageSource =>
    () => ({ width: 1, height: 1, format: "rgba8unorm", levels: [new Uint8Array([v, 0, 0, 255])] });

for (const where of ["between steps", "draw after presentation"] as const)
    test(`replaceImage ${where}`, async () => {
        let camera = 0,
            id = -1,
            armed = false;
        const errors: string[] = [];
        const app = await createApp({
            defaults: false,
            plugins: [
                ...DEFAULT_PLUGINS,
                ChipPlugin,
                {
                    name: "ReleaseTiming",
                    dependencies: [...DEFAULT_PLUGINS, ChipPlugin],
                    initialize(w: World) {
                        camera = w.create();
                        w.add(camera, Transform, { translation: [0, 0, 5, 0] });
                        w.add(camera, Camera, {
                            mode: CameraMode.Orthographic,
                            size: 1,
                            antialias: 0,
                            clearColor: 0,
                        });
                        w.add(camera, StandardRenderer);
                        w.add(camera, Tonemapping, { method: TonemappingMethod.None });
                        attachTexture(w, camera, { width: 8, height: 8 });
                        id = registerImage(w, pixel(255), { isSrgb: false });
                        const plane = w.resource(Meshes).add({
                            name: "plane",
                            vertices: new Float32Array([
                                -0.5, -0.5, 0, 0, 0, 0, 1, 0, 0.5, -0.5, 0, 1, 0, 0, 1, 0, 0.5, 0.5,
                                0, 1, 0, 0, 1, 1, -0.5, 0.5, 0, 0, 0, 0, 1, 1,
                            ]),
                            indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
                        });
                        const e = w.create();
                        w.add(e, Transform, { scale: [4, 4, 1] });
                        w.add(e, MeshInstance, { mesh: plane });
                        w.add(
                            e,
                            MeshMaterial,
                            w.resource(ChipMaterial).add({ image: id }) as never,
                        );
                    },
                    systems: [
                        {
                            name: "replace-in-draw",
                            group: "draw",
                            after: [PresentationSystem],
                            update(w: World) {
                                if (!armed) return;
                                armed = false;
                                replaceImage(w, id, pixel(64), { isSrgb: false });
                            },
                        },
                    ],
                },
            ],
        });
        const world = app.world;
        world.gpu.device.addEventListener("uncapturederror", (e) =>
            errors.push((e as GPUUncapturedErrorEvent).error.message),
        );
        world.step(1 / 60);
        const before = (await captureTexture(world, camera)).rgba.slice(0, 4);
        let thrown: unknown;
        try {
            if (where === "between steps") replaceImage(world, id, pixel(64), { isSrgb: false });
            else armed = true;
            world.step(1 / 60);
            await world.gpu.device.queue.onSubmittedWorkDone();
        } catch (error) {
            thrown = error;
        }
        await prepareImages(world);
        let after: number[] = [];
        let later: unknown;
        try {
            world.step(1 / 60);
            after = Array.from((await captureTexture(world, camera)).rgba.slice(0, 4));
        } catch (error) {
            later = error;
        }
        let recovered: number[] = [];
        let third: unknown;
        try {
            world.step(1 / 60);
            recovered = Array.from((await captureTexture(world, camera)).rgba.slice(0, 4));
        } catch (error) {
            third = String(error).slice(0, 80);
        }
        app.dispose();
        expect(Array.from(before)).toEqual([255, 0, 0, 255]);
        expect(thrown).toBeUndefined();
        expect(later).toBeUndefined();
        expect(third).toBeUndefined();
        expect(errors).toEqual([]);
        expect(after).toEqual([137, 0, 0, 255]);
        expect(recovered).toEqual([137, 0, 0, 255]);
    });
