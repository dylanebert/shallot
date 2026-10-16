import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { attachTexture, Camera, captureTexture } from "../../core/rendering";
import { Transform } from "../../core/transform";
import { DEFAULT_PLUGINS } from "../../standard";
import {
    BackgroundContext,
    Backgrounds,
    backgroundLayout,
    CameraBackground,
    registerBackground,
    StandardRenderer,
} from "./index";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [...DEFAULT_PLUGINS] }]);

test("backgrounds sharing a name retain independent identity on two cameras", async () => {
    const app = subjects()[0];
    const { world } = app;
    const layout = backgroundLayout({});
    const first = registerBackground(world, {
        name: "same",
        layout,
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return d.vec3f(1, 0, 0);
        }),
    });
    const second = registerBackground(world, {
        name: "same",
        layout,
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return d.vec3f(0, 1, 0);
        }),
    });
    expect(first).not.toBe(second);
    const cameras = [first, second].map((background) => {
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        world.add(camera, CameraBackground, { background });
        attachTexture(world, camera, { width: 8, height: 8 });
        return camera;
    });
    world.step(0);
    world.step(0);
    const frames = await Promise.all(cameras.map((camera) => captureTexture(world, camera)));
    expect(frames[0].rgba[0]).toBeGreaterThan(frames[0].rgba[1]!);
    expect(frames[1].rgba[1]).toBeGreaterThan(frames[1].rgba[0]!);
    for (const camera of cameras) world.destroy(camera);
});

test("each background owns identically named bindings and observes resource replacement", async () => {
    const { world } = subjects()[0];
    const color = d.struct({ value: d.vec3f });
    const layout = backgroundLayout({ color: { type: "uniform", struct: color } });
    const red = world.gpu.root.createBuffer(color, { value: d.vec3f(1, 0, 0) }).$usage("uniform");
    const green = world.gpu.root.createBuffer(color, { value: d.vec3f(0, 1, 0) }).$usage("uniform");
    const resources = [{ color: red }, { color: green }];
    const fs = tgpu.fn(
        [BackgroundContext],
        d.vec3f,
    )(() => {
        "use gpu";
        return layout.$.color.value;
    });
    const cameras = resources.map((bindings) => {
        const background = registerBackground(world, {
            name: "owned",
            layout,
            fs,
            bindings: () => bindings,
        });
        expect(world.resource(Backgrounds).get(background)?.name).toBe("owned");
        const camera = world.create();
        world.add(camera, Transform);
        world.add(camera, Camera);
        world.add(camera, StandardRenderer);
        world.add(camera, CameraBackground, { background });
        attachTexture(world, camera, { width: 8, height: 8 });
        return camera;
    });
    world.step(0);
    world.step(0);
    const first = (await captureTexture(world, cameras[0]!)).rgba;
    const second = (await captureTexture(world, cameras[1]!)).rgba;
    expect(first[0]).toBeGreaterThan(first[1]!);
    expect(second[1]).toBeGreaterThan(second[0]!);
    resources[0]!.color = green;
    world.step(0);
    const replaced = (await captureTexture(world, cameras[0]!)).rgba;
    expect(replaced).toEqual(second);
    for (const camera of cameras) world.destroy(camera);
});
