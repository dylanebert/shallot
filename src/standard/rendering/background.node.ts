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
    world.storage(CameraBackground).background.set(cameras[0]!, second);
    world.step(0);
    expect((await captureTexture(world, cameras[0]!)).rgba).toEqual(frames[1].rgba);
    for (const camera of cameras) world.destroy(camera);
});

test("background bindings observe a resized storage resource", async () => {
    const { world } = subjects()[0];
    const layout = backgroundLayout({ colors: { type: "storage", element: d.vec3f } });
    const bindings = {
        colors: world.gpu.root
            .createBuffer(d.arrayOf(d.vec3f, 1), [d.vec3f(1, 0, 0)])
            .$usage("storage"),
    };
    const background = registerBackground(world, {
        name: "resized",
        layout,
        fs: tgpu.fn(
            [BackgroundContext],
            d.vec3f,
        )(() => {
            "use gpu";
            return layout.$.colors[0]!;
        }),
        bindings: () => bindings,
    });
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, CameraBackground, { background });
    attachTexture(world, camera, { width: 8, height: 8 });
    world.step(0);
    const before = (await captureTexture(world, camera)).rgba;
    expect(before[0]).toBeGreaterThan(before[1]!);
    bindings.colors = world.gpu.root
        .createBuffer(d.arrayOf(d.vec3f, 2), [d.vec3f(0, 1, 0), d.vec3f(0)])
        .$usage("storage");
    world.step(0);
    const after = (await captureTexture(world, camera)).rgba;
    expect(after[1]).toBeGreaterThan(after[0]!);
    world.destroy(camera);
});

test("each background observes changed binding records and replaced resources", async () => {
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
    const cameras = resources.map((_, index) => {
        const background = registerBackground(world, {
            name: "owned",
            layout,
            fs,
            bindings: () => resources[index]!,
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
    resources[0] = { color: red };
    world.step(0);
    expect((await captureTexture(world, cameras[0]!)).rgba).toEqual(first);
    for (const camera of cameras) world.destroy(camera);
});
