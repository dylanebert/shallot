import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DEFAULT_PLUGINS } from "../../standard";
import { corners, LOWER, pixel, SIZE, stackedFrame, stackedScene } from "../stacked-camera.fixture";
import { Vignette, VignettePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, VignettePlugin] },
]);

test("a vignette on a camera stacked over another shades only what that camera drew", async () => {
    const { world } = subjects()[0];
    const { lower, upper } = stackedScene(world, 1);
    const plain = pixel(await stackedFrame(world, lower), SIZE / 2, SIZE / 2 + 6);
    world.add(upper, Vignette, { radius: 0.2 });
    const rgba = await stackedFrame(world, lower);
    expect(new Set(corners(rgba).map(String))).toEqual(new Set([String(LOWER)]));
    const shaded = pixel(rgba, SIZE / 2, SIZE / 2 + 6);
    expect(shaded[0]!).toBeLessThan(plain[0]! - 20);
});
