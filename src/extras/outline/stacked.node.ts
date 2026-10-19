import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DEFAULT_PLUGINS } from "../../standard";
import { corners, LOWER, stackedFrame, stackedScene } from "../stacked-camera.fixture";
import { Outline, OutlinePlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, OutlinePlugin] },
]);

test("an outline on a camera stacked over another leaves the lower image where it draws no band", async () => {
    const { world } = subjects()[0];
    const { lower, cube } = stackedScene(world, 1);
    world.add(cube, Outline, { width: 3, color: 0xff00ff });
    const rgba = await stackedFrame(world, lower);
    expect(new Set(corners(rgba).map(String))).toEqual(new Set([String(LOWER)]));
    let band = 0;
    for (let i = 0; i < rgba.length; i += 4)
        if (rgba[i]! > 200 && rgba[i + 2]! > 200 && rgba[i + 1]! < 100) band++;
    expect(band).toBeGreaterThan(20);
});
