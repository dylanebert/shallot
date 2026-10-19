import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { DEFAULT_PLUGINS } from "../../standard";
import { corners, LOWER, pixel, SIZE, stackedFrame, stackedScene } from "../stacked-camera.fixture";
import { Fog, FogPlugin } from ".";

setDefaultTimeout(TIMEOUT.node);

const subjects = gpuApps(import.meta.path, [
    { defaults: false, plugins: [...DEFAULT_PLUGINS, FogPlugin] },
]);

test("fog on a camera stacked over another reaches the lower image only through that camera's coverage", async () => {
    const { world } = subjects()[0];
    const { lower } = stackedScene(world, 1);
    const clear = pixel(await stackedFrame(world, lower), SIZE / 2, SIZE / 2);
    world.add(world.create(), Fog, { density: 0.1, color: 0xffffff });
    const rgba = await stackedFrame(world, lower);
    expect(new Set(corners(rgba).map(String))).toEqual(new Set([String(LOWER)]));
    // A partly covered edge pixel is the fogged cube blended over the lower clear by one coverage, so
    // every channel gives the same blend weight, within dither.
    const surface = pixel(rgba, SIZE / 2, SIZE / 2);
    expect(surface[2]!).toBeGreaterThan(clear[2]! + 20);
    const near = (a: number[], b: number[]) =>
        a.every((value, c) => c > 2 || Math.abs(value - b[c]!) <= 2);
    let edges = 0;
    const outside: string[] = [];
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const value = pixel(rgba, x, y);
            if (near(value, LOWER) || near(value, surface)) continue;
            edges++;
            const weights = [0, 1, 2].map(
                (c) => (value[c]! - LOWER[c]!) / (surface[c]! - LOWER[c]!),
            );
            if (Math.max(...weights) - Math.min(...weights) > 0.1 || Math.min(...weights) < 0)
                outside.push(`${x},${y}: ${value} weights ${weights.map((w) => w.toFixed(2))}`);
        }
    }
    expect(edges).toBeGreaterThan(8);
    expect(outside).toEqual([]);
});
