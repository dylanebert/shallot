import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { ImageAddressMode, ImageFilterMode, registerImage } from "../../core/rendering";
import { probeTexture } from "../../engine";
import { MeshRenderPlugin, StandardRenderingPlugin } from "./index";
import {
    capturePlane,
    detailBoundary,
    IMAGE_SIZE,
    planeLod,
    spawnRecedingPlane,
    TONES,
    uploadDiagnosticImages,
} from "./receding-plane.fixture";

setDefaultTimeout(TIMEOUT.gpu);
const repeat = { addressModeU: ImageAddressMode.Repeat, addressModeV: ImageAddressMode.Repeat };
const ids = { hardware: 0, manual: 0, nearest: 0 };
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            {
                name: "RecedingPlaneImages",
                initialize(world) {
                    // The default sampler takes the hardware path; repeat takes the manual one.
                    ids.hardware = registerImage(world, new Blob());
                    ids.manual = registerImage(world, new Blob(), { sampler: repeat });
                    ids.nearest = registerImage(world, new Blob(), {
                        sampler: {
                            ...repeat,
                            magFilter: ImageFilterMode.Nearest,
                            minFilter: ImageFilterMode.Nearest,
                            mipmapFilter: ImageFilterMode.Nearest,
                        },
                    });
                },
            },
        ],
    },
]);

// Observed on Apple M4 Max (Metal), 320x180: one selected level steps 230-270 per level of detail
// within a row or two; filtering between levels stays below 75.
const STEP = 150;

test("a receding plane minifies without a detail boundary; explicit nearest draws nearest texels", async () => {
    const { world } = subjects()[0];
    const texture = uploadDiagnosticImages(world, 3);
    const plane = spawnRecedingPlane(world, ids.hardware, [{ width: 320, height: 180 }]);
    const paths: Record<string, ReturnType<typeof detailBoundary>> = {};
    for (const path of ["hardware", "manual"] as const) {
        const [capture] = await capturePlane(world, plane, ids[path]);
        paths[path] = detailBoundary(capture, 0, 4.5);
    }
    const steepest = Math.max(...Object.values(paths).map((boundary) => boundary.slope));
    expect(steepest, JSON.stringify(paths)).toBeLessThan(STEP);

    const [nearest] = await capturePlane(world, plane, ids.nearest);
    // The measure sees a boundary where one level is requested.
    expect(detailBoundary(nearest, 0, 4.5).slope).toBeGreaterThan(STEP);
    const texels = new Set<number>();
    for (let level = 0; level < texture.mipLevelCount; level++) {
        const size = IMAGE_SIZE >> level;
        const { bytes } = await probeTexture(world, texture, {
            mipLevel: level,
            origin: [0, 0, ids.nearest],
            size: [size, size, 1],
        });
        const red = new Uint8Array(bytes);
        for (let i = 0; i < red.length; i += 4) texels.add(red[i]);
    }
    const near = (value: number, set: Iterable<number>) =>
        [...set].some((texel) => Math.abs(texel - value) <= 1);
    const { rgba, width, height } = nearest;
    const strays: string[] = [];
    let base = 0;
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const lod = planeLod(width, height, x, y);
            if (lod === undefined) continue;
            const value = rgba[(y * width + x) * 4];
            if (!near(value, texels)) strays.push(`(${x}, ${y}) = ${value}`);
            if (lod >= 0.25) continue;
            base++;
            if (!near(value, TONES)) strays.push(`level 0 (${x}, ${y}) = ${value}`);
        }
    expect(strays.slice(0, 8)).toEqual([]);
    expect(base).toBeGreaterThan(width * 10);
});
