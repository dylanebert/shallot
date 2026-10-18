import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { ImageAddressMode, ImageFilterMode, registerImage } from "../../core/rendering";
import { MeshRenderPlugin, StandardRenderingPlugin } from "./index";
import {
    capturePlane,
    detailBoundary,
    diagnosticChain,
    planeLod,
    referenceError,
    spawnRecedingPlane,
    TONES,
    trilinearReference,
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

// What this covers: the default (hardware) and repeat (manual) paths draw no detail boundary and
// match trilinear filtering of the uploaded chain at the geometry's level of detail, from the near
// rows through lod 4.5; an all-nearest sampler steps and draws only texels of its chain. The nearest
// readback does not establish which level was selected; image-sampling.gpu.test.ts compares
// selection against hardware.

// Observed on Apple M4 Max (Metal), 320x180: one selected level steps 230-270 per level of detail
// within a row or two; filtering between levels stays below 75.
const STEP = 150;
// Row contrast within this of the trilinear reference, lod -1..4.5. Observed: 0.21 hardware and
// 0.05 manual on M4 Max (Metal), with derivatives per 2x2 quad.
const REFERENCE = 1.5;

type Scene = {
    world: ReturnType<typeof subjects>[number]["world"];
    plane: ReturnType<typeof spawnRecedingPlane>;
};
let scene: Scene;
// Upload, the first frame's pipeline compilation and the reference are setup, outside the budget.
beforeAll(async () => {
    const { world } = subjects()[0];
    uploadDiagnosticImages(world, 3);
    const plane = spawnRecedingPlane(world, ids.hardware, [{ width: 320, height: 180 }]);
    await capturePlane(world, plane, ids.hardware);
    trilinearReference(320, 180);
    scene = { world, plane };
});

test("a receding plane minifies as trilinear filtering, without a detail boundary; explicit nearest draws nearest texels", async () => {
    const { world, plane } = scene;
    const paths: Record<string, { slope: number; error: number; lod?: number }> = {};
    for (const path of ["hardware", "manual"] as const) {
        const [capture] = await capturePlane(world, plane, ids[path]);
        const { error, lod } = referenceError(capture, -1, 4.5);
        paths[path] = { slope: detailBoundary(capture, 0, 4.5).slope, error, lod };
    }
    const report = JSON.stringify(paths);
    for (const path of Object.values(paths)) {
        expect(path.slope, report).toBeLessThan(STEP);
        expect(path.error, report).toBeLessThan(REFERENCE);
    }

    const [nearest] = await capturePlane(world, plane, ids.nearest);
    // The measure sees a boundary where one level is requested.
    expect(detailBoundary(nearest, 0, 4.5).slope).toBeGreaterThan(STEP);
    const texels = new Set<number>();
    for (const level of diagnosticChain())
        for (let i = 0; i < level.length; i += 4) texels.add(level[i]);
    const values = [...texels];
    const near = (value: number, set: readonly number[]) =>
        set.some((texel) => Math.abs(texel - value) <= 1);
    const { rgba, width, height } = nearest;
    const strays: string[] = [];
    let base = 0;
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const lod = planeLod(width, height, x, y);
            if (lod === undefined) continue;
            const value = rgba[(y * width + x) * 4];
            if (!near(value, values)) strays.push(`(${x}, ${y}) = ${value}`);
            if (lod >= 0.25) continue;
            base++;
            if (!near(value, TONES)) strays.push(`level 0 (${x}, ${y}) = ${value}`);
        }
    expect(strays.slice(0, 8)).toEqual([]);
    expect(base).toBeGreaterThan(width * 10);
});
