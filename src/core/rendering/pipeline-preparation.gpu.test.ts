import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import { drawLine, LinesPlugin } from "../../extras/lines";
import { TextPlugin } from "../../extras/text";
import { Transform } from "../transform";
import { attachTexture, Camera } from ".";

setDefaultTimeout(TIMEOUT.gpu);

const labels: string[][] = [];
const subjects = gpuApps(
    import.meta.path,
    [undefined, 0, 1, "auto" as const, "warm" as const, "stack" as const].map((variant, i) => ({
        defaults: false,
        pixelRatio: variant === "auto" ? ("auto" as const) : 1,
        plugins: [
            {
                name: "PreparationControl",
                dependencies: [TextPlugin, LinesPlugin],
                initialize(world) {
                    labels[i] = [];
                    const device = world.gpu.device;
                    const sync = device.createRenderPipeline.bind(device);
                    const async = device.createRenderPipelineAsync.bind(device);
                    device.createRenderPipeline = (descriptor) => {
                        labels[i]!.push(descriptor.label ?? "");
                        return sync(descriptor);
                    };
                    device.createRenderPipelineAsync = (descriptor) => {
                        labels[i]!.push(descriptor.label ?? "");
                        return async(descriptor);
                    };
                    if (typeof variant === "number") {
                        const camera = world.create();
                        world.add(camera, Transform);
                        world.add(camera, Camera, { antialias: variant });
                        attachTexture(world, camera, { width: 8, height: 8 });
                    }
                },
                warm(world) {
                    if (variant === "warm") {
                        const camera = world.create();
                        world.add(camera, Transform);
                        world.add(camera, Camera, { antialias: 0 });
                        attachTexture(world, camera, { width: 8, height: 8 });
                    }
                    if (variant === "stack") {
                        const lower = world.create();
                        world.add(lower, Transform);
                        world.add(lower, Camera, { antialias: 0 });
                        const target = attachTexture(world, lower, { width: 8, height: 8 });
                        const upper = world.create();
                        world.add(upper, Transform);
                        world.add(upper, Camera, { antialias: 0, order: 1 });
                        attachTexture(world, upper, target);
                    }
                },
                systems: [
                    {
                        name: "late-line",
                        group: "draw",
                        before: LinesPlugin.systems,
                        update(world) {
                            drawLine(world, [0, 0, -2], [1, 0, -2], 0xffffff);
                        },
                    },
                ],
            },
        ],
    })),
);

for (const [index, count] of [
    [0, 0],
    [1, 1],
    [2, 4],
] as const) {
    test(`initial sample count ${count}: absent text/line variants compile none, present variants compile once`, () => {
        subjects();
        for (const family of ["text", "lines"]) {
            const actual = labels[index]!.filter((label) =>
                new RegExp(`^${family}-[14]$`).test(label),
            );
            expect(actual).toEqual(count ? [`${family}-${count}`] : []);
        }
        expect(labels[index]!.filter((label) => label === "upscaling")).toEqual([]);
    });
}
test("a camera declared by a dependent warm hook prepares only its reachable sample count", () => {
    subjects();
    expect(labels[4]!.filter((label) => /^(text|lines)-[14]$/.test(label)).sort()).toEqual([
        "lines-1",
        "text-1",
    ]);
    expect(labels[4]!.filter((label) => label === "upscaling")).toEqual([]);
});
test("a fixed-ratio initial stack prepares one blit and deduplicates its shared sample count", () => {
    subjects();
    expect(labels[5]!.filter((label) => /^(text|lines)-[14]$/.test(label)).sort()).toEqual([
        "lines-1",
        "text-1",
    ]);
    expect(labels[5]!.filter((label) => label === "upscaling")).toEqual(["upscaling"]);
});
test("auto pixel ratio prepares upscaling even before a display change", () => {
    subjects();
    expect(labels[3]!.filter((label) => label === "upscaling")).toEqual(["upscaling"]);
});
test("a later camera with another sample count warns instead of eagerly compiling the absent line variant", () => {
    const world = subjects()[1].world;
    const camera = world.create();
    world.add(camera, Transform);
    world.add(camera, Camera, { antialias: 1 });
    attachTexture(world, camera, { width: 8, height: 8 });
    expectWarning(
        /Pipeline "lines-4" owned by "CorePipeline\/1" created after the precompile drain\./,
    );
    expectWarning(
        /Pipeline "text-4" owned by "CorePipeline\/1" created after the precompile drain\./,
    );
    world.step(0);
});
