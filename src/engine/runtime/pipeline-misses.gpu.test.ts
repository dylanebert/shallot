import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { expectWarning } from "../../../scripts/test-console";
import type { World } from "../ecs";
import { precompile, precompileState, withPipelineOwner } from "./gpu";

setDefaultTimeout(TIMEOUT.gpu);

function pipeline(world: World, label: string) {
    const module = world.gpu.device.createShaderModule({
        code: "@compute @workgroup_size(1) fn main() {}",
    });
    return world.gpu.device.createComputePipeline({
        label,
        layout: world.gpu.device.createPipelineLayout({ bindGroupLayouts: [] }),
        compute: { module },
    });
}

const warmWarnings: unknown[][] = [];
let asyncShader: GPUShaderModule;
let ownedShader: GPUShaderModule;
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            {
                name: "Control",
                gpu: {},
                systems: [
                    {
                        name: "late",
                        group: "draw",
                        update(world) {
                            pipeline(world, "scheduled-control");
                        },
                    },
                ],
                warm(world) {
                    const warn = spyOn(console, "warn").mockImplementation((...args) => {
                        warmWarnings.push(args);
                    });
                    try {
                        ownedShader = withPipelineOwner(world.gpu.device, "ShaderOwner", () =>
                            world.gpu.device.createShaderModule({
                                code: "@compute @workgroup_size(1) fn main() {}",
                            }),
                        );
                        const prepared = pipeline(world, "warm-control");
                        precompile(world, "warm-control", () => [prepared]);
                    } finally {
                        warn.mockRestore();
                    }
                },
            },
        ],
    },
    {
        defaults: false,
        plugins: [
            {
                name: "AsyncOwner",
                gpu: {},
                async warm(world) {
                    await Promise.resolve();
                    asyncShader = world.gpu.device.createShaderModule({
                        code: "@compute @workgroup_size(1) fn main() {}",
                    });
                },
                systems: [
                    {
                        name: "late",
                        group: "draw",
                        update(world) {
                            world.gpu.device.createComputePipeline({
                                label: "async-owner-control",
                                layout: world.gpu.device.createPipelineLayout({
                                    bindGroupLayouts: [],
                                }),
                                compute: { module: asyncShader },
                            });
                        },
                    },
                ],
            },
        ],
    },
]);

test("async warm shader creation retains the late executing system and plugin as owner", () => {
    const world = subjects()[1].world;
    expectWarning(
        /Pipeline "async-owner-control" owned by "AsyncOwner\/late" created after the precompile drain\./,
    );
    world.step(0);
    expect(world.currentSystem).toBeUndefined();
});

test("outside a system a late pipeline falls back to its shader owner, not another callback scope", () => {
    const world = subjects()[0].world;
    expectWarning(
        /Pipeline "fallback-owner-control" owned by "ShaderOwner" created after the precompile drain\./,
    );
    withPipelineOwner(world.gpu.device, "OtherCallback", () =>
        world.gpu.device.createComputePipeline({
            label: "fallback-owner-control",
            layout: world.gpu.device.createPipelineLayout({ bindGroupLayouts: [] }),
            compute: { module: ownedShader },
        }),
    );
});

test("a pipeline created in warm and registered with precompile does not warn", () => {
    const state = subjects()[0].world.resource(precompileState);
    expect(state.drained).toBe(true);
    expect(state.labels.has("warm-control")).toBe(true);
    expect(warmWarnings).toEqual([]);
});

test("a scheduled late pipeline names its system owner without coupling the scheduler to GPU state", () => {
    const world = subjects()[0].world;
    expectWarning(
        /Pipeline "scheduled-control" owned by "Control\/late" created after the precompile drain\./,
    );
    world.step(0);
    expect(world.currentSystem).toBeUndefined();
});

test("each pipeline created after the drain warns once with its name, owner and rule", () => {
    const world = subjects()[0].world;
    expectWarning(
        /Pipeline "late-control" owned by "LateOwner" created after the precompile drain\. Create it in initialize or warm and register it with precompile\./,
    );
    const warn = spyOn(console, "warn");
    try {
        withPipelineOwner(world.gpu.device, "LateOwner", () => pipeline(world, "late-control"));
        expect(warn).toHaveBeenCalledTimes(1);
    } finally {
        warn.mockRestore();
    }
});
