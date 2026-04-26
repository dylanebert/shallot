import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { describe, test, expect, beforeAll, beforeEach } from "bun:test";
import { compileRasterShader, compileSkyShader } from "../src/standard/raster/forward";
import type { SurfaceData } from "../src/standard/render/surface";
import { clearDefaultSurfaces, registerProperties } from "../src/standard/render/surface";

import { compilePresentShader } from "../src/standard/render/present";

describe("raster shader", () => {
    const basicSurface: SurfaceData = {};

    test("includes applyHaze function", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("fn applyHaze");
    });

    test("includes sky uniform binding", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("sky: Sky");
    });

    test("applies haze to lit color", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("applyHaze(reflectedColor, dist)");
    });

    test("computes distance from camera", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("length(input.worldPos - scene.cameraWorld[3].xyz)");
    });
});

describe("unified fragment shader", () => {
    const basicSurface: SurfaceData = {};

    test("uses input.color.a for opacity", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("surface.opacity = input.color.a");
    });

    test("outputs surface.opacity as alpha", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).toContain("surface.opacity);");
    });

    test("no WBOIT remnants", () => {
        const shader = compileRasterShader([basicSurface], false);
        expect(shader).not.toContain("fn fs_transparent");
        expect(shader).not.toContain("TransparentOutput");
        expect(shader).not.toContain("accum");
        expect(shader).not.toContain("reveal");
    });

    test("includes shadow sampling with shadows enabled", () => {
        const shader = compileRasterShader([basicSurface], true);
        expect(shader).toContain("sampleShadow");
    });
});

describe("sky shader", () => {
    test("default sky shader has no entityId", () => {
        const shader = compileSkyShader();
        expect(shader).not.toContain("entityId");
    });

    test("sky shader with entityId outputs zero", () => {
        const shader = compileSkyShader(true);
        expect(shader).toContain("output.entityId = 0u");
        expect(shader).toContain("struct FragmentOutput");
    });
});

describe("shadow sampling", () => {
    const basicSurface: SurfaceData = {};

    test("PCF shadow sampling present", () => {
        const shader = compileRasterShader([basicSurface], true);
        expect(shader).toContain("textureSampleCompareLevel");
        expect(shader).toContain("sampleShadow");
    });
});

describe("multi-surface raster compilation", () => {
    beforeEach(() => {
        clearDefaultSurfaces();
    });

    test("multiple surface variants produce dispatch functions", () => {
        const surfaces: SurfaceData[] = [
            {},
            { vertex: "pos.y += sin(pos.x);" },
            { fragment: "(*surface).baseColor = vec3(1.0, 0.0, 0.0);" },
            { vertex: "pos *= 1.1;", fragment: "(*surface).roughness = 0.2;" },
        ];
        const shader = compileRasterShader(surfaces, false);

        for (let i = 0; i < surfaces.length; i++) {
            expect(shader).toContain(`fn userVertexTransform_${i}`);
            expect(shader).toContain(`fn userFragment_${i}`);
            expect(shader).toContain(`fn applyLighting_${i}`);
            expect(shader).toContain(`case ${i}u`);
        }
    });

    test("instance field surfaces through raster pipeline", () => {
        const fields = [{ name: "originY", type: "f32" as const }];
        registerProperties(fields);
        const surfaces: SurfaceData[] = [
            {},
            {
                properties: fields,
                fragment: "(*surface).baseColor.g = inst.originY;",
            },
        ];
        const shader = compileRasterShader(surfaces, false);

        expect(shader).toContain("let inst = instanceData[eid];");
        expect(shader).toContain("inst.originY");
        expect(shader).toContain("fn dispatchLighting");
    });

    test("instance field surfaces with shadows", () => {
        const fields = [{ name: "scale", type: "f32" as const }];
        registerProperties(fields);
        const surfaces: SurfaceData[] = [
            {},
            {
                properties: fields,
                vertex: "pos *= inst.scale;",
                fragment: "(*surface).reflectivity = inst.scale;",
            },
        ];
        const shader = compileRasterShader(surfaces, true);

        expect(shader).toContain("inst.scale");
        expect(shader).toContain("sampleShadow");
        expect(shader).toContain("fn dispatchLighting");
    });

    test("mixed surfaces with and without shadows", () => {
        const surfaces: SurfaceData[] = [
            {},
            { fragment: "(*surface).baseColor = vec3(0.5);" },
            { vertex: "pos.y += 1.0;", fragment: "(*surface).roughness = 0.8;" },
        ];

        const noShadow = compileRasterShader(surfaces, false);
        const withShadow = compileRasterShader(surfaces, true);

        for (let i = 0; i < surfaces.length; i++) {
            expect(noShadow).toContain(`case ${i}u`);
            expect(withShadow).toContain(`case ${i}u`);
        }

        expect(noShadow).toContain("shadowFactor = 1.0");
        expect(withShadow).toContain("sampleShadow");
    });
});

describe("GPU pipeline compilation", () => {
    const skipReason = shouldSkipGPU();
    let device: GPUDevice;

    beforeAll(async () => {
        if (skipReason) return;
        const ctx = await initGPU();
        device = ctx.device;
    });

    test.skipIf(!!skipReason)("raster shader compiles to opaque pipeline", async () => {
        const code = compileRasterShader([{}], false);
        const module = device.createShaderModule({ code });
        const pipeline = await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: "rgba8unorm" }, { format: "r32uint" }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
            primitive: { topology: "triangle-list", cullMode: "back" },
        });
        expect(pipeline).toBeDefined();
    });

    test.skipIf(!!skipReason)(
        "raster shader with shadows compiles to opaque pipeline",
        async () => {
            const code = compileRasterShader([{}], true);
            const module = device.createShaderModule({ code });
            const pipeline = await device.createRenderPipelineAsync({
                layout: "auto",
                vertex: { module, entryPoint: "vs" },
                fragment: {
                    module,
                    entryPoint: "fs",
                    targets: [{ format: "rgba8unorm" }, { format: "r32uint" }],
                },
                depthStencil: {
                    format: "depth24plus",
                    depthWriteEnabled: true,
                    depthCompare: "less",
                },
                primitive: { topology: "triangle-list", cullMode: "back" },
            });
            expect(pipeline).toBeDefined();
        },
    );

    test.skipIf(!!skipReason)("raster shader compiles to transparent pipeline", async () => {
        const code = compileRasterShader([{}], false, true, true);
        const module = device.createShaderModule({ code });
        const pipeline = await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [
                    {
                        format: "rgba8unorm",
                        blend: {
                            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
                        },
                    },
                    { format: "r32uint", writeMask: 0 },
                ],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "less-equal",
            },
            primitive: { topology: "triangle-list", cullMode: "none" },
        });
        expect(pipeline).toBeDefined();
    });

    test.skipIf(!!skipReason)("sky shader compiles to pipeline with entityId", async () => {
        const code = compileSkyShader(true);
        const module = device.createShaderModule({ code });
        const pipeline = await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: "rgba8unorm" }, { format: "r32uint" }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
            primitive: { topology: "triangle-list" },
        });
        expect(pipeline).toBeDefined();
    });

    test.skipIf(!!skipReason)("sky shader compiles to pipeline without entityId", async () => {
        const code = compileSkyShader();
        const module = device.createShaderModule({ code });
        const pipeline = await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: "rgba8unorm" }],
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
            primitive: { topology: "triangle-list" },
        });
        expect(pipeline).toBeDefined();
    });
});

describe("present shader", () => {
    test("contains per-pixel effect functions", () => {
        const shader = compilePresentShader();
        expect(shader).toContain("fn applyFXAA");
        expect(shader).toContain("fn applyVignette");
        expect(shader).toContain("fn aces");
        expect(shader).toContain("fn applyPosterize");
        expect(shader).toContain("fn applyDither");
        expect(shader).toContain("fn linearToSrgb");
    });

    test("includes Scene struct uniform", () => {
        const shader = compilePresentShader();
        expect(shader).toContain("scene: Scene");
        expect(shader).toContain("scene.fxaaEnabled");
        expect(shader).toContain("scene.tonemapMode");
        expect(shader).toContain("scene.exposure");
        expect(shader).toContain("scene.vignetteStrength");
        expect(shader).toContain("scene.posterizeBands");
        expect(shader).toContain("scene.ditherStrength");
    });

    test("applies effects in gamma-correct order", () => {
        const shader = compilePresentShader();
        const tonemap = shader.indexOf("color = applyTonemap(color);");
        const srgb = shader.indexOf("color = linearToSrgb(saturate(color));");
        const dither = shader.indexOf("color = applyDither(");
        const posterize = shader.indexOf("color = applyPosterize(color);");
        const vignette = shader.indexOf("color = applyVignette(");
        expect(tonemap).toBeGreaterThan(-1);
        expect(srgb).toBeGreaterThan(tonemap);
        expect(dither).toBeGreaterThan(srgb);
        expect(posterize).toBeGreaterThan(dither);
        expect(vignette).toBeGreaterThan(posterize);
    });
});

describe("present shader GPU compile", () => {
    const skipReason = shouldSkipGPU();

    test.skipIf(!!skipReason)("compiles to a working render pipeline", async () => {
        const ctx = await initGPU();
        const device = ctx.device;
        const code = compilePresentShader();
        const module = device.createShaderModule({ code });
        const pipeline = await device.createRenderPipelineAsync({
            layout: "auto",
            vertex: { module, entryPoint: "vs" },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: "triangle-list" },
        });
        expect(pipeline).toBeDefined();
    });
});
