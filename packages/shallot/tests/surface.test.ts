import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { describe, test, expect, beforeAll, beforeEach, spyOn } from "bun:test";
import {
    surface,
    surfaceRegistry,
    clearDefaultSurfaces,
    SurfaceType,
    compileSurface,
    registerProperties,
    instanceLayout,
    instanceStride,
    instanceStructWGSL,
    instanceBindingWGSL,
    instancePackingShader,
    hasProperties,
    propertyCount,
} from "../src/standard/render/surface";
import type { SurfaceData } from "../src/standard/render/surface";
import { compileSurfaceBlock } from "../src/standard/render/surface/compile";
import type { PipelineVariantConfig } from "../src/standard/render/surface/compile";
import { validateSurface } from "../src/standard/render/surface/validate";

describe("surface registry", () => {
    beforeEach(() => {
        clearDefaultSurfaces();
    });

    describe("built-in surfaces", () => {
        test("SurfaceType.Default is 0", () => {
            expect(SurfaceType.Default).toBe(0);
        });

        test("SurfaceType.Normals is 1", () => {
            expect(SurfaceType.Normals).toBe(1);
        });

        test("SurfaceType.Depth is 2", () => {
            expect(SurfaceType.Depth).toBe(2);
        });

        test("SurfaceType.Albedo is 3", () => {
            expect(SurfaceType.Albedo).toBe(3);
        });

        test("surfaceRegistry.get returns surface for SurfaceType.Default", () => {
            const data = surfaceRegistry.get(SurfaceType.Default);
            expect(data).toBeDefined();
        });

        test("surfaceRegistry.get returns surface for SurfaceType.Normals", () => {
            const data = surfaceRegistry.get(SurfaceType.Normals);
            expect(data).toBeDefined();
            expect(data!.fragment).toContain("worldNormal");
        });

        test("surfaceRegistry.get returns surface for SurfaceType.Depth", () => {
            const data = surfaceRegistry.get(SurfaceType.Depth);
            expect(data).toBeDefined();
            expect(data!.fragment).toContain("depth");
        });
    });

    describe("surface()", () => {
        test("returns sequential IDs starting at 4", () => {
            const id1 = surface({ fragment: "(*surface).baseColor = vec3(1.0);" });
            const id2 = surface({ fragment: "(*surface).baseColor = vec3(0.0);" });

            expect(id1).toBe(4);
            expect(id2).toBe(5);
        });

        test("registered surface is retrievable via surfaceRegistry.get", () => {
            const customData = { fragment: "(*surface).baseColor = vec3(1.0, 0.0, 0.0);" };
            const id = surface(customData);

            const retrieved = surfaceRegistry.get(id);
            expect(retrieved).toBe(customData);
        });

        test("surface with vertex transform is stored", () => {
            const customData = {
                vertex: "pos = localPos + normal * 0.1;",
                fragment: "(*surface).baseColor = vec3(1.0);",
            };
            const id = surface(customData);

            const retrieved = surfaceRegistry.get(id);
            expect(retrieved!.vertex).toBe(customData.vertex);
        });
    });

    describe("clearDefaultSurfaces()", () => {
        test("resets to built-ins only", () => {
            surface({ fragment: "(*surface).baseColor = vec3(1.0);" });
            surface({ fragment: "(*surface).baseColor = vec3(0.0);" });

            clearDefaultSurfaces();

            const id = surface({ fragment: "(*surface).baseColor = vec3(0.5);" });
            expect(id).toBe(4);
        });

        test("built-in surfaces remain after clear", () => {
            clearDefaultSurfaces();

            expect(surfaceRegistry.get(SurfaceType.Default)).toBeDefined();
            expect(surfaceRegistry.get(SurfaceType.Normals)).toBeDefined();
            expect(surfaceRegistry.get(SurfaceType.Depth)).toBeDefined();
            expect(surfaceRegistry.get(SurfaceType.Albedo)).toBeDefined();
        });
    });

    describe("named surfaces", () => {
        test("surface() with name registers retrievable name", () => {
            const id = surface({ fragment: "(*surface).baseColor = vec3(1.0);" }, "bark");
            expect(surfaceRegistry.getByName("bark")).toBe(id);
        });

        test("built-in names resolve to correct IDs", () => {
            expect(surfaceRegistry.getByName("default")).toBe(0);
            expect(surfaceRegistry.getByName("normals")).toBe(1);
            expect(surfaceRegistry.getByName("depth")).toBe(2);
            expect(surfaceRegistry.getByName("albedo")).toBe(3);
        });

        test("clearDefaultSurfaces clears names; built-ins available after clear", () => {
            surface({ fragment: "" }, "bark");
            clearDefaultSurfaces();

            expect(surfaceRegistry.getByName("bark")).toBeUndefined();
            expect(surfaceRegistry.getByName("default")).toBe(0);
        });

        test("unknown name returns undefined", () => {
            expect(surfaceRegistry.getByName("nonexistent")).toBeUndefined();
        });
    });

    describe("compileSurface()", () => {
        test("produces WGSL with standard structs", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("struct VertexInput");
            expect(code).toContain("struct VertexOutput");
            expect(code).toContain("struct Scene");
            expect(code).toContain("struct SurfaceData");
        });

        test("produces WGSL with standard bindings", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("@group(0) @binding(0)");
            expect(code).toContain("@group(0) @binding(1)");
            expect(code).toContain("@group(0) @binding(2)");
            expect(code).toContain("entityIds");
            expect(code).toContain("matrices");
            expect(code).toContain("data");
            expect(code).toContain("struct Data");
        });

        test("includes user fragment code", () => {
            const data = { fragment: "(*surface).baseColor = vec3(0.5, 0.2, 0.8);" };
            const code = compileSurface(data);

            expect(code).toContain("vec3(0.5, 0.2, 0.8)");
        });

        test("uses identity vertex transform when not provided", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("return VertexTransformResult(localPos, meshUv);");
        });

        test("includes custom vertex transform when provided", () => {
            const data = {
                vertex: "pos = localPos + normal * sin(localPos.x);",
            };
            const code = compileSurface(data);

            expect(code).toContain("sin(localPos.x)");
        });

        test("produces WGSL with entry points", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("@vertex");
            expect(code).toContain("fn vs(");
            expect(code).toContain("@fragment");
            expect(code).toContain("fn fs(");
        });

        test("applies lighting uniformly after user fragment", () => {
            const data = { fragment: "(*surface).baseColor = vec3(1.0, 0.0, 0.0);" };
            const code = compileSurface(data);

            expect(code).toContain("userFragment");
            expect(code).toContain("NdotL");
            expect(code).toContain("litColor");
        });

        test("uses per-entity PBR data", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("d.pbr");
            expect(code).toContain("d.emission");
            expect(code).toContain("surface.roughness = d.pbr.x");
            expect(code).toContain("surface.reflectivity = d.pbr.y");
        });

        test("applies PBR lighting", () => {
            const data = {};
            const code = compileSurface(data);

            expect(code).toContain("NdotL");
            expect(code).toContain("litColor");
            expect(code).toContain("ambient");
            expect(code).toContain("diffuse");
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

        test.skipIf(!!skipReason)("default surface compiles to render pipeline", async () => {
            const code = compileSurface({});
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

        test.skipIf(!!skipReason)("surface with custom vertex compiles to pipeline", async () => {
            const code = compileSurface({ vertex: "pos = localPos + normal * sin(localPos.x);" });
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

        test.skipIf(!!skipReason)("surface with custom fragment compiles to pipeline", async () => {
            const code = compileSurface({
                fragment: "(*surface).baseColor = vec3(1.0, 0.0, 0.0);",
            });
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
    });

    describe("compileSurfaceBlock()", () => {
        test("generates vertex and fragment dispatch without lighting config", () => {
            const block = compileSurfaceBlock([{}]);

            expect(block).toContain("fn userVertexTransform_0");
            expect(block).toContain("fn userFragment_0");
            expect(block).toContain("fn dispatchVertexTransform");
            expect(block).toContain("fn dispatchFragment");
            expect(block).not.toContain("fn dispatchLighting");
            expect(block).not.toContain("fn applyLighting_0");
        });

        test("generates lighting dispatch with config", () => {
            const config: PipelineVariantConfig = {
                lighting: {
                    params: "rayDir: vec3<f32>, shadowFactor: f32",
                    body: () => "return vec3(1.0);",
                },
            };
            const block = compileSurfaceBlock([{}], config);

            expect(block).toContain("fn applyLighting_0");
            expect(block).toContain("fn dispatchLighting");
            expect(block).toContain("rayDir: vec3<f32>, shadowFactor: f32");
        });

        test("generates correct number of surface variants", () => {
            const surfaces = [
                {},
                { fragment: "(*surface).baseColor = vec3(1.0);" },
                { vertex: "pos.y += 1.0;" },
            ];
            const block = compileSurfaceBlock(surfaces);

            expect(block).toContain("fn userVertexTransform_0");
            expect(block).toContain("fn userVertexTransform_1");
            expect(block).toContain("fn userVertexTransform_2");
            expect(block).toContain("case 0u");
            expect(block).toContain("case 1u");
            expect(block).toContain("case 2u");
        });

        test("lighting dispatch forwards correct param names", () => {
            const config: PipelineVariantConfig = {
                lighting: {
                    params: "shadowFactor: f32, fragCoord: vec2<f32>, viewZ: f32",
                    body: () => "return vec3(0.0);",
                },
            };
            const block = compileSurfaceBlock([{}], config);

            expect(block).toContain("applyLighting_0(surface, shadowFactor, fragCoord, viewZ)");
        });

        test("includes user fragment code in variant", () => {
            const block = compileSurfaceBlock([{ fragment: "(*surface).baseColor = vec3(0.5);" }]);
            expect(block).toContain("vec3(0.5)");
        });

        test("includes user vertex code in variant", () => {
            const block = compileSurfaceBlock([{ vertex: "pos.y += sin(pos.x);" }]);
            expect(block).toContain("sin(pos.x)");
        });
    });

    describe("validateSurface()", () => {
        test("warns on entityIds in fragment", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({ fragment: "let e = entityIds[eid];" });
            validateSurface({ fragment: "let m = matrices[eid];" });
            expect(spy).toHaveBeenCalledTimes(1);
            expect(spy).toHaveBeenCalledWith(expect.stringContaining("not available in RT"));
            spy.mockRestore();
        });

        test("warns on discard in fragment", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({ fragment: "discard;" });
            expect(spy).toHaveBeenCalledWith(expect.stringContaining("surface.opacity = 0.0"));
            spy.mockRestore();
        });

        test("warns on return in vertex snippet", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({ vertex: "return localPos;" });
            expect(spy).toHaveBeenCalledWith(expect.stringContaining("modify 'pos' variable"));
            spy.mockRestore();
        });

        test("no warnings for clean or empty surface", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({});
            validateSurface({
                vertex: "pos.y += sin(pos.x);",
                fragment: "(*surface).baseColor = vec3(1.0);",
            });
            expect(spy).not.toHaveBeenCalled();
            spy.mockRestore();
        });

        test("warns on inst. access without properties", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({ fragment: "let x = inst.originY;" });
            expect(spy).toHaveBeenCalledWith(expect.stringContaining("no properties declared"));
            spy.mockRestore();
        });

        test("no warning on inst. access with properties", () => {
            const spy = spyOn(console, "warn").mockImplementation(() => {});
            validateSurface({
                properties: [{ name: "originY", type: "f32" }],
                fragment: "let x = inst.originY;",
            });
            expect(spy).not.toHaveBeenCalled();
            spy.mockRestore();
        });
    });

    describe("instance field registry", () => {
        test("registers fields and tracks layout", () => {
            registerProperties([
                { name: "originY", type: "f32" },
                { name: "grassHeight", type: "f32" },
            ]);

            expect(hasProperties()).toBe(true);
            expect(propertyCount()).toBe(2);

            const layout = instanceLayout();
            expect(layout.get("originY")?.type).toBe("f32");
            expect(layout.get("originY")?.offset).toBe(0);
            expect(layout.get("grassHeight")?.type).toBe("f32");
            expect(layout.get("grassHeight")?.offset).toBe(4);
        });

        test("deduplicates same name + same type", () => {
            registerProperties([{ name: "originY", type: "f32" }]);
            registerProperties([{ name: "originY", type: "f32" }]);
            expect(propertyCount()).toBe(1);
        });

        test("throws on same name + different type", () => {
            registerProperties([{ name: "originY", type: "f32" }]);
            expect(() => {
                registerProperties([{ name: "originY", type: "u32" }]);
            }).toThrow("cannot re-register");
        });

        test("stride is 16-byte aligned", () => {
            registerProperties([{ name: "a", type: "f32" }]);
            expect(instanceStride()).toBe(16);

            clearDefaultSurfaces();
            registerProperties([
                { name: "a", type: "f32" },
                { name: "b", type: "f32" },
                { name: "c", type: "f32" },
                { name: "d", type: "f32" },
                { name: "e", type: "f32" },
            ]);
            expect(instanceStride()).toBe(32);
        });

        test("clearDefaultSurfaces clears instance fields", () => {
            registerProperties([{ name: "originY", type: "f32" }]);
            clearDefaultSurfaces();
            expect(hasProperties()).toBe(false);
            expect(propertyCount()).toBe(0);
        });

        test("surface() auto-registers properties", () => {
            surface({
                properties: [
                    { name: "originY", type: "f32" },
                    { name: "grassHeight", type: "f32" },
                ],
                fragment: "let t = inst.originY;",
            });
            expect(hasProperties()).toBe(true);
            expect(propertyCount()).toBe(2);
        });
    });

    describe("instance data WGSL generation", () => {
        test("generates struct with fields and padding", () => {
            registerProperties([
                { name: "originY", type: "f32" },
                { name: "grassHeight", type: "f32" },
            ]);

            const struct = instanceStructWGSL();
            expect(struct).toContain("struct InstanceData");
            expect(struct).toContain("originY: f32,");
            expect(struct).toContain("grassHeight: f32,");
            expect(struct).toContain("_pad");
        });

        test("generates binding declaration", () => {
            registerProperties([{ name: "a", type: "f32" }]);
            const binding = instanceBindingWGSL(7);
            expect(binding).toContain("@group(0) @binding(7)");
            expect(binding).toContain("instanceData: array<InstanceData>");
        });

        test("empty fields produce empty strings", () => {
            expect(instanceStructWGSL()).toBe("");
            expect(instanceBindingWGSL(7)).toBe("");
        });

        test("generates packing compute shader", () => {
            registerProperties([
                { name: "originY", type: "f32" },
                { name: "grassHeight", type: "f32" },
            ]);

            const shader = instancePackingShader();
            expect(shader).toContain("struct InstanceData");
            expect(shader).toContain("@compute @workgroup_size(64)");
            expect(shader).toContain("d.originY = bitcast<f32>(source[0u * count + eid])");
            expect(shader).toContain("d.grassHeight = bitcast<f32>(source[1u * count + eid])");
            expect(shader).toContain("instanceData[eid] = d;");
        });

        test("empty fields produce empty packing shader", () => {
            expect(instancePackingShader()).toBe("");
        });

        test("u32 field reads directly, others bitcast", () => {
            registerProperties([
                { name: "flags", type: "u32" },
                { name: "height", type: "f32" },
            ]);

            const shader = instancePackingShader();
            expect(shader).toContain("source: array<u32>");
            expect(shader).toContain("d.flags = source[0u * count + eid]");
            expect(shader).toContain("d.height = bitcast<f32>(source[1u * count + eid])");

            const struct = instanceStructWGSL();
            expect(struct).toContain("flags: u32,");
            expect(struct).toContain("height: f32,");
        });
    });

    describe("instance data preamble injection", () => {
        test("injects inst preamble for surface with properties", () => {
            registerProperties([{ name: "originY", type: "f32" }]);
            const block = compileSurfaceBlock([
                {
                    properties: [{ name: "originY", type: "f32" }],
                    fragment: "let y = inst.originY;",
                },
            ]);
            expect(block).toContain("let inst = instanceData[eid];");
            expect(block).toContain("inst.originY");
        });

        test("no preamble for surface without properties", () => {
            const block = compileSurfaceBlock([{ fragment: "(*surface).baseColor = vec3(1.0);" }]);
            expect(block).not.toContain("instanceData");
        });
    });

    describe("multi-surface compilation across pipelines", () => {
        const SurfaceVariants: Record<string, SurfaceData> = {
            default: {},
            "vertex-only": { vertex: "pos.y += sin(pos.x);" },
            "fragment-only": { fragment: "(*surface).baseColor = vec3(1.0, 0.0, 0.0);" },
            "vertex+fragment": {
                vertex: "pos *= 1.1;",
                fragment: "(*surface).roughness = 0.2;",
            },
            "with-properties": {
                properties: [{ name: "originY", type: "f32" }],
                fragment: "(*surface).baseColor.g = inst.originY;",
            },
            "properties+vertex": {
                properties: [{ name: "scale", type: "f32" }],
                vertex: "pos *= inst.scale;",
                fragment: "(*surface).reflectivity = inst.scale;",
            },
        };

        const PipelineConfigs: Record<string, PipelineVariantConfig | undefined> = {
            raster: {
                lighting: {
                    params: "shadowFactor: f32, fragCoord: vec2<f32>, viewZ: f32",
                    body: () => "return surface.baseColor;",
                },
            },
            raytracing: {
                lighting: {
                    params: "rayDir: vec3<f32>, shadowFactor: f32",
                    body: () => "return surface.baseColor * shadowFactor;",
                },
            },
        };

        // Pairwise: each surface variant × each pipeline config
        for (const [surfaceName, surfaceData] of Object.entries(SurfaceVariants)) {
            for (const [pipelineName, config] of Object.entries(PipelineConfigs)) {
                describe(`${surfaceName} × ${pipelineName}`, () => {
                    beforeEach(() => {
                        clearDefaultSurfaces();
                        if (surfaceData.properties) {
                            registerProperties(surfaceData.properties);
                        }
                    });

                    test("generates dispatch functions", () => {
                        const block = compileSurfaceBlock([surfaceData], config);
                        expect(block).toContain("fn dispatchVertexTransform");
                        expect(block).toContain("fn dispatchFragment");
                        if (config?.lighting) {
                            expect(block).toContain("fn dispatchLighting");
                            expect(block).toContain("fn applyLighting_0");
                        } else {
                            expect(block).not.toContain("fn dispatchLighting");
                        }
                    });

                    test("includes user code", () => {
                        const block = compileSurfaceBlock([surfaceData], config);
                        if (surfaceData.vertex) {
                            expect(block).toContain(surfaceData.vertex);
                        }
                        if (surfaceData.fragment) {
                            expect(block).toContain(surfaceData.fragment);
                        }
                    });

                    if (surfaceData.properties) {
                        test("injects instance preamble", () => {
                            const block = compileSurfaceBlock([surfaceData], config);
                            expect(block).toContain("let inst = instanceData[eid];");
                        });
                    }
                });
            }
        }

        // Multi-surface: all variants together through each pipeline
        for (const [pipelineName, config] of Object.entries(PipelineConfigs)) {
            test(`all variants combined × ${pipelineName}`, () => {
                clearDefaultSurfaces();
                const allFields = Object.values(SurfaceVariants).flatMap((s) => s.properties ?? []);
                if (allFields.length > 0) registerProperties(allFields);

                const surfaces = Object.values(SurfaceVariants);
                const block = compileSurfaceBlock(surfaces, config);

                for (let i = 0; i < surfaces.length; i++) {
                    expect(block).toContain(`fn userVertexTransform_${i}`);
                    expect(block).toContain(`fn userFragment_${i}`);
                    expect(block).toContain(`case ${i}u`);
                    if (config?.lighting) {
                        expect(block).toContain(`fn applyLighting_${i}`);
                    }
                }
            });
        }

        describe("GPU compilation", () => {
            const skipReason = shouldSkipGPU();
            let device: GPUDevice;

            beforeAll(async () => {
                if (skipReason) return;
                const ctx = await initGPU();
                device = ctx.device;
            });

            const GpuSurfaces: [string, SurfaceData][] = [
                ["vertex-only", { vertex: "pos.y += 0.5;" }],
                ["fragment-only", { fragment: "(*surface).baseColor = vec3(0.5);" }],
                [
                    "vertex+fragment",
                    { vertex: "pos *= 1.1;", fragment: "(*surface).roughness = 0.1;" },
                ],
            ];

            for (const [label, surfaceData] of GpuSurfaces) {
                test.skipIf(!!skipReason)(`${label} compiles via compileSurface`, async () => {
                    clearDefaultSurfaces();
                    const code = compileSurface(surfaceData);
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
            }
        });
    });
});
