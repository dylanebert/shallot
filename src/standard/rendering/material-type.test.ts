import { expect, expectTypeOf, test } from "bun:test";
import * as d from "typegpu/data";
import { materialLayout, materialType } from "./material-type";

test("a material without owned bindings needs only its parameter schema", () => {
    const Parameters = d.struct({ tint: d.vec4f, scale: d.f32 });
    const material = materialType({
        name: "NoOwnedBindings",
        parameters: Parameters,
        defaults: { tint: d.vec4f(1), scale: 2 },
        vertex: (input) => {
            "use gpu";
            expectTypeOf(input.parameters).toEqualTypeOf<d.Infer<typeof Parameters>>();
            return { world: input.world, worldNormal: input.worldNormal };
        },
        fragment: (ctx) => {
            "use gpu";
            expectTypeOf(ctx.parameters).toEqualTypeOf<d.Infer<typeof Parameters>>();
            return d.vec4f(ctx.parameters.tint);
        },
    });
    expect(material.parameters).toBe(Parameters);
    expect(material.layout.parameters).toBe(Parameters);
    expect(material.layout.attributes).toEqual({});
});

test("texture and sampler options declare their WGSL resource types", () => {
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        plain: { type: "texture" },
        layers: { type: "texture", dimension: "2d-array" },
        depth: { type: "texture", sampleType: "depth" },
        linear: { type: "sampler" },
        less: { type: "sampler", samplerType: "comparison" },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["plain"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["layers"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2dArray<d.F32>>>();
    expectTypeOf<Resources["depth"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureDepth2d>>();
    expectTypeOf<Resources["linear"]>().toEqualTypeOf<d.InferGPU<d.WgslSampler>>();
    expectTypeOf<Resources["less"]>().toEqualTypeOf<d.InferGPU<d.WgslComparisonSampler>>();
    const { plain, layers, depth, linear, less } = layout.entries;
    expect([plain.texture.type, layers.texture.type, depth.texture.type]).toEqual([
        "texture_2d",
        "texture_2d_array",
        "texture_depth_2d",
    ]);
    expect([linear.sampler, less.sampler]).toEqual(["filtering", "comparison"]);
});

test("a union of texture or sampler options declares the union of their WGSL resource types", () => {
    const sampleType = "float" as "float" | "depth";
    const samplerType = "filtering" as "filtering" | "comparison";
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        either: { type: "texture", sampleType },
        sampler: { type: "sampler", samplerType },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["either"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTextureDepth2d>
    >();
    expectTypeOf<Resources["sampler"]>().toEqualTypeOf<
        d.InferGPU<d.WgslSampler> | d.InferGPU<d.WgslComparisonSampler>
    >();
    expect(layout.entries.either.texture.type).toBe("texture_2d");
});

test("an optional texture or sampler option also declares its default WGSL resource type", () => {
    const layers: { type: "texture"; dimension?: "2d-array" } = { type: "texture" };
    const less: { type: "sampler"; samplerType?: "comparison" } = { type: "sampler" };
    const layout = materialLayout(d.struct({ unused: d.f32 }), { layers, less });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["layers"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture2dArray<d.F32>>
    >();
    expectTypeOf<Resources["less"]>().toEqualTypeOf<
        d.InferGPU<d.WgslSampler> | d.InferGPU<d.WgslComparisonSampler>
    >();
    expect([layout.entries.layers.texture.type, layout.entries.less.sampler]).toEqual([
        "texture_2d",
        "filtering",
    ]);
});

test("an option that may be undefined also declares its default WGSL resource type", () => {
    const layers: { type: "texture"; dimension: "2d-array" | undefined } = {
        type: "texture",
        dimension: undefined,
    };
    const less: { type: "sampler"; samplerType: "comparison" | undefined } = {
        type: "sampler",
        samplerType: undefined,
    };
    const layout = materialLayout(d.struct({ unused: d.f32 }), { layers, less });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["layers"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture2dArray<d.F32>>
    >();
    expectTypeOf<Resources["less"]>().toEqualTypeOf<
        d.InferGPU<d.WgslSampler> | d.InferGPU<d.WgslComparisonSampler>
    >();
    expect([layout.entries.layers.texture.type, layout.entries.less.sampler]).toEqual([
        "texture_2d",
        "filtering",
    ]);
});
