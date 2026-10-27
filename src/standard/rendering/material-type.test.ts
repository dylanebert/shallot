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
