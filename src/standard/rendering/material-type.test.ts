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
        unfiltered: { type: "texture", filterable: false },
        unfilteredLayers: { type: "texture", dimension: "2d-array", filterable: false },
        filtered: { type: "texture", filterable: true },
        nearest: { type: "sampler", samplerType: "non-filtering" },
        signed: { type: "texture", sampleType: "sint" },
        unsigned: { type: "texture", sampleType: "uint" },
        signedLayers: { type: "texture", dimension: "2d-array", sampleType: "sint" },
        unsignedLayers: { type: "texture", dimension: "2d-array", sampleType: "uint" },
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
    expectTypeOf<Resources["unfiltered"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["unfilteredLayers"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2dArray<d.F32>>
    >();
    expectTypeOf<Resources["filtered"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["nearest"]>().toEqualTypeOf<d.InferGPU<d.WgslSampler>>();
    const { unfiltered, unfilteredLayers, filtered, nearest } = layout.entries;
    expect([unfiltered.texture.type, unfilteredLayers.texture.type, filtered.texture.type]).toEqual(
        ["texture_2d", "texture_2d_array", "texture_2d"],
    );
    expect([unfiltered.sampleType, unfilteredLayers.sampleType]).toEqual([
        "unfilterable-float",
        "unfilterable-float",
    ]);
    expect([plain, layers, filtered].some((entry) => "sampleType" in entry)).toBe(false);
    expect(nearest.sampler).toBe("non-filtering");
    expectTypeOf<Resources["signed"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.I32>>>();
    expectTypeOf<Resources["unsigned"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.U32>>>();
    expectTypeOf<Resources["signedLayers"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2dArray<d.I32>>
    >();
    expectTypeOf<Resources["unsignedLayers"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2dArray<d.U32>>
    >();
    const integers = [
        layout.entries.signed,
        layout.entries.unsigned,
        layout.entries.signedLayers,
        layout.entries.unsignedLayers,
    ];
    expect(integers.map(({ texture }) => texture.type)).toEqual([
        "texture_2d",
        "texture_2d",
        "texture_2d_array",
        "texture_2d_array",
    ]);
    expect(integers.map(({ texture }) => texture.sampleType.type)).toEqual([
        "i32",
        "u32",
        "i32",
        "u32",
    ]);
    expect(integers.some((entry) => "sampleType" in entry)).toBe(false);
});

test("a union of texture or sampler options declares the union of their WGSL resource types", () => {
    const sampleType = "float" as "float" | "depth";
    const samplerType = "filtering" as "filtering" | "comparison";
    const filterable = false as boolean;
    const nearestOrLess = "non-filtering" as "non-filtering" | "comparison";
    const integer = "uint" as "float" | "sint" | "uint";
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        either: { type: "texture", sampleType },
        sampler: { type: "sampler", samplerType },
        maybeFiltered: { type: "texture", filterable },
        nearestOrLess: { type: "sampler", samplerType: nearestOrLess },
        integer: { type: "texture", sampleType: integer },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["either"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTextureDepth2d>
    >();
    expectTypeOf<Resources["sampler"]>().toEqualTypeOf<
        d.InferGPU<d.WgslSampler> | d.InferGPU<d.WgslComparisonSampler>
    >();
    expect(layout.entries.either.texture.type).toBe("texture_2d");
    expectTypeOf<Resources["maybeFiltered"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["nearestOrLess"]>().toEqualTypeOf<
        d.InferGPU<d.WgslSampler> | d.InferGPU<d.WgslComparisonSampler>
    >();
    expectTypeOf(layout.entries.maybeFiltered.sampleType).toEqualTypeOf<
        "unfilterable-float" | undefined
    >();
    expect([layout.entries.maybeFiltered.sampleType, layout.entries.nearestOrLess.sampler]).toEqual(
        ["unfilterable-float", "non-filtering"],
    );
    expectTypeOf<Resources["integer"]>().toEqualTypeOf<
        | d.InferGPU<d.WgslTexture2d<d.F32>>
        | d.InferGPU<d.WgslTexture2d<d.I32>>
        | d.InferGPU<d.WgslTexture2d<d.U32>>
    >();
    expect(layout.entries.integer.texture.sampleType.type).toBe("u32");
});

test("an optional texture or sampler option also declares its default WGSL resource type", () => {
    const layers: { type: "texture"; dimension?: "2d-array" } = { type: "texture" };
    const less: { type: "sampler"; samplerType?: "comparison" } = { type: "sampler" };
    const unfiltered: { type: "texture"; filterable?: false } = { type: "texture" };
    const nearest: { type: "sampler"; samplerType?: "non-filtering" } = { type: "sampler" };
    const unsigned: { type: "texture"; sampleType?: "uint" } = { type: "texture" };
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        layers,
        less,
        unfiltered,
        nearest,
        unsigned,
    });
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
    expectTypeOf<Resources["unfiltered"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["nearest"]>().toEqualTypeOf<d.InferGPU<d.WgslSampler>>();
    expectTypeOf(layout.entries.unfiltered.sampleType).toEqualTypeOf<
        "unfilterable-float" | undefined
    >();
    expectTypeOf(layout.entries.nearest.sampler).toEqualTypeOf<"filtering" | "non-filtering">();
    expect([layout.entries.unfiltered.sampleType, layout.entries.nearest.sampler]).toEqual([
        undefined,
        "filtering",
    ]);
    expectTypeOf<Resources["unsigned"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture2d<d.U32>>
    >();
    expect(layout.entries.unsigned.texture.sampleType.type).toBe("f32");
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
    const unfiltered: { type: "texture"; filterable: false | undefined } = {
        type: "texture",
        filterable: undefined,
    };
    const nearest: { type: "sampler"; samplerType: "non-filtering" | undefined } = {
        type: "sampler",
        samplerType: undefined,
    };
    const unsigned: { type: "texture"; sampleType: "uint" | undefined } = {
        type: "texture",
        sampleType: undefined,
    };
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        layers,
        less,
        unfiltered,
        nearest,
        unsigned,
    });
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
    expectTypeOf<Resources["unfiltered"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    expectTypeOf<Resources["nearest"]>().toEqualTypeOf<d.InferGPU<d.WgslSampler>>();
    expectTypeOf(layout.entries.unfiltered.sampleType).toEqualTypeOf<
        "unfilterable-float" | undefined
    >();
    expectTypeOf(layout.entries.nearest.sampler).toEqualTypeOf<"filtering" | "non-filtering">();
    expect([layout.entries.unfiltered.sampleType, layout.entries.nearest.sampler]).toEqual([
        undefined,
        "filtering",
    ]);
    expectTypeOf<Resources["unsigned"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture2d<d.U32>>
    >();
    expect(layout.entries.unsigned.texture.sampleType.type).toBe("f32");
});
