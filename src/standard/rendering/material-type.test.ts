import { expect, expectTypeOf, test } from "bun:test";
import * as d from "typegpu/data";
import { type MaterialBinding, materialLayout, materialType } from "./material-type";

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

test("dimension options declare their WGSL texture types", () => {
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        line: { type: "texture", dimension: "1d" },
        volume: { type: "texture", dimension: "3d" },
        cube: { type: "texture", dimension: "cube" },
        cubes: { type: "texture", dimension: "cube-array" },
        unfilteredVolume: { type: "texture", dimension: "3d", filterable: false },
        unsignedVolume: { type: "texture", dimension: "3d", sampleType: "uint" },
        signedCube: { type: "texture", dimension: "cube", sampleType: "sint" },
        depthLayers: { type: "texture", dimension: "2d-array", sampleType: "depth" },
        depthCube: { type: "texture", dimension: "cube", sampleType: "depth" },
        depthCubes: { type: "texture", dimension: "cube-array", sampleType: "depth" },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["line"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture1d<d.F32>>>();
    expectTypeOf<Resources["volume"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture3d<d.F32>>>();
    expectTypeOf<Resources["cube"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureCube<d.F32>>>();
    expectTypeOf<Resources["cubes"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureCubeArray<d.F32>>>();
    expectTypeOf<Resources["unfilteredVolume"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture3d<d.F32>>
    >();
    expectTypeOf<Resources["unsignedVolume"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture3d<d.U32>>>();
    expectTypeOf<Resources["signedCube"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureCube<d.I32>>>();
    expectTypeOf<Resources["depthLayers"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureDepth2dArray>>();
    expectTypeOf<Resources["depthCube"]>().toEqualTypeOf<d.InferGPU<d.WgslTextureDepthCube>>();
    expectTypeOf<Resources["depthCubes"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepthCubeArray>
    >();
    const e = layout.entries;
    expect(
        [
            e.line,
            e.volume,
            e.cube,
            e.cubes,
            e.unfilteredVolume,
            e.unsignedVolume,
            e.signedCube,
            e.depthLayers,
            e.depthCube,
            e.depthCubes,
        ].map(({ texture }) => texture.type),
    ).toEqual([
        "texture_1d",
        "texture_3d",
        "texture_cube",
        "texture_cube_array",
        "texture_3d",
        "texture_3d",
        "texture_cube",
        "texture_depth_2d_array",
        "texture_depth_cube",
        "texture_depth_cube_array",
    ]);
    expect([
        e.unsignedVolume.texture.sampleType.type,
        e.signedCube.texture.sampleType.type,
    ]).toEqual(["u32", "i32"]);
    expect(e.unfilteredVolume.sampleType).toBe("unfilterable-float");
});

test("multisampled options declare multisampled WGSL texture types", () => {
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        samples: { type: "texture", multisampled: true },
        unsignedSamples: { type: "texture", sampleType: "uint", multisampled: true },
        signedSamples: { type: "texture", sampleType: "sint", multisampled: true },
        depthSamples: { type: "texture", sampleType: "depth", multisampled: true },
        single: { type: "texture", multisampled: false },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["samples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureMultisampled2d<d.F32>>
    >();
    expectTypeOf<Resources["unsignedSamples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureMultisampled2d<d.U32>>
    >();
    expectTypeOf<Resources["signedSamples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureMultisampled2d<d.I32>>
    >();
    expectTypeOf<Resources["depthSamples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepthMultisampled2d>
    >();
    expectTypeOf<Resources["single"]>().toEqualTypeOf<d.InferGPU<d.WgslTexture2d<d.F32>>>();
    const { samples, unsignedSamples, signedSamples, depthSamples, single } = layout.entries;
    expect(
        [samples, unsignedSamples, signedSamples, depthSamples, single].map(
            ({ texture }) => texture.type,
        ),
    ).toEqual([
        "texture_multisampled_2d",
        "texture_multisampled_2d",
        "texture_multisampled_2d",
        "texture_depth_multisampled_2d",
        "texture_2d",
    ]);
    // a multisampled float texture cannot be filtered
    expect(samples.sampleType).toBe("unfilterable-float");
    expect(
        [unsignedSamples, signedSamples, depthSamples, single].some((e) => "sampleType" in e),
    ).toBe(false);
});

test("storage texture options declare their WGSL storage texture types", () => {
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        read: { type: "storage-texture", format: "r32uint", access: "read" },
        write: { type: "storage-texture", format: "rgba8unorm", access: "write" },
        readWrite: {
            type: "storage-texture",
            format: "r32float",
            access: "read_write",
            dimension: "2d-array",
        },
        line: { type: "storage-texture", format: "r32sint", access: "write", dimension: "1d" },
    });
    const volumeLayout = materialLayout(d.struct({ unused: d.f32 }), {
        volume: { type: "storage-texture", format: "rgba16float", access: "read", dimension: "3d" },
    });
    type Resources = (typeof layout)["$"];
    expectTypeOf<Resources["read"]>().toEqualTypeOf<
        d.InferGPU<d.WgslStorageTexture2d<"r32uint", "read-only">>
    >();
    expectTypeOf<Resources["write"]>().toEqualTypeOf<
        d.InferGPU<d.WgslStorageTexture2d<"rgba8unorm", "write-only">>
    >();
    expectTypeOf<Resources["readWrite"]>().toEqualTypeOf<
        d.InferGPU<d.WgslStorageTexture2dArray<"r32float", "read-write">>
    >();
    expectTypeOf<Resources["line"]>().toEqualTypeOf<
        d.InferGPU<d.WgslStorageTexture1d<"r32sint", "write-only">>
    >();
    expectTypeOf<(typeof volumeLayout)["$"]["volume"]>().toEqualTypeOf<
        d.InferGPU<d.WgslStorageTexture3d<"rgba16float", "read-only">>
    >();
    const entries = [
        ...Object.values(layout.entries),
        ...Object.values(volumeLayout.entries),
    ].filter((entry) => "storageTexture" in entry);
    expect(
        entries.map(({ storageTexture: { type, format, access } }) => [type, format, access]),
    ).toEqual([
        ["texture_storage_2d", "r32uint", "read-only"],
        ["texture_storage_2d", "rgba8unorm", "write-only"],
        ["texture_storage_2d_array", "r32float", "read-write"],
        ["texture_storage_1d", "r32sint", "write-only"],
        ["texture_storage_3d", "rgba16float", "read-only"],
    ]);
    expect(entries.every(({ visibility }) => visibility.join() === "fragment")).toBe(true);
});

test("a storage texture binding with vertex visibility is refused", () => {
    const binding = {
        type: "storage-texture",
        format: "r32uint",
        access: "read",
        visibility: ["vertex", "fragment"],
    } as MaterialBinding;
    expect(() => materialLayout(d.struct({ unused: d.f32 }), { counts: binding })).toThrow(
        'material binding "counts" is a storage texture, which binds to the fragment stage only',
    );
});

test("a fifth fragment storage texture is refused", () => {
    const binding: MaterialBinding = {
        type: "storage-texture",
        format: "r32uint",
        access: "write",
    };
    const four = { a: binding, b: binding, c: binding, d: binding };
    expect(() => materialLayout(d.struct({ unused: d.f32 }), four)).not.toThrow();
    expect(() => materialLayout(d.struct({ unused: d.f32 }), { ...four, e: binding })).toThrow(
        "material layout uses 5 fragment storage textures; WebGPU's maxStorageTexturesInFragmentStage is 4",
    );
});

test("a union of texture or sampler options declares the union of their WGSL resource types", () => {
    const sampleType = "float" as "float" | "depth";
    const samplerType = "filtering" as "filtering" | "comparison";
    const filterable = false as boolean;
    const nearestOrLess = "non-filtering" as "non-filtering" | "comparison";
    const integer = "uint" as "float" | "sint" | "uint";
    const dimension = "3d" as "3d" | "cube";
    const depthDimension = "cube" as "2d-array" | "cube";
    const multisampled = true as boolean;
    const access = "read" as "read" | "write";
    const storageDimension = "3d" as "2d-array" | "3d";
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        either: { type: "texture", sampleType },
        sampler: { type: "sampler", samplerType },
        maybeFiltered: { type: "texture", filterable },
        nearestOrLess: { type: "sampler", samplerType: nearestOrLess },
        integer: { type: "texture", sampleType: integer },
        volumeOrCube: { type: "texture", dimension },
        depthLayersOrCube: { type: "texture", dimension: depthDimension, sampleType: "depth" },
        maybeSamples: { type: "texture", multisampled },
        maybeDepthSamples: { type: "texture", sampleType: "depth", multisampled },
        readOrWrite: { type: "storage-texture", format: "r32uint", access },
        layersOrVolume: {
            type: "storage-texture",
            format: "r32uint",
            access: "write",
            dimension: storageDimension,
        },
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
    expectTypeOf<Resources["volumeOrCube"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture3d<d.F32>> | d.InferGPU<d.WgslTextureCube<d.F32>>
    >();
    expectTypeOf<Resources["depthLayersOrCube"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepth2dArray> | d.InferGPU<d.WgslTextureDepthCube>
    >();
    expect([
        layout.entries.volumeOrCube.texture.type,
        layout.entries.depthLayersOrCube.texture.type,
    ]).toEqual(["texture_3d", "texture_depth_cube"]);
    expectTypeOf<Resources["maybeSamples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTextureMultisampled2d<d.F32>>
    >();
    expectTypeOf<Resources["maybeDepthSamples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepth2d> | d.InferGPU<d.WgslTextureDepthMultisampled2d>
    >();
    expect([
        layout.entries.maybeSamples.texture.type,
        layout.entries.maybeDepthSamples.texture.type,
    ]).toEqual(["texture_multisampled_2d", "texture_depth_multisampled_2d"]);
    expectTypeOf<Resources["readOrWrite"]>().toEqualTypeOf<
        | d.InferGPU<d.WgslStorageTexture2d<"r32uint", "read-only">>
        | d.InferGPU<d.WgslStorageTexture2d<"r32uint", "write-only">>
    >();
    expectTypeOf<Resources["layersOrVolume"]>().toEqualTypeOf<
        | d.InferGPU<d.WgslStorageTexture2dArray<"r32uint", "write-only">>
        | d.InferGPU<d.WgslStorageTexture3d<"r32uint", "write-only">>
    >();
    expect([
        layout.entries.readOrWrite.storageTexture.access,
        layout.entries.layersOrVolume.storageTexture.type,
    ]).toEqual(["read-only", "texture_storage_3d"]);
});

test("an optional texture or sampler option also declares its default WGSL resource type", () => {
    const layers: { type: "texture"; dimension?: "2d-array" } = { type: "texture" };
    const less: { type: "sampler"; samplerType?: "comparison" } = { type: "sampler" };
    const unfiltered: { type: "texture"; filterable?: false } = { type: "texture" };
    const nearest: { type: "sampler"; samplerType?: "non-filtering" } = { type: "sampler" };
    const unsigned: { type: "texture"; sampleType?: "uint" } = { type: "texture" };
    const volume: { type: "texture"; dimension?: "3d" } = { type: "texture" };
    const samples: { type: "texture"; multisampled?: true } = { type: "texture" };
    const storageVolume: {
        type: "storage-texture";
        format: "r32uint";
        access: "write";
        dimension?: "3d";
    } = { type: "storage-texture", format: "r32uint", access: "write" };
    const depthCube: { type: "texture"; dimension?: "cube"; sampleType: "depth" } = {
        type: "texture",
        sampleType: "depth",
    };
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        layers,
        less,
        unfiltered,
        nearest,
        unsigned,
        volume,
        depthCube,
        samples,
        storageVolume,
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
    expectTypeOf<Resources["volume"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture3d<d.F32>>
    >();
    expectTypeOf<Resources["depthCube"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepth2d> | d.InferGPU<d.WgslTextureDepthCube>
    >();
    expect([layout.entries.volume.texture.type, layout.entries.depthCube.texture.type]).toEqual([
        "texture_2d",
        "texture_depth_2d",
    ]);
    expectTypeOf<Resources["samples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTextureMultisampled2d<d.F32>>
    >();
    expect(layout.entries.samples.texture.type).toBe("texture_2d");
    expectTypeOf<Resources["storageVolume"]>().toEqualTypeOf<
        | d.InferGPU<d.WgslStorageTexture2d<"r32uint", "write-only">>
        | d.InferGPU<d.WgslStorageTexture3d<"r32uint", "write-only">>
    >();
    expect(layout.entries.storageVolume.storageTexture.type).toBe("texture_storage_2d");
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
    const volume: { type: "texture"; dimension: "3d" | undefined } = {
        type: "texture",
        dimension: undefined,
    };
    const samples: { type: "texture"; multisampled: true | undefined } = {
        type: "texture",
        multisampled: undefined,
    };
    const storageVolume: {
        type: "storage-texture";
        format: "r32uint";
        access: "write";
        dimension: "3d" | undefined;
    } = { type: "storage-texture", format: "r32uint", access: "write", dimension: undefined };
    const depthCube: { type: "texture"; dimension: "cube" | undefined; sampleType: "depth" } = {
        type: "texture",
        dimension: undefined,
        sampleType: "depth",
    };
    const layout = materialLayout(d.struct({ unused: d.f32 }), {
        layers,
        less,
        unfiltered,
        nearest,
        unsigned,
        volume,
        depthCube,
        samples,
        storageVolume,
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
    expectTypeOf<Resources["volume"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTexture3d<d.F32>>
    >();
    expectTypeOf<Resources["depthCube"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTextureDepth2d> | d.InferGPU<d.WgslTextureDepthCube>
    >();
    expect([layout.entries.volume.texture.type, layout.entries.depthCube.texture.type]).toEqual([
        "texture_2d",
        "texture_depth_2d",
    ]);
    expectTypeOf<Resources["samples"]>().toEqualTypeOf<
        d.InferGPU<d.WgslTexture2d<d.F32>> | d.InferGPU<d.WgslTextureMultisampled2d<d.F32>>
    >();
    expect(layout.entries.samples.texture.type).toBe("texture_2d");
    expectTypeOf<Resources["storageVolume"]>().toEqualTypeOf<
        | d.InferGPU<d.WgslStorageTexture2d<"r32uint", "write-only">>
        | d.InferGPU<d.WgslStorageTexture3d<"r32uint", "write-only">>
    >();
    expect(layout.entries.storageVolume.storageTexture.type).toBe("texture_storage_2d");
});
