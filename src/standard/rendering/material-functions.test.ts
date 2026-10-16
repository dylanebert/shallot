import { expect, expectTypeOf, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { materialLayout, materialType } from "./material-type";

test("plain material shaders infer parameters and varyings and close over owned bindings", () => {
    const Parameters = d.struct({ tint: d.vec4f, scale: d.f32 });
    const Uniform = d.struct({ offset: d.vec3f });
    const layout = materialLayout(Parameters, { motion: { type: "uniform", struct: Uniform } });
    const material = materialType({
        name: "PlainFunctions",
        layout,
        defaults: { tint: d.vec4f(1), scale: 2 },
        varyings: { shade: d.vec3f },
        vertex: (input) => {
            "use gpu";
            expectTypeOf(input.parameters).toEqualTypeOf<d.Infer<typeof Parameters>>();
            expectTypeOf(layout.$.motion).toEqualTypeOf<d.Infer<typeof Uniform>>();
            return {
                world: input.world,
                worldNormal: input.worldNormal,
                shade: layout.$.motion.offset,
            };
        },
        fragment: (ctx) => {
            "use gpu";
            expectTypeOf(ctx.parameters).toEqualTypeOf<d.Infer<typeof Parameters>>();
            expectTypeOf(ctx.shade).toEqualTypeOf<d.v3f>();
            return d.vec4f(ctx.shade, ctx.parameters.scale);
        },
    });
    expect(material.layout).toBe(layout);
});

test("material registration refuses a prebuilt TypeGPU fragment and names the plain-function remedy", () => {
    const Parameters = d.struct({ tint: d.vec4f });
    const fragment = tgpu.fn(
        [d.struct({ parameters: Parameters })],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.parameters.tint);
    });
    expect(() =>
        materialType({
            name: "WrappedFragment",
            parameters: Parameters,
            defaults: { tint: d.vec4f(1) },
            // @ts-expect-error Material stages accept plain functions, not prebuilt TypeGPU functions.
            fragment,
        }),
    ).toThrow(
        'material type "WrappedFragment" fragment shader must be a plain "use gpu" function, not a prebuilt TypeGPU function',
    );
});

test("material registration refuses a prebuilt TypeGPU vertex and names the plain-function remedy", () => {
    const Parameters = d.struct({ tint: d.vec4f });
    const Vertex = d.struct({ world: d.vec4f, worldNormal: d.vec3f });
    const vertex = tgpu.fn(
        [Vertex],
        Vertex,
    )((input) => {
        "use gpu";
        return Vertex({ world: input.world, worldNormal: input.worldNormal });
    });
    expect(() =>
        materialType({
            name: "WrappedVertex",
            parameters: Parameters,
            defaults: { tint: d.vec4f(1) },
            // @ts-expect-error Material stages accept plain functions, not prebuilt TypeGPU functions.
            vertex,
            fragment: (ctx) => {
                "use gpu";
                return d.vec4f(ctx.parameters.tint);
            },
        }),
    ).toThrow(
        'material type "WrappedVertex" vertex shader must be a plain "use gpu" function, not a prebuilt TypeGPU function',
    );
});
