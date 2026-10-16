import { expect, expectTypeOf, test } from "bun:test";
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
