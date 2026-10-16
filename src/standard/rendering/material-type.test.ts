import { expect, expectTypeOf, test } from "bun:test";
import * as d from "typegpu/data";
import { materialType } from "./material-type";

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
