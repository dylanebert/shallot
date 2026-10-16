import { expect, expectTypeOf, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { materialFragmentContext, materialType, materialVertexInput } from "./material-type";

test("a material without owned bindings needs only its parameter schema", () => {
    const Parameters = d.struct({ tint: d.vec4f, scale: d.f32 });
    const Context = materialFragmentContext(Parameters);
    const Vertex = materialVertexInput(Parameters);
    expectTypeOf<d.Infer<typeof Context>["parameters"]>().toEqualTypeOf<
        d.Infer<typeof Parameters>
    >();
    expectTypeOf<d.Infer<typeof Vertex>["parameters"]>().toEqualTypeOf<
        d.Infer<typeof Parameters>
    >();
    const material = materialType({
        name: "NoOwnedBindings",
        parameters: Parameters,
        defaults: { tint: d.vec4f(1), scale: 2 },
        fragment: tgpu.fn(
            [Context],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            return d.vec4f(ctx.parameters.tint);
        }),
    });
    expect(material.parameters).toBe(Parameters);
    expect(material.layout.parameters).toBe(Parameters);
    expect(material.layout.attributes).toEqual({});
});
