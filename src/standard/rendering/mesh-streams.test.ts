import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import type { Mesh } from "../../core/mesh";
import type { World } from "../../engine";
import { type MaterialType, materialType } from "./material-type";
import { compileMaterial, meshStreamMask } from "./pipelines";

const colorOnly = materialType({
    name: "ColorStreamKey",
    parameters: d.struct({ unused: d.u32 }),
    defaults: { unused: 0 },
    fragmentInputs: { color: true },
    fragment: (ctx) => {
        "use gpu";
        return ctx.color;
    },
});
test("stream specialization retains physical addressing but collapses unconsumed layouts to zero", () => {
    const mesh = (streams?: Mesh["streams"]) => ({ streams }) as Mesh;
    expect(meshStreamMask(colorOnly, mesh())).toBe(0);
    expect(meshStreamMask(colorOnly, mesh({ uv1: true, tangent: true }))).toBe(0);
    expect(meshStreamMask(colorOnly, mesh({ color: true }))).toBe(4);
    expect(meshStreamMask(colorOnly, mesh({ uv1: true, tangent: true, color: true }))).toBe(7);
    expect(
        meshStreamMask(
            { ...colorOnly, fragmentInputs: undefined },
            mesh({ uv1: true, tangent: true, color: true }),
        ),
    ).toBe(0);
});
test("the inter-stage budget counts and names only present declared streams", () => {
    const world = {
        gpu: { device: { limits: { maxInterStageShaderVariables: 9 } } },
    } as unknown as World;
    const surface = {
        ...colorOnly,
        fragmentInputs: { uv1: true, color: true, tangent: true },
    } as MaterialType;
    expect(() => compileMaterial(world, surface, 1, 0, 0, 7)).toThrow(
        /10 inter-stage variables.*uv1.*tangent.*color/,
    );
});
