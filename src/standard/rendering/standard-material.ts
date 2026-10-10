import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { litPbr } from "./engine";
import { StandardMaterial, StandardMaterialInput } from "./material-data";
import { materialFragmentContext, materialLayout, materialType } from "./material-type";
import { Pbr } from "./shade";

const standardLayout = materialLayout(StandardMaterialInput, {});
const StandardFragmentContext = materialFragmentContext();

const standardFragment = tgpu.fn(
    [StandardFragmentContext],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const material = StandardMaterialInput(standardLayout.$.materialParameters[ctx.material]);
    const pbr = Pbr({
        albedo: material.baseColor.xyz,
        metallic: material.metallic,
        roughness: material.perceptualRoughness,
        occlusion: material.occlusion,
        dielectric: 0,
        diffuseWrap: material.diffuseWrap,
    });
    const emissive = material.emissive;
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), 1);
});

/** The built-in mesh material type. Its type-local table is the sole source of its PBR parameters. */
export const StandardMaterialType = materialType({
    name: "StandardMaterial",
    parameters: StandardMaterialInput,
    layout: standardLayout,
    fragment: standardFragment,
    defaults: StandardMaterial(),
    blend: "opaque",
    depthPass: { prepass: true, shadows: true },
});
