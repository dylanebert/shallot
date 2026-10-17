import {
    lit,
    MaterialPlugin,
    materialLayout,
    materialType,
} from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";

const Parameters = d.struct({ grass: d.vec3f, dirt: d.vec3f });
export const TerrainMaterial = materialType({
    name: "VoxelTerrain",
    layout: materialLayout(Parameters, {}),
    defaults: { grass: d.vec3f(0.1, 0.2, 0.05), dirt: d.vec3f(0.1, 0.06, 0.03) },
    fragment: (ctx) => {
        "use gpu";
        let color = d.vec3f(ctx.parameters.dirt);
        if (ctx.worldNormal.y > 0.5) color = d.vec3f(ctx.parameters.grass);
        return d.vec4f(lit(color, ctx.worldNormal), 1);
    },
});
export const TerrainMaterialPlugin = MaterialPlugin(TerrainMaterial);
