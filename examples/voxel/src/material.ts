import { xformNormal, xformPoint } from "@dylanebert/shallot/rendering";
import {
    lit,
    MaterialPlugin,
    materialLayout,
    materialType,
} from "@dylanebert/shallot/standard/rendering";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

const Parameters = d.struct({ grass: d.vec3f, dirt: d.vec3f, inner: d.f32, outer: d.f32 });
const layout = materialLayout(Parameters, { voxel: { type: "attribute", element: d.u32 } });
export const TerrainMaterial = materialType({
    name: "VoxelTerrain",
    layout,
    defaults: {
        grass: d.vec3f(0.1, 0.2, 0.05),
        dirt: d.vec3f(0.1, 0.06, 0.03),
        inner: 0,
        outer: 1,
    },
    vertex: (input) => {
        "use gpu";
        // A face corner cannot identify its voxel on the tangent axes. The producer's
        // packed centre and face also give depth/shadow passes their real normal.
        const voxel = layout.$.voxel[input.vidx];
        const localCenter = d.vec3f(
            d.f32(voxel & 255) + 0.5,
            d.f32((voxel >>> 8) & 255) + 0.5,
            d.f32((voxel >>> 16) & 255) + 0.5,
        );
        const center = xformPoint(input.xform, localCenter);
        const scale =
            1 -
            std.smoothstep(input.parameters.inner, input.parameters.outer, std.length(center.xz));
        const world = center.add(input.world.xyz.sub(center).mul(scale));
        const face = voxel >>> 24;
        let normal = d.vec3f(1, 0, 0);
        if (face === 1) normal = d.vec3f(-1, 0, 0);
        if (face === 2) normal = d.vec3f(0, 1, 0);
        if (face === 3) normal = d.vec3f(0, -1, 0);
        if (face === 4) normal = d.vec3f(0, 0, 1);
        if (face === 5) normal = d.vec3f(0, 0, -1);
        return { world: d.vec4f(world, 1), worldNormal: xformNormal(input.xform, normal) };
    },
    fragment: (ctx) => {
        "use gpu";
        let color = d.vec3f(ctx.parameters.dirt);
        if (ctx.worldNormal.y > 0.5) color = d.vec3f(ctx.parameters.grass);
        return d.vec4f(lit(color, ctx.worldNormal), 1);
    },
});
export const TerrainMaterialPlugin = MaterialPlugin(TerrainMaterial);
