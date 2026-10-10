import * as d from "typegpu/data";
import { component, u32 } from "../../engine";

/** Dense per-MeshInstance fields consumed by preprocess and the material shaders. */
export const MeshInstanceInput = d
    .struct({
        mesh: d.u32,
        materialType: d.u32,
        material: d.u32,
        flags: d.u32,
        cullBounds: d.vec4f,
    })
    .$name("MeshInstanceInput");

/**
 * Selects one material row from one World-local material type. `type` identifies the shader and typed
 * parameter table; `material` is a row in that table. Mesh draws group by `(type, mesh)`. Missing
 * MeshMaterial selects StandardMaterial type 0, row 0.
 */
export const MeshMaterial = component(
    "MeshMaterial",
    { type: u32, material: u32 },
    { defaults: () => ({ type: 0, material: 0 }) },
);

/** Standard forward PBR parameters, stored as one typed row per StandardMaterial instance. */
export const StandardMaterialInput = d
    .struct({
        baseColor: d.vec4f,
        metallic: d.f32,
        perceptualRoughness: d.f32,
        emissive: d.vec3f,
        occlusion: d.f32,
        diffuseWrap: d.f32,
        unlit: d.u32,
    })
    .$name("StandardMaterialInput");

export type StandardMaterial = d.Infer<typeof StandardMaterialInput>;
type StandardMaterialOptions = {
    baseColor?: readonly [number, number, number, number];
    metallic?: number;
    perceptualRoughness?: number;
    emissive?: readonly [number, number, number];
    occlusion?: number;
    diffuseWrap?: number;
    unlit?: boolean;
};

/** Linear colours; emissive is independent of baseColor. diffuseWrap 1 preserves Shallot's diffuse look. */
export function StandardMaterial(values: StandardMaterialOptions = {}): StandardMaterial {
    return {
        baseColor: d.vec4f(...(values.baseColor ?? [1, 1, 1, 1])),
        metallic: values.metallic ?? 0,
        perceptualRoughness: values.perceptualRoughness ?? 0.5,
        emissive: d.vec3f(...(values.emissive ?? [0, 0, 0])),
        occlusion: values.occlusion ?? 1,
        diffuseWrap: values.diffuseWrap ?? 1,
        unlit: values.unlit ? 1 : 0,
    };
}
