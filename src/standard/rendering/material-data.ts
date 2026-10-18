import * as d from "typegpu/data";
import { component, f32, u32 } from "../../engine";

/**
 * One instance's mesh, material and culling row, read by preprocess and the material shaders.
 * Standard fills entity rows from MeshInstance, MeshMaterial and NotShadowCaster; a producer table
 * writes every field of its rows itself, and a wrong value renders wrong without an error:
 * - `mesh`: a MeshHandle the table declared. An undeclared or unknown handle draws nothing.
 * - `materialType`, `material`: a declared MaterialHandle's `type` and `material`. Another pair draws
 *   nothing; another row of the same type draws with that row's parameters.
 * - `alphaMode`: that handle's `alphaMode`, an AlphaModeCode. It selects the pipeline: a code no
 *   declared handle of the type holds draws nothing, as does any code but Opaque, Mask and
 *   AlphaToCoverage.
 * - `alphaCutoff`: that handle's `alphaCutoff`, an alpha in [0, 1]. Mask, and AlphaToCoverage on
 *   single-sample views, discard fragments whose alpha is below the row's value.
 * - `flags`: bit 0 set keeps the row out of shadow passes, as NotShadowCaster does; write 0 to cast.
 *   Other bits are reserved and must be 0.
 * - `cullBounds`: a local-space culling sphere `(center.xyz, radius)` in mesh units, scaled by the
 *   placement's largest absolute scale; a negative radius uses the mesh's bounds. A radius of 0
 *   culls the row once its center leaves a view, so write `(0, 0, 0, -1)` unless you supply a sphere.
 */
export const MeshInstanceInput = d
    .struct({
        mesh: d.u32,
        materialType: d.u32,
        material: d.u32,
        alphaMode: d.u32,
        alphaCutoff: d.f32,
        flags: d.u32,
        cullBounds: d.vec4f,
    })
    .$name("MeshInstanceInput");

/**
 * Selects one material row from one World-local material type. `type` identifies the shader and typed
 * parameter table; `material` is a row in that table. `alphaMode` and `alphaCutoff` copy that row's alpha
 * into the dense mesh table; the row owns it, and marked copies reconcile immediately before that
 * table gathers fields for upload. A direct write to either is replaced at the next upload, including
 * frames without a fixed tick. Writes after upload reconcile next frame. External parameter sources
 * instead publish alpha directly into the GPU mesh table; their CPU copy is not authoritative.
 * Draws group by `(type, alpha pipeline key, mesh)`. Missing MeshMaterial selects StandardMaterial
 * type 0, row 0 with Opaque mode.
 */
export const MeshMaterial = component(
    "MeshMaterial",
    { type: u32, material: u32, alphaMode: u32, alphaCutoff: f32 },
    { defaults: () => ({ type: 0, material: 0, alphaMode: 0, alphaCutoff: 0.5 }) },
);

/** Standard forward PBR parameters, stored as one typed row per StandardMaterial instance. */
export const StandardMaterialInput = d
    .struct({
        baseColor: d.vec4f,
        baseColorTexture: d.i32,
        metallic: d.f32,
        perceptualRoughness: d.f32,
        emissive: d.vec3f,
        occlusion: d.f32,
        unlit: d.u32,
    })
    .$name("StandardMaterialInput");

export type StandardMaterial = d.Infer<typeof StandardMaterialInput>;
type StandardMaterialOptions = {
    baseColor?: readonly [number, number, number, number];
    /** Image id from registerImage, registered before warm; multiplies baseColor RGBA at mesh UVs. */
    baseColorTexture?: number;
    metallic?: number;
    perceptualRoughness?: number;
    emissive?: readonly [number, number, number];
    occlusion?: number;
    unlit?: boolean;
};

/**
 * Linear colours; emissive is independent of baseColor.
 * baseColorTexture is an id from registerImage before warm; its sampled RGBA multiplies baseColor
 * at mesh UVs. Omit it to leave baseColor untextured, without sampling the image array.
 */
export function StandardMaterial(values: StandardMaterialOptions = {}): StandardMaterial {
    return {
        baseColor: d.vec4f(...(values.baseColor ?? [1, 1, 1, 1])),
        baseColorTexture: values.baseColorTexture ?? -1,
        metallic: values.metallic ?? 0,
        perceptualRoughness: values.perceptualRoughness ?? 0.5,
        emissive: d.vec3f(...(values.emissive ?? [0, 0, 0])),
        occlusion: values.occlusion ?? 1,
        unlit: values.unlit ? 1 : 0,
    };
}
