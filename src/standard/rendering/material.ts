import type { World } from "../../engine";
import { MeshMaterial, StandardMaterial, StandardMaterialInput } from "./material-data";
import { StandardMaterialType, VertexMaterialType } from "./standard-material";

export type { MaterialHandle, MaterialType } from "./material-type";
export {
    AlphaMode,
    AlphaPipelineKey,
    MaterialPlugin,
    MaterialTypes,
    materialAssets,
    materialType,
    materialTypeId,
    materialTypes,
} from "./material-type";
export {
    MeshMaterial,
    StandardMaterial,
    StandardMaterialInput,
    StandardMaterialType as Materials,
    StandardMaterialType,
    VertexMaterialType,
};

/** The StandardMaterial type's per-world GPU parameter table. */
export function materialTable(world: World) {
    return world.resource(StandardMaterialType).table;
}
