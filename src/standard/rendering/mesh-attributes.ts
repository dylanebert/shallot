import * as d from "typegpu/data";
import type { Mesh } from "../../core/mesh";
import type { MaterialType } from "./material-type";

export function materialAttributeError(
    mesh: Mesh,
    material: MaterialType<any, any, any>,
): string | null {
    for (const name in material.layout.attributes) {
        const element = material.layout.attributes[name]!;
        const stream = mesh.attributes?.[name];
        if (!stream || !d.deepEqual(stream.dataType.elementType, element))
            return `mesh "${mesh.name}" ${stream ? "has a different schema for" : "has no"} attribute "${name}" that material type "${material.name}" reads`;
    }
    return null;
}
