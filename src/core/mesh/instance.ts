import { component, u32 } from "../../engine";
import { defaultMeshHandle } from "./mesh";

/** Mesh handle for this entity's geometry. Defaults to the built-in cube. */
export const MeshInstance = component(
    "MeshInstance",
    {
        mesh: u32,
    },
    {
        defaults: (world) => ({ mesh: defaultMeshHandle(world) }),
    },
);
