import { component, u32, vec4 } from "../../engine";
import { defaultMeshHandle } from "./mesh";

/** Mesh handle for this entity's geometry. Defaults to the built-in cube. */
export const MeshInstance = component(
    "MeshInstance",
    {
        mesh: u32,
        /** Optional local-space culling sphere `(center.xyz, radius)`; a negative radius uses the mesh bounds. */
        cullBounds: vec4,
    },
    {
        defaults: (world) => ({ mesh: defaultMeshHandle(world), cullBounds: [0, 0, 0, -1] }),
    },
);
