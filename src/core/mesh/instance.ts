import { component, u32 } from "../../engine";
import { Meshes } from "./mesh";

/** Mesh registry ID for this entity's geometry. Defaults to the built-in cube. */
export const MeshInstance = component(
    "MeshInstance",
    {
        mesh: u32,
    },
    {
        defaults: (world) => ({ mesh: world.resource(Meshes).id("cube") ?? 0 }),
    },
);
