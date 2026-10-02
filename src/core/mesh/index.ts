import type { Plugin } from "../../engine";
import { registration } from "../../engine";
import { MeshInstance } from "./instance";
import { Meshes } from "./mesh";

export { MeshInstance } from "./instance";

import { clearMeshes, flushMeshes, initializeMeshState } from "./mesh";
import { initMeshes } from "./primitives";

export type { Mesh, MeshBinding, MeshIndex } from "./mesh";
export {
    Meshes,
    registerMesh,
} from "./mesh";

/** Owns this world's mesh registry and GPU storage, including the unit cube, sphere and capsule.
 * Static registrations during initialize are packed together at warm, before the first draw.
 */
export const MeshPlugin: Plugin = {
    name: "Mesh",
    components: [
        registration("MeshInstance", MeshInstance, {
            defaults: (world) => ({ mesh: world.resource(Meshes).id("cube") ?? 0 }),
        }),
    ],
    initialize(world) {
        initializeMeshState(world);
        clearMeshes(world);
        initMeshes(world);
    },
    warm: flushMeshes,
};
