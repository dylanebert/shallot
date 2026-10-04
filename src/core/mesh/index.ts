import type { Plugin } from "../../engine";
import { MeshInstance } from "./instance";

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
    components: [MeshInstance],
    initialize(world) {
        initializeMeshState(world);
        clearMeshes(world);
        initMeshes(world);
    },
    warm: flushMeshes,
};
