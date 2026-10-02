import type { Plugin } from "../../engine";
import { clearMeshes, flushMeshes, initializeMeshState } from "./mesh";
import { initMeshes } from "./primitives";

export type { Mesh, MeshBinding, MeshIndex, MeshStorage, QuantStreams } from "./mesh";
export {
    Meshes,
    meshBounds,
    packMeshes,
    quantizeMeshes,
    registerMesh,
    VERTEX_FLOATS,
    VERTEX_STRIDE,
} from "./mesh";

/** Owns this world's mesh registry and GPU storage, including the unit cube, sphere and capsule.
 * Static registrations during initialize are packed together at warm, before the first draw.
 */
export const MeshPlugin: Plugin = {
    name: "Mesh",
    initialize(world) {
        initializeMeshState(world);
        clearMeshes(world);
        initMeshes(world);
    },
    warm: flushMeshes,
};
