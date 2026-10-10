import type { Plugin } from "../../engine";
import { MeshInstance } from "./instance";

export { MeshInstance } from "./instance";
export { capsule, cube, sphere } from "./primitives";

import {
    clearMeshes,
    flushMeshes,
    initializeMeshState,
    PrepareMeshesSystem,
    setDefaultMeshHandle,
} from "./mesh";
import { initMeshes } from "./primitives";

export type { Mesh, MeshBinding, MeshHandle, MeshIndex } from "./mesh";
export {
    Meshes,
    registerMesh,
} from "./mesh";

/** Owns this world's mesh registry and GPU storage, including the unit cube, sphere and capsule.
 * Registrations during initialize are packed together at warm, before the first draw; later
 * ones are packed together at the start of the next draw group. `registerMesh` refuses before
 * this plugin initializes.
 */
export const MeshPlugin: Plugin = {
    gpu: {},
    name: "Mesh",
    components: [MeshInstance],
    systems: [PrepareMeshesSystem],
    initialize(world) {
        initializeMeshState(world);
        clearMeshes(world);
        setDefaultMeshHandle(world, initMeshes(world));
    },
    warm: flushMeshes,
};
