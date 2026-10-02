import { MeshPlugin } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import { type Plugin, registration } from "../../engine";
import { MeshMaterial } from "./material";
import {
    initializeMeshPreprocess,
    initMeshPreprocess,
    MeshPreprocessSystem,
    warmMeshPreprocess,
} from "./preprocess";

/** Packs mesh instances into per-view, per-surface indirect draws. */
export const MeshRenderPlugin: Plugin = {
    name: "MeshRender",
    systems: [MeshPreprocessSystem],
    components: [registration("MeshMaterial", MeshMaterial, { defaults: () => ({ material: 0 }) })],
    dependencies: [RenderingPlugin, MeshPlugin],
    initialize(world) {
        initializeMeshPreprocess(world);
        initMeshPreprocess(world);
    },
    warm: warmMeshPreprocess,
};
