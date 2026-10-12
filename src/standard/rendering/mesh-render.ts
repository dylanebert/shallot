import { MeshPlugin } from "../../core/mesh";
import { RenderingPlugin } from "../../core/rendering";
import type { Plugin } from "../../engine";
import { MeshMaterial } from "./material";
import { MeshMaterialAlphaSystem } from "./material-type";
import {
    initializeMeshPreprocess,
    initMeshPreprocess,
    MeshPreprocessSystem,
    warmMeshPreprocess,
} from "./preprocess";

/** Packs mesh instances into per-view, per-surface indirect draws. */
export const MeshRenderPlugin: Plugin = {
    gpu: {},
    name: "MeshRender",
    systems: [MeshMaterialAlphaSystem, MeshPreprocessSystem],
    components: [MeshMaterial],
    dependencies: [RenderingPlugin, MeshPlugin],
    initialize(world) {
        initializeMeshPreprocess(world);
        initMeshPreprocess(world);
    },
    warm: warmMeshPreprocess,
};
