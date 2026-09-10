// glTF import — the author surface: bring a .gltf / .glb into the mesh registry and place it. A scene
// referencing a primitive by name (`part="mesh: model.glb#0"`) imports it declaratively — GltfPlugin's
// preloader loads the file before the scene resolves, and its route sync decorates the Part with the
// textured/skinned surface + material. The importer itself stays a one-way utility — it registers meshes /
// surfaces / VATs and returns a descriptor, creating no entities; place programmatically via `placeGltf` /
// `placeScene`. The decode / cache / tooling surface follows at the end of this file, off the `extras` barrel.

export type { GltfHandle, GltfImport, GltfPlacement } from "./assets";
export { GltfPlugin, loadGltf, placeGltf, placeScene } from "./assets";
export { Textured } from "./routes";

// glTF extension surface for tooling + custom async pipelines: the deviceless `decode` and the
// content-keyed cache (`ensureDecoded` / `register`), off-thread decode (`decodeInWorker`), the
// union-staging progress (`unionPending`) + cache management (`invalidate` / `clearGltfCache` /
// `gltfCacheStats`), the PBR baseColor size-bucket names, and the raw parsed-glTF types. The author happy
// path (`loadGltf` / `placeScene` / `GltfPlugin`) rides the barrel.

export type { DecodedGltf } from "./assets";
export {
    clearGltfCache,
    decode,
    ensureDecoded,
    gltfCacheStats,
    invalidate,
    register,
    unionPending,
} from "./assets";
export type { GltfImage, GltfInstance, GltfJson, GltfMaterial, GltfMesh, GltfScene } from "./gltf";
export { ALBEDO_BUCKETS, ALBEDO_NAMES } from "./image";
export { decodeInWorker } from "./pool";
