// glTF import — the author surface: bring a .gltf / .glb into the mesh registry and place it. A scene
// referencing a primitive by name (`part="mesh: model.glb#0"`) imports it declaratively — GltfPlugin's
// preloader loads the file before the scene resolves, and its route sync decorates the Part with the
// textured/skinned surface + material. The importer itself stays a one-way utility — it registers meshes /
// surfaces / VATs and returns a descriptor, creating no entities; place programmatically via `placeGltf` /
// `placeScene`. The decode / cache / tooling surface is `@dylanebert/shallot/gltf/core`.

export type { GltfHandle, GltfImport, GltfPlacement } from "./assets";
export { GltfPlugin, loadGltf, placeGltf, placeScene } from "./assets";
export { Textured } from "./routes";
export { Skin } from "./skin";
