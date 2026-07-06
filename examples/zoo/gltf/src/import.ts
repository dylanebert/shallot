import { GltfPlugin, loadGltf, type Plugin, placeGltf, type State } from "@dylanebert/shallot";

// #doc:intro
// glTF import: reference a `.glb` or `.gltf` primitive by name in the scene and it loads itself, or import
// programmatically and place it from code.

// #doc:code source:gltf/public/scenes/gltf.scene
// The declarative path: point a Part's mesh at `<file>#<primitive>` and the engine imports the file before
// the scene loads. Textures and skinned animation come along automatically. The registered name lives in
// the same mesh registry as a built-in (`mesh: cube`), so everything else about the entity is ordinary.

// #doc:code
// The programmatic path: `loadGltf` registers the asset's primitives and hands back a descriptor, creating
// no entities. Place a handle yourself with `placeGltf` (or the whole asset with `placeScene`). Here a row
// of boxes off one import:
// #region import
const Models = {
    name: "Models",
    dependencies: [GltfPlugin],
    async warm(state: State) {
        const { meshes } = await loadGltf(state, "box.gltf");
        for (let i = 0; i < 3; i++) placeGltf(state, meshes[0], { pos: [i * 1.2 - 1.2, 1.6, 0] });
    },
} satisfies Plugin;
// #endregion

export default Models;
