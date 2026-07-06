import { GltfPlugin, loadGltf, type Plugin, placeScene, type State } from "@dylanebert/shallot";

// #doc:code
// ### Place a whole scene
//
// A full environment is many primitives you don't want to name one by one. `placeScene` takes the import
// descriptor and spawns one Part per node placement — the standard part/transform flow — so the scene
// authors only the camera and lights. Import in `warm` when nothing references the meshes by name (a single
// asset you place with a Part by name imports in `initialize` instead, so the name resolves at parse):
// #region sponza
const Sponza = {
    name: "Sponza",
    dependencies: [GltfPlugin],
    async warm(state: State) {
        placeScene(state, await loadGltf(state, "sponza/Sponza-KTX-Draco.glb"));
    },
} satisfies Plugin;
// #endregion

export default Sponza;
