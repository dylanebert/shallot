import { GltfPlugin, loadGltf, type Plugin, placeScene, type State } from "@dylanebert/shallot";

// the asset-swap flow's fixture, as the pure-import flow: loadGltf registers the box's mesh, placeScene
// spawns it (one Part with the model's baseColorFactor as its Color). Re-importing on a rebuild is what the
// editor's live asset-swap drives — the watcher invalidates "box.gltf" and rebuilds, so this warm re-runs,
// re-decodes the edited file off-thread, and re-spawns the box with the new color, no page reload.
const Box = {
    name: "Box",
    dependencies: [GltfPlugin],
    async warm(state: State) {
        placeScene(state, await loadGltf(state, "box.gltf"));
    },
} satisfies Plugin;

export default Box;
