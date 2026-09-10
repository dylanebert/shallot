import {
    GltfPlugin,
    loadGltf,
    type Plugin,
    placeGltf,
    placeScene,
    type State,
    Transform,
} from "@dylanebert/shallot";

// The declarative path lives in the scene: point a Part's mesh at `<file>#<primitive>` and the engine
// imports the file before load. Textures and skinned animation come along; the registered name lives in
// the same mesh registry as a built-in (`mesh: cube`), so everything else about the entity is ordinary.

// The programmatic path: `loadGltf` registers the asset's primitives and hands back a descriptor,
// creating no entities. Place a single mesh with `placeGltf`, or spawn a whole asset's node layout at
// its baked transforms with `placeScene` — one call for a full environment (a level, a set).
const Models = {
    name: "Models",
    dependencies: [GltfPlugin],
    async warm(state: State) {
        const asset = await loadGltf(state, "box.gltf");
        for (let i = 0; i < 3; i++)
            placeGltf(state, asset.meshes[0], { pos: [i * 1.2 - 1.2, 1.6, 0] });
        // placeScene lands the asset at its authored origin and returns the eids; reposition them here so
        // they don't overlap the instanced row above.
        for (const eid of placeScene(state, asset)) Transform.pos.set(eid, 0, 3.2, 0, 0);
    },
} satisfies Plugin;

// A skinned mesh bakes its clips to a vertex-animation texture and plays them by name. To pose a rig
// from your own code instead (a ragdoll, IK, a network stream), import it with `loadGltf(url, { live:
// true })` and write its joint palette each frame through `LiveSkin` — the ragdoll recipe is the worked
// example.

export default Models;
