// the live joint-palette skinning substrate — the author surface: add `SkinPlugin`, allocate a rig's
// palette block, write a pose each frame, register a mesh's per-vertex joints/weights. Producers are
// procedural (a physics ragdoll, a scripted driver) or an importer (`extras/gltf` converts a glTF rig into
// the same substrate data). It provides no surface: the WGSL a surface splices to read a palette is
// `@dylanebert/shallot/skin/core`.

export { LiveSkin, LiveSkinSystem, Skin, SkinPlugin, skinMatrix, skinTraits } from "./live";
