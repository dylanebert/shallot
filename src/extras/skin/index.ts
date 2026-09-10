// the live joint-palette skinning substrate — the author surface: add `SkinPlugin`, allocate a rig's
// palette block, write a pose each frame, register a mesh's per-vertex joints/weights. Producers are
// procedural (a physics ragdoll, a scripted driver) or an importer (`extras/gltf` converts a glTF rig into
// the same substrate data). It provides no surface: the WGSL a surface splices to read a palette is
// `@dylanebert/shallot/skin/core`.

export { LiveSkin, LiveSkinSystem, Skin, SkinPlugin, skinMatrix, skinTraits } from "./live";

// skinning-substrate extension surface: the WGSL a surface splices to read the joint palette — the vs
// itself, the two preamble chunks (plus the `SkinParams` schema one of them emits), and the bindings they
// name. A surface owning a material path composes
// these (`extras/gltf`'s `skin-live` PBR trio is the worked case). The author happy path (`SkinPlugin` /
// `LiveSkin` / `Skin` / `skinMatrix`) rides `extras`; the block-layout arithmetic + the CPU twins of the
// blend stay internal to `live.ts`, imported directly by its tests.

export { LIVE_SKIN_VS, liveTintWgsl, SkinParams, skinParamsWgsl } from "./live";
