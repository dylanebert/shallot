// Foreign symbols cited by rules but absent from the source token index.
// check-docs asserts every class nonempty, every entry cited and absent from
// the tree, and the total pinned. No uncited slots can launder dead citations.
export const FOREIGN_NAMESPACES: Record<string, ReadonlySet<string>> = {
    TypeGPU: new Set(["sideEffects"]),
    SteamAudio: new Set(["gain_effect", "direct_effect"]),
};
