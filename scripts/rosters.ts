// Foreign symbols cited by rules but absent from the source token index.
// check-docs asserts every class nonempty, every entry cited and absent from
// the tree, and the total pinned. No uncited slots can launder dead citations.
export const FOREIGN_NAMESPACES: Record<string, ReadonlySet<string>> = {
    Bevy: new Set([
        "GpuImage",
        "RenderApp",
        "add_slot_edge",
        "ChangeDetection",
        "apply_deferred",
        "SystemParam",
    ]),
    Jolt: new Set(["SolveConstraints", "WalkStairs"]),
    Box3D: new Set([
        "b3Shape_SetSphere",
        "SetCapsule",
        "b3Shape_SetFilter",
        "SetCollideConnected",
        "emitAux",
        "SceneStepFn",
        "g_scenes",
        "b3DynamicTree",
        "BOX3D_FORCE_OVERFLOW",
    ]),
    webphysics: new Set([
        "contactSlop",
        "dispatchBodyCount",
        "broadPhase.ts",
        "reference/webphysics/.../avbdState.ts",
    ]),
    Bullet: new Set(["BatchSolveKernelContact"]),
    TypeGPU: new Set(["sideEffects"]),
    SteamAudio: new Set(["gain_effect", "direct_effect"]),
    WasmFeatures: new Set(["memory64"]),
};
