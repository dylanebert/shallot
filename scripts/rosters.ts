// Committed rosters for the check-docs citation-resolution arm.
//
// A roster is a resolution target for tokens that don't resolve against the
// shallot source tree (*.ts / *.rs / *.wgsl) but are legitimate foreign
// references — WGSL built-in functions/types, WebGPU API names, symbols from
// foreign codebases (Bevy, Jolt, Box3D, webphysics, Bullet, PlayCanvas, D3D12,
// Vulkan, CUDA, TypeGPU, SteamAudio, WasmFeatures, Tools). Each roster is
// asserted non-empty by the arm, and every entry is asserted cited by at
// least one rule file (both ways: a real member, genuinely needed) AND
// asserted absent from the tree token index (the disjointness law: every
// disjunct's member set is disjoint from every other's, so each surviving
// member is load-bearing and removing one reds). Zero slack means a launder
// cannot occupy an existing slot, and adding one moves the pinned entry
// count in the diff that adds it.
//
// These rosters are COMMITTED FILES inside shallot — a consumer checkout of
// shallot has them, so the arm does not degrade to a skip the way it would if
// the roster were read from kex's `reference/` tree (absent from a consumer
// checkout, the same way the gltf corpus already is in a linked worktree).

// ── WGSL built-in functions and types ──────────────────────────────────────
//
// From the WGSL specification. A cited WGSL built-in that no .ts/.rs/.wgsl
// file in the shallot tree contains resolves against this roster. Pruned to
// only entries cited by at least one rule file AND absent from the tree token
// index (round 7 — 34 entries that also resolved in the tree were pruned,
// making every surviving entry load-bearing against a swap-in).
export const WGSL_BUILTINS: ReadonlySet<string> = new Set([
    // unpack/pack
    "unpack4x8snorm",
    "pack4x8snorm",
    // subgroup operations
    "subgroupAdd",
    // matrix types
    "mat4x4",
]);

// ── WebGPU API types and methods ──────────────────────────────────────────
//
// From the WebGPU IDL. A cited WebGPU API name that no .ts/.rs/.wgsl file in
// the shallot tree contains resolves against this roster. Includes names that
// were removed from the spec (e.g. `writeTimestamp`) — the roster resolves
// the name; the prose documents the removal. Pruned to only entries cited
// by at least one rule file AND absent from the tree token index (round 7).
export const WEBGPU_IDL: ReadonlySet<string> = new Set([
    // types
    "GPURenderBundle",
    // methods
    "executeBundles",
    "drawIndexed",
    // timestamp (writeTimestamp was removed from the spec)
    "TimestampWrites",
    "writeTimestamp",
]);

// ── Foreign-namespace vendored symbol lists ───────────────────────────────
//
// Each namespace is a foreign codebase whose symbols are cited in the rules as
// structural references, not shallot code. A cited symbol from one of these
// namespaces that doesn't resolve against the shallot tree resolves against
// the namespace's roster. Pruned to only entries cited by at least one rule
// file AND absent from the tree token index (round 7).
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
    PlayCanvas: new Set(["LightTextureAtlas"]),
    D3D12: new Set(["ExecuteIndirect"]),
    Vulkan: new Set(["vkCmdDrawIndirectCount"]),
    CUDA: new Set(["__threadfence"]),
    TypeGPU: new Set(["__TYPEGPU_AUTONAME__", "sideEffects"]),
    SteamAudio: new Set(["gain_effect", "direct_effect"]),
    WasmFeatures: new Set(["memory64"]),
    Tools: new Set(["PowerVR", "RenderDoc", "webgpu_inspector"]),
};
