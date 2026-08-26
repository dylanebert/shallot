// Committed rosters for the check-docs citation-resolution arm.
//
// A roster is a resolution target for tokens that don't resolve against the
// shallot source tree (*.ts / *.rs / *.wgsl) but are legitimate foreign
// references — WGSL built-in functions/types, WebGPU API names, symbols from
// foreign codebases (Bevy, Jolt, Box3D, webphysics, Bullet, PlayCanvas, D3D12,
// Vulkan, CUDA), or x86/ISA mnemonics. Each roster is asserted non-empty by
// the arm: a roster that loses its last entry would make the arm vacuously
// green for that class, so the arm catches it.
//
// These rosters are COMMITTED FILES inside shallot — a consumer checkout of
// shallot has them, so the arm does not degrade to a skip the way it would if
// the roster were read from kex's `reference/` tree (absent from a consumer
// checkout, the same way the gltf corpus already is in a linked worktree).

// ── WGSL built-in functions and types ──────────────────────────────────────
//
// From the WGSL specification. A cited WGSL built-in that no .ts/.rs/.wgsl
// file in the shallot tree contains resolves against this roster.
export const WGSL_BUILTINS: ReadonlySet<string> = new Set([
    // unpack/pack
    "unpack4x8snorm",
    "unpack4x8unorm",
    "unpack2x16float",
    "unpack2x16snorm",
    "unpack4x8unorm",
    "pack4x8snorm",
    "pack4x8unorm",
    "pack2x16float",
    "pack2x16snorm",
    // subgroup operations
    "subgroupAdd",
    "subgroupBallot",
    "subgroupElect",
    "subgroupBroadcast",
    "subgroupShuffle",
    "subgroupMul",
    "subgroupMin",
    "subgroupMax",
    // matrix types
    "mat2x2",
    "mat2x3",
    "mat2x4",
    "mat3x2",
    "mat3x3",
    "mat3x4",
    "mat4x2",
    "mat4x3",
    "mat4x4",
    // vector swizzle / built-in types
    "vec2",
    "vec3",
    "vec4",
    // atomic functions
    "atomicAdd",
    "atomicSub",
    "atomicMin",
    "atomicMax",
    "atomicAnd",
    "atomicOr",
    "atomicXor",
    "atomicExchange",
    "atomicCompareExchangeWeak",
    // texture functions
    "textureSample",
    "textureSampleLevel",
    "textureSampleGrad",
    "textureSampleCompare",
    "textureLoad",
    "textureStore",
    "textureDimensions",
    // barrier / workgroup
    "storageBarrier",
    "workgroupBarrier",
    "workgroupUniformLoad",
    // derivative
    "dpdx",
    "dpdy",
    "fwidth",
    "dpdxCoarse",
    "dpdyCoarse",
    "fwidthCoarse",
    "dpdxFine",
    "dpdyFine",
    "fwidthFine",
    // packed dot
    "dot4U8Packed",
    "dot4I8Packed",
    // misc
    "arrayLength",
    "select",
    "clamp",
]);

// ── WebGPU API types and methods ──────────────────────────────────────────
//
// From the WebGPU IDL. A cited WebGPU API name that no .ts/.rs/.wgsl file in
// the shallot tree contains resolves against this roster. Includes names that
// were removed from the spec (e.g. `writeTimestamp`) — the roster resolves
// the name; the prose documents the removal.
export const WEBGPU_IDL: ReadonlySet<string> = new Set([
    // types
    "GPURenderBundle",
    "GPURenderPassEncoder",
    "GPUComputePassEncoder",
    "GPUCommandEncoder",
    "GPUBuffer",
    "GPUTexture",
    "GPUSampler",
    "GPUBindGroup",
    "GPUPipelineLayout",
    "GPUShaderModule",
    "GPUDevice",
    "GPUAdapter",
    "GPUCanvasContext",
    // methods
    "executeBundles",
    "drawIndexed",
    "drawIndexedIndirect",
    "drawIndirect",
    "setIndexBuffer",
    "setVertexBuffer",
    "setBindGroup",
    "createBindGroup",
    "createBuffer",
    "createTexture",
    "createSampler",
    "createComputePipeline",
    "createRenderPipeline",
    "createComputePipelineAsync",
    "createRenderPipelineAsync",
    "createShaderModule",
    "createBindGroupLayout",
    "beginComputePass",
    "beginRenderPass",
    "endComputePass",
    "endRenderPass",
    "executeIndirect",
    "copyBufferToBuffer",
    "copyBufferToTexture",
    "copyTextureToBuffer",
    "requestDevice",
    "requestAdapter",
    // timestamp (writeTimestamp was removed from the spec)
    "TimestampWrites",
    "writeTimestamp",
]);

// ── Foreign-namespace vendored symbol lists ───────────────────────────────
//
// Each namespace is a foreign codebase whose symbols are cited in the rules as
// structural references, not shallot code. A cited symbol from one of these
// namespaces that doesn't resolve against the shallot tree resolves against
// the namespace's roster.
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
        "avbdState",
        "broadPhase.ts",
        "reference/webphysics/.../avbdState.ts",
    ]),
    Bullet: new Set(["BatchSolveKernelContact"]),
    PlayCanvas: new Set(["LightTextureAtlas"]),
    D3D12: new Set(["ExecuteIndirect"]),
    Vulkan: new Set(["vkCmdDrawIndirectCount"]),
    CUDA: new Set(["__threadfence"]),
};

// ── x86/ISA mnemonic roster ────────────────────────────────────────────────
//
// x86 and other ISA mnemonics cited in the rules as hardware references, not
// shallot code.
export const X86_ISA: ReadonlySet<string> = new Set(["vgatherdps"]);
