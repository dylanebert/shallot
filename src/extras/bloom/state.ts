import type { EffectPass } from "../../core/rendering";

export const BloomCompositeMode = {
    EnergyConserving: 0,
    Additive: 1,
} as const;

interface BlendSettings {
    intensity: number;
    lowFrequencyBoost: number;
    lowFrequencyBoostCurvature: number;
    highPassFrequency: number;
    compositeMode: number;
}
// Bevy bloom/mod.rs::compute_blend_factor; mip 0 blends the top mip into the original.
export function blendFactor(bloom: BlendSettings, mip: number, maxMip: number): number {
    let boost =
        (1 - (1 - mip / maxMip) ** (1 / (1 - bloom.lowFrequencyBoostCurvature))) *
        bloom.lowFrequencyBoost;
    const highPass =
        1 -
        Math.min(
            Math.max((mip / maxMip - bloom.highPassFrequency) / bloom.highPassFrequency, 0),
            1,
        );
    boost *= bloom.compositeMode === BloomCompositeMode.EnergyConserving ? 1 - bloom.intensity : 1;
    return (bloom.intensity + boost) * highPass;
}
interface Chain {
    texture: GPUTexture;
    width: number;
    height: number;
    views: GPUTextureView[];
    groups: GPUBindGroup[];
    blends: GPUColorDict[];
}
export interface CameraState {
    buffer: GPUBuffer;
    bytes: Float32Array;
    words: Uint32Array;
    effect: EffectPass;
    chain: Chain | null;
    groups: WeakMap<GPUTextureView, { first: GPUBindGroup; final: GPUBindGroup }>;
    mode: number;
    settings: BlendSettings;
}
export const stateKey = {
    create: () => ({
        first: null as GPURenderPipeline | null,
        down: null as GPURenderPipeline | null,
        up: [] as GPURenderPipeline[],
        composite: null as GPURenderPipeline | null,
        layout: null as GPUBindGroupLayout | null,
        sampler: null as GPUSampler | null,
        cameras: new Map<number, CameraState>(),
        pass: {
            label: "bloom",
            colorAttachments: [{ view: null!, loadOp: "clear", storeOp: "store" }],
        } as GPURenderPassDescriptor,
    }),
};
export type State = ReturnType<typeof stateKey.create>;
