export interface SharedPassContext {
    readonly device: GPUDevice;
    readonly format: GPUTextureFormat;
    readonly maskFormat: GPUTextureFormat;
    readonly eidFormat: GPUTextureFormat;
}

export interface OverlayDraw {
    order: number;
    draw(pass: GPURenderPassEncoder, ctx: SharedPassContext): void;
    dispose?(): void;
}
