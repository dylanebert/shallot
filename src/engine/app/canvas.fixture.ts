/** A headless canvas surface with context-owned textures, released on resize or unconfigure.
 * Real browser swapchain textures belong to the context, not the world's allocation factories. */
export class CanvasContext {
    #configuration: GPUCanvasConfiguration | undefined;
    #texture: GPUTexture | undefined;
    #width: number;
    #height: number;
    readonly canvas: HTMLCanvasElement;

    constructor(canvas: HTMLCanvasElement, width: number, height: number) {
        this.canvas = canvas;
        this.#width = width;
        this.#height = height;
    }

    configure(configuration: GPUCanvasConfiguration): void {
        this.unconfigure();
        this.#configuration = configuration;
    }

    unconfigure(): void {
        this.#texture?.destroy();
        this.#texture = undefined;
        this.#configuration = undefined;
    }

    setSize(width: number, height: number): void {
        if (width === this.#width && height === this.#height) return;
        this.#texture?.destroy();
        this.#texture = undefined;
        this.#width = width;
        this.#height = height;
    }

    getCurrentTexture(): GPUTexture {
        const configuration = this.#configuration;
        if (!configuration) throw new Error("headless canvas context is not configured");
        this.#texture ??= configuration.device.createTexture({
            label: "headless-canvas-current-texture",
            size: [this.#width, this.#height],
            format: configuration.format,
            usage:
                (configuration.usage ?? GPUTextureUsage.RENDER_ATTACHMENT) |
                GPUTextureUsage.COPY_SRC,
        });
        return this.#texture;
    }
}
