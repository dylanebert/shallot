export interface View {
    element: HTMLCanvasElement | null;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    textures: Map<string, GPUTexture>;
    textureViews: Map<string, GPUTextureView>;
    width: number;
    height: number;
    observer: ResizeObserver | null;
    dirty: boolean;
}

function syncCanvasSize(canvas: HTMLCanvasElement): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
}

export function createWebView(element: HTMLCanvasElement, device: GPUDevice): View {
    const context = element.getContext("webgpu") as unknown as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    element.style.imageRendering = "pixelated";
    context.configure({
        device,
        format,
        alphaMode: "premultiplied",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    syncCanvasSize(element);

    const view: View = {
        element,
        context,
        format,
        textures: new Map(),
        textureViews: new Map(),
        width: element.width,
        height: element.height,
        observer: null,
        dirty: true,
    };

    const observer = new ResizeObserver(() => {
        syncCanvasSize(element);
        view.dirty = true;
    });
    observer.observe(element);
    view.observer = observer;

    return view;
}
