export const SCENE_UNIFORM_SIZE = 352;
export const SKY_UNIFORM_SIZE = 192;
export const Z_FORMAT: GPUTextureFormat = "depth24plus";
export const DEPTH_FORMAT: GPUTextureFormat = "r32float";
export const MASK_FORMAT: GPUTextureFormat = "r8unorm";
export const EID_FORMAT: GPUTextureFormat = "r32uint";
export const COLOR_FORMAT: GPUTextureFormat = "rgba16float";

export function createSceneBuffer(device: GPUDevice): GPUBuffer {
    return device.createBuffer({
        label: "scene",
        size: SCENE_UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

export function createSkyBuffer(device: GPUDevice): GPUBuffer {
    return device.createBuffer({
        label: "sky",
        size: SKY_UNIFORM_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
}

function ensureLazy(
    device: GPUDevice,
    name: string,
    format: GPUTextureFormat,
    usage: number,
    width: number,
    height: number,
    textures: Map<string, GPUTexture>,
    textureViews: Map<string, GPUTextureView>,
): void {
    const existing = textures.get(name);
    if (
        existing &&
        existing.width === width &&
        existing.height === height &&
        existing.usage === usage
    )
        return;
    existing?.destroy();
    const tex = device.createTexture({
        label: name,
        size: { width, height },
        format,
        usage,
    });
    textures.set(name, tex);
    textureViews.set(name, tex.createView());
}

export function ensureTextures(
    device: GPUDevice,
    width: number,
    height: number,
    textures: Map<string, GPUTexture>,
    textureViews: Map<string, GPUTextureView>,
    needsDepth: boolean,
    needsMask: boolean,
): void {
    const existing = textures.get("color");
    const sizeChanged = !existing || existing.width !== width || existing.height !== height;

    if (sizeChanged) {
        existing?.destroy();
        textures.get("eid")?.destroy();
        textures.get("z")?.destroy();

        const color = device.createTexture({
            label: "color",
            size: { width, height },
            format: COLOR_FORMAT,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        });

        const eid = device.createTexture({
            label: "eid",
            size: { width, height },
            format: EID_FORMAT,
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        const z = device.createTexture({
            label: "z",
            size: { width, height },
            format: Z_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        textures.set("color", color);
        textureViews.set("color", color.createView());
        textures.set("eid", eid);
        textureViews.set("eid", eid.createView());
        textures.set("z", z);
        textureViews.set("z", z.createView());
    }

    const dw = needsDepth ? width : 1;
    const dh = needsDepth ? height : 1;
    ensureLazy(
        device,
        "depth",
        DEPTH_FORMAT,
        GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING,
        dw,
        dh,
        textures,
        textureViews,
    );

    const mw = needsMask ? width : 1;
    const mh = needsMask ? height : 1;
    ensureLazy(
        device,
        "mask",
        MASK_FORMAT,
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        mw,
        mh,
        textures,
        textureViews,
    );
}
