import type { World } from "../../engine";
import { probeTexture } from "../../engine/runtime";

/** Copies the real mask before the first flood, and exposes the production final seed field. */
export function captureSeeds(world: World, camera: number, width: number, height: number) {
    const { device } = world.gpu;
    const snapshot = device.createTexture({
        size: [width, height],
        format: "rg16uint",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });
    world.own(snapshot);
    const views = new WeakMap<GPUTextureView, GPUTexture>();
    const createTexture = device.createTexture.bind(device);
    const createEncoder = device.createCommandEncoder.bind(device);
    let finalSeed: GPUTexture | undefined;
    let copies = 0;
    device.createTexture = (descriptor) => {
        if (!descriptor.label?.startsWith("outline-seed")) return createTexture(descriptor);
        const texture = createTexture({
            ...descriptor,
            usage: descriptor.usage | GPUTextureUsage.COPY_SRC,
        });
        // The production seven-slot ladder always finishes in seed B.
        if (descriptor.label === `outline-seedB-${camera}`) finalSeed = texture;
        const createView = texture.createView.bind(texture);
        texture.createView = (descriptor) => {
            const view = createView(descriptor);
            views.set(view, texture);
            return view;
        };
        return texture;
    };
    device.createCommandEncoder = (descriptor) => {
        const encoder = createEncoder(descriptor);
        const begin = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (descriptor) => {
            const pass = begin(descriptor);
            if (descriptor.label !== `outline-mask/${camera}`) return pass;
            const source = views.get(
                Array.from(descriptor.colorAttachments)[0]!.view as GPUTextureView,
            )!;
            const end = pass.end.bind(pass);
            pass.end = () => {
                end();
                encoder.copyTextureToTexture({ texture: source }, { texture: snapshot }, [
                    width,
                    height,
                ]);
                copies++;
            };
            return pass;
        };
        return encoder;
    };
    return {
        get copies() {
            return copies;
        },
        async read() {
            if (!finalSeed || copies === 0) throw new Error("No outline mask was captured");
            return {
                mask: new Uint16Array((await probeTexture(world, snapshot)).bytes as ArrayBuffer),
                nearest: new Uint16Array(
                    (await probeTexture(world, finalSeed)).bytes as ArrayBuffer,
                ),
            };
        },
        restore() {
            device.createTexture = createTexture;
            device.createCommandEncoder = createEncoder;
        },
    };
}
