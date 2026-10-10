import { CHROMIUM_USE } from "../chromium";

export const CHROMIUM_VARIANTS = [
    {
        id: "a",
        name: "current",
        args: [...CHROMIUM_USE.launchOptions.args],
    },
    {
        id: "b",
        name: "current plus unsafe SwiftShader",
        args: [...CHROMIUM_USE.launchOptions.args, "--enable-unsafe-swiftshader"],
    },
    {
        id: "c",
        name: "unsafe WebGPU only",
        args: CHROMIUM_USE.launchOptions.args.filter(
            (arg) =>
                arg !== "--enable-gpu" &&
                arg !== "--enable-features=WebGPUDeveloperFeatures" &&
                arg !== "--enable-webgpu-developer-features",
        ),
    },
    {
        id: "d",
        name: "documented Vulkan with lavapipe",
        args: [
            "--enable-unsafe-webgpu",
            "--use-angle=vulkan",
            "--enable-features=Vulkan",
            "--disable-vulkan-surface",
        ],
    },
] as const;

export type ChromiumVariant = (typeof CHROMIUM_VARIANTS)[number]["id"];

export function chromiumVariant(id: string): (typeof CHROMIUM_VARIANTS)[number] {
    const variant = CHROMIUM_VARIANTS.find((item) => item.id === id);
    if (!variant) throw new Error(`Unknown Chromium WebGPU variant: ${id}`);
    return variant;
}
