export const CHROMIUM_USE = {
    channel: "chromium" as const,
    launchOptions: {
        args: [
            "--enable-unsafe-webgpu",
            "--enable-features=WebGPUDeveloperFeatures",
            "--enable-webgpu-developer-features",
            "--enable-gpu",
        ],
    },
};
