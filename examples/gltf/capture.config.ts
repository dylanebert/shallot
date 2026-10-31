import config from "./playwright.config";

// GLTF_SAMPLE_VIEWER=<built v1.0.10 checkout> bunx playwright test -c examples/gltf/capture.config.ts
export default {
    ...config,
    testDir: "../../diagnostics/gltf-look",
    workers: 1,
    use: { ...config.use, viewport: { width: 960, height: 720 }, deviceScaleFactor: 1 },
    webServer: [
        config.webServer,
        {
            ...config.webServer,
            command: "bun ../../diagnostics/gltf-look/serve.ts",
            url: "http://127.0.0.1:4176",
        },
    ],
    testMatch: "**/*.capture.ts",
};
