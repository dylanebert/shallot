import type { PlaywrightTestConfig } from "playwright/test";

export const CHROMIUM_ARGS = [
    "--enable-unsafe-webgpu",
    "--enable-features=WebGPUDeveloperFeatures",
    "--enable-webgpu-developer-features",
    "--enable-gpu",
];

const config = {
    testDir: ".",
    testMatch: "**/*.e2e.ts",
    timeout: 20_000,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        launchOptions: { args: CHROMIUM_ARGS },
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
    projects: [
        {
            name: "loading-screen",
            testMatch: ["**/examples/loading-screen/**/*.e2e.ts", "**/src/core/rendering/capture.e2e.ts"],
            use: { baseURL: "http://127.0.0.1:4173" },
        },
        {
            name: "first-person-input",
            testMatch: "**/src/core/input/browser.e2e.ts",
            use: { baseURL: "http://127.0.0.1:4174" },
        },
    ],
    webServer: [
        {
            command:
                "bunx vite build examples/loading-screen --config examples/loading-screen/vite.config.ts && bunx vite preview examples/loading-screen --config examples/loading-screen/vite.config.ts --host 127.0.0.1 --port 4173 --strictPort",
            url: "http://127.0.0.1:4173",
            reuseExistingServer: !process.env.CI,
            timeout: 120_000,
        },
        {
            command:
                "bunx vite build examples/first-person --config examples/first-person/vite.config.ts && bunx vite preview examples/first-person --config examples/first-person/vite.config.ts --host 127.0.0.1 --port 4174 --strictPort",
            url: "http://127.0.0.1:4174",
            reuseExistingServer: !process.env.CI,
            timeout: 120_000,
        },
    ],
} satisfies PlaywrightTestConfig;

export default config;
