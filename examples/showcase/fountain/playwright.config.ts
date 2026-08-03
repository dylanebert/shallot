import { defineConfig } from "@playwright/test";

const PORT = 3114;
const URL = `http://localhost:${PORT}`;

export default defineConfig({
    testDir: "./test",
    fullyParallel: false,
    workers: 1,
    reporter: [["list"]],
    timeout: 120_000,
    webServer: {
        command: `bunx shallot dev . --port ${PORT} --strict-port`,
        url: URL,
        reuseExistingServer: !process.env.CI,
        timeout: 180_000,
    },
    use: {
        baseURL: URL,
        channel: "chrome",
        launchOptions: {
            args: [
                "--enable-unsafe-webgpu",
                "--enable-features=WebGPUDeveloperFeatures",
                "--enable-dawn-features=allow_unsafe_apis",
            ],
        },
    },
});
