import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    ...BROWSER_CONFIG,
    testDir: ".",
    globalSetup: fileURLToPath(new URL("../browser-setup.ts", import.meta.url)),
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4402" },
    webServer: {
        command: "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4402 --strictPort",
        cwd: fileURLToPath(new URL("../..", import.meta.url)),
        url: "http://127.0.0.1:4402",
        ...WEB_SERVER_CONFIG,
    },
});
