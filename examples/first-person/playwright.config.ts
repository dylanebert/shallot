import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: ".",
    testIgnore: ["**/tests/input/**", "**/tests/touch/**"],
    ...BROWSER_CONFIG,
    globalSetup: "./tests/browser-setup.ts",
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4312",
    },
    webServer: [
        {
            // Vite directly, not package scripts: in the engine repo the example has no manifest, and `bun run`
            // would reach the engine's own scripts.
            command:
                "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4312 --strictPort",
            url: "http://127.0.0.1:4312",
            ...WEB_SERVER_CONFIG,
        },
        {
            command: "bunx vite --host 127.0.0.1 --port 4313 --strictPort",
            url: "http://127.0.0.1:4313/tests/stepped.html",
            ...WEB_SERVER_CONFIG,
        },
    ],
});
