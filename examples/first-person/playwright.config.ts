import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export const config = {
    testDir: ".",
    testIgnore: "**/tests/input/**",
    ...BROWSER_CONFIG,
    globalSetup: "./tests/browser-setup.ts",
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4175",
    },
    webServer: [
        {
            // Vite directly, not package scripts: in the engine repo the example has no manifest, and `bun run`
            // would reach the engine's own scripts.
            command:
                "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4175 --strictPort",
            url: "http://127.0.0.1:4175",
            ...WEB_SERVER_CONFIG,
        },
        {
            command: "bunx vite --host 127.0.0.1 --port 4176 --strictPort",
            url: "http://127.0.0.1:4176/tests/stepped.html",
            ...WEB_SERVER_CONFIG,
        },
    ],
} satisfies PlaywrightTestConfig;

export default config;
