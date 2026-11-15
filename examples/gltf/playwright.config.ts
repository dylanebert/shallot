import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: ".",
    testIgnore: [
        "**/tests/stepped/**",
        "**/tests/panels/**",
        "**/tests/materials/**",
        "**/tests/crowd/**",
    ],
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4351",
    },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4351 --strictPort",
        url: "http://127.0.0.1:4351",
        ...WEB_SERVER_CONFIG,
    },
});
