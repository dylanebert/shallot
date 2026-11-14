import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: ".",
    testIgnore: [
        "**/tests/stepped/**",
        "**/tests/stepped-controls/**",
        "**/tests/panels/**",
        "**/tests/hints/**",
    ],
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4352",
    },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4352 --strictPort",
        url: "http://127.0.0.1:4352",
        ...WEB_SERVER_CONFIG,
    },
});
