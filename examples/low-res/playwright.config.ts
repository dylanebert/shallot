import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: "tests",
    testIgnore: "**/frames/**",
    ...BROWSER_CONFIG,
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4324" },
    webServer: {
        command: "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4324 --strictPort",
        url: "http://127.0.0.1:4324",
        ...WEB_SERVER_CONFIG,
    },
});
