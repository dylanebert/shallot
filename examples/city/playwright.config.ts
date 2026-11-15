import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";
export default defineConfig({
    ...BROWSER_CONFIG,
    testDir: "tests",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4305" },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4305 --strictPort",
        url: "http://127.0.0.1:4305/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
});
