import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    ...BROWSER_CONFIG,
    testDir: ".",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4426" },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4426 --strictPort",
        url: "http://127.0.0.1:4426",
        ...WEB_SERVER_CONFIG,
    },
});
