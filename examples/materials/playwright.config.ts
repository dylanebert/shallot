import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";
export default defineConfig({
    ...BROWSER_CONFIG,
    testDir: ".",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4326" },
    webServer: [
        {
            command:
                "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4326 --strictPort",
            url: "http://127.0.0.1:4326",
            ...WEB_SERVER_CONFIG,
        },
        {
            command: "bunx vite --host 127.0.0.1 --port 4327 --strictPort",
            url: "http://127.0.0.1:4327/tests/stepped.html",
            ...WEB_SERVER_CONFIG,
        },
    ],
});
