import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: ".",
    ...BROWSER_CONFIG,
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4409 --strictPort",
        url: "http://127.0.0.1:4409/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
});
