import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import { defineConfig } from "playwright/test";

const example = process.env.TOUCH_EXAMPLE ?? "first-person";
export default defineConfig({
    ...BROWSER_CONFIG,
    testDir: ".",
    testMatch: "first-person-touch.e2e.ts",
    workers: 1,
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4188" },
    webServer: {
        ...WEB_SERVER_CONFIG,
        command: "bunx vite --host 127.0.0.1 --port 4188 --strictPort",
        cwd: new URL(`../examples/${example}`, import.meta.url).pathname,
        url: "http://127.0.0.1:4188/tests/touch.html",
    },
});
