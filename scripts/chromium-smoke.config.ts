import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "playwright/test";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "./chromium";

const subject = fileURLToPath(new URL("./chromium-smoke/", import.meta.url));

export default {
    ...BROWSER_CONFIG,
    testDir: subject,
    testMatch: "webgpu-smoke.playwright.ts",
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4176",
    },
    webServer: {
        command: "bun server.ts",
        cwd: subject,
        url: "http://127.0.0.1:4176",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
