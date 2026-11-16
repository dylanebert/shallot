import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4211" },
    projects: [
        { name: "chromium", use: { browserName: "chromium" } },
        { name: "webkit", use: { browserName: "webkit", channel: "", launchOptions: {} } },
    ],
    webServer: {
        ...WEB_SERVER_CONFIG,
        command: "bunx vite --host 127.0.0.1 --port 4211 --strictPort",
        cwd: fileURLToPath(new URL(".", import.meta.url)),
        url: "http://127.0.0.1:4211",
    },
} satisfies PlaywrightTestConfig;
