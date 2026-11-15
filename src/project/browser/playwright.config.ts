import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4207",
    },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4207 --strictPort",
        cwd: fileURLToPath(new URL(".", import.meta.url)),
        url: "http://127.0.0.1:4207/frames.html",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
