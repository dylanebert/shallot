import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4182",
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4182 --strictPort",
        cwd: dirname(fileURLToPath(import.meta.url)),
        url: "http://127.0.0.1:4182",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
