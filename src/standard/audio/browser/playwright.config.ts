import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

const subject = fileURLToPath(new URL(".", import.meta.url)).replaceAll("\\", "/");

export default {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4173",
    },
    webServer: {
        command: `bunx vite --config "${subject}vite.config.ts" --host 127.0.0.1 --port 4173 --strictPort`,
        url: "http://127.0.0.1:4173",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
