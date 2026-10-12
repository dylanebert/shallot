import type { PlaywrightTestConfig } from "playwright/test";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "./chromium";

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
    globalSetup: "./tests/browser-setup.ts",
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4175",
    },
    webServer: {
        command: `bun run build && bun run preview --host 127.0.0.1 --port 4175 --strictPort`,
        url: "http://127.0.0.1:4175",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;

export default config;
