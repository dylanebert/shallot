import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4353", deviceScaleFactor: 1 },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4353 --strictPort",
        url: "http://127.0.0.1:4353",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;

export default config;
