import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    ...BROWSER_CONFIG,
    testDir: "tests",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4391" },
    webServer: {
        ...WEB_SERVER_CONFIG,
        command: "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4391 --strictPort",
        url: "http://127.0.0.1:4391",
    },
} satisfies PlaywrightTestConfig;
