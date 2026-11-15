import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    testDir: "tests",
    ...BROWSER_CONFIG,
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4300" },
    webServer: [
        {
            command:
                "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4300 --strictPort",
            url: "http://127.0.0.1:4300",
            ...WEB_SERVER_CONFIG,
        },
        {
            command: "bunx vite --host 127.0.0.1 --port 4301 --strictPort",
            url: "http://127.0.0.1:4301/tests/stepped.html",
            ...WEB_SERVER_CONFIG,
        },
    ],
} satisfies PlaywrightTestConfig;
