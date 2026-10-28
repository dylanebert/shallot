import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";
export default {
    ...BROWSER_CONFIG,
    testDir: "tests",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4210" },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4210 --strictPort",
        url: "http://127.0.0.1:4210/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
