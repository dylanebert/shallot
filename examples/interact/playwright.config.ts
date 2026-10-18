import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";
export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4186" },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4186 --strictPort",
        url: "http://127.0.0.1:4186/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
