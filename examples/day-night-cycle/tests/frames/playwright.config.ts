import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    testDir: ".",
    ...BROWSER_CONFIG,
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4266 --strictPort",
        url: "http://127.0.0.1:4266/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
