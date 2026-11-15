import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    testDir: ".",
    testIgnore: [
        "**/tests/stepped/**",
        "**/tests/stepped-controls/**",
        "**/tests/panels/**",
        "**/tests/hints/**",
        "**/tests/layout/**",
    ],
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4352",
    },
    webServer: {
        command: "bunx vite --host 127.0.0.1 --port 4352 --strictPort",
        url: "http://127.0.0.1:4352",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
