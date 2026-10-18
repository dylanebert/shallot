import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";
export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    testIgnore: "**/tests/presentation/**",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4193" },
    webServer: {
        command: "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4193 --strictPort",
        url: "http://127.0.0.1:4193",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
