import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export const config = {
    testDir: ".",
    testIgnore: "**/tests/stepped/**",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4314",
    },
    webServer: {
        command: "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4314 --strictPort",
        url: "http://127.0.0.1:4314",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;

export default config;
