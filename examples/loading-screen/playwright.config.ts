import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "playwright/test";
import { CHROMIUM_ARGS } from "../../scripts/chromium";

const subject = fileURLToPath(new URL(".", import.meta.url));

export const config = {
    testDir: ".",
    testMatch: "**/*.e2e.ts",
    timeout: 20_000,
    globalTimeout: 9_000,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        launchOptions: { args: CHROMIUM_ARGS },
        baseURL: "http://127.0.0.1:4173",
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
    webServer: {
        command: `bunx vite build "${subject}" --config "${subject}vite.config.ts" && bunx vite preview "${subject}" --config "${subject}vite.config.ts" --host 127.0.0.1 --port 4173 --strictPort`,
        url: "http://127.0.0.1:4173",
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
    },
} satisfies PlaywrightTestConfig;

export default config;
