import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "playwright/test";
import { CHROMIUM_USE } from "../../scripts/chromium";

const subject = fileURLToPath(new URL(".", import.meta.url));

export const config = {
    testDir: ".",
    testMatch: "**/*.e2e.ts",
    timeout: 20_000,
    globalTimeout: 22_000,
    fullyParallel: false,
    workers: 1,
    reporter: "list",
    use: {
        browserName: "chromium",
        ...CHROMIUM_USE,
        baseURL: "http://127.0.0.1:4175",
        viewport: { width: 1280, height: 720 },
        deviceScaleFactor: 1,
    },
    webServer: {
        command: `bunx vite build "${subject}" --config "${subject}vite.config.ts" && bunx vite preview "${subject}" --config "${subject}vite.config.ts" --host 127.0.0.1 --port 4175 --strictPort`,
        url: "http://127.0.0.1:4175",
        reuseExistingServer: false,
        timeout: 120_000,
    },
} satisfies PlaywrightTestConfig;

export default config;
