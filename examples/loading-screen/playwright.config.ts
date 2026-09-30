import { fileURLToPath } from "node:url";
import type { PlaywrightTestConfig } from "playwright/test";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "../../scripts/chromium";

const subject = fileURLToPath(new URL(".", import.meta.url));

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4173",
    },
    webServer: {
        command: `bunx vite build "${subject}" --config "${subject}vite.config.ts" && bunx vite preview "${subject}" --config "${subject}vite.config.ts" --host 127.0.0.1 --port 4173 --strictPort`,
        url: "http://127.0.0.1:4173",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;

export default config;
