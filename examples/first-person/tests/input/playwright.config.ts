import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";
export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    globalSetup: fileURLToPath(new URL("../browser-setup.ts", import.meta.url)),
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4175" },
    webServer: {
        command:
            "bunx vite build --config tests/input/vite.config.ts && bunx vite preview --config tests/input/vite.config.ts --host 127.0.0.1 --port 4175 --strictPort",
        cwd: fileURLToPath(new URL("../..", import.meta.url)),
        url: "http://127.0.0.1:4175",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
