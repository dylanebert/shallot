import { fileURLToPath } from "node:url";
import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";
export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4194" },
    webServer: {
        command:
            "bunx vite --config tests/presentation/vite.config.ts --host 127.0.0.1 --port 4194 --strictPort",
        cwd: fileURLToPath(new URL("../..", import.meta.url)),
        url: "http://127.0.0.1:4194/tests/stepped.html",
        ...WEB_SERVER_CONFIG,
    },
} satisfies PlaywrightTestConfig;
