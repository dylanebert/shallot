import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
    use: {
        ...BROWSER_CONFIG.use,
        baseURL: "http://127.0.0.1:4265",
    },
    webServer: [
        {
            command:
                "bunx vite build && bunx vite preview --host 127.0.0.1 --port 4265 --strictPort",
            url: "http://127.0.0.1:4265",
            ...WEB_SERVER_CONFIG,
        },
        // The stepped page under tests/ is served by Vite's dev server, which compiles it as the build
        // compiles the app's own page; the build holds only the app's page.
        {
            command: "bunx vite --host 127.0.0.1 --port 4266 --strictPort",
            url: "http://127.0.0.1:4266/tests/stepped.html",
            ...WEB_SERVER_CONFIG,
        },
    ],
} satisfies PlaywrightTestConfig;

export default config;
