import { BROWSER_CONFIG, WEB_SERVER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    ...BROWSER_CONFIG,
    testDir: "tests",
    use: { ...BROWSER_CONFIG.use, baseURL: "http://127.0.0.1:4430" },
    webServer: [
        {
            command: "bunx vite --host 127.0.0.1 --port 4430 --strictPort",
            url: "http://127.0.0.1:4430",
            ...WEB_SERVER_CONFIG,
        },
        {
            command: "bun server.ts --port 4431 --host 127.0.0.1",
            url: "http://127.0.0.1:4431",
            // The socket-only endpoint answers HTTP with 426; wait for its listening message.
            wait: { stdout: /Multiplayer server: ws:\/\/localhost:4431/ },
            ...WEB_SERVER_CONFIG,
        },
    ],
} satisfies PlaywrightTestConfig;
