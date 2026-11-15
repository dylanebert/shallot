import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

// Run with SHALLOT_CAPTURE_DIR=<directory> bunx playwright test -c scripts/phone/capture.config.ts.
export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    timeout: 0,
    globalTimeout: 0,
    testMatch: "*.capture.ts",
    metadata: { captureDirectory: process.env.SHALLOT_CAPTURE_DIR ?? ".artifacts/phone" },
    use: {
        ...BROWSER_CONFIG.use,
        viewport: { width: 390, height: 844 },
        hasTouch: true,
    },
} satisfies PlaywrightTestConfig;
