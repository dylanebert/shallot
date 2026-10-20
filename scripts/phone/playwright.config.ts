import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export default {
    ...BROWSER_CONFIG,
    testDir: ".",
    use: {
        ...BROWSER_CONFIG.use,
        viewport: { width: 390, height: 844 },
        hasTouch: true,
    },
} satisfies PlaywrightTestConfig;
