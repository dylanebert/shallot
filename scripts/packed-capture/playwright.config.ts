import type { PlaywrightTestConfig } from "playwright/test";
import { BROWSER_CONFIG } from "../chromium";

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
} satisfies PlaywrightTestConfig;

export default config;
