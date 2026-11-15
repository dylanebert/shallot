import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import type { PlaywrightTestConfig } from "playwright/test";

export const config = {
    testDir: ".",
    ...BROWSER_CONFIG,
} satisfies PlaywrightTestConfig;

export default config;
