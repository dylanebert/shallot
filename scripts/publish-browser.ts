import { existsSync } from "node:fs";
import { BROWSER_CONFIG } from "@dylanebert/shallot/testing";
import { chromium } from "playwright";

/** Tests consume the browser installed by `bunx playwright install chromium`, never install it. */
export function publishTestChromium() {
    const executablePath = process.env.SHALLOT_TEST_CHROMIUM ?? chromium.executablePath();
    if (!existsSync(executablePath))
        throw new Error(
            `Missing Playwright Chromium: ${executablePath}. Run bunx playwright install chromium.`,
        );
    return chromium.launch({
        ...BROWSER_CONFIG.use.launchOptions,
        channel: BROWSER_CONFIG.use.channel,
        executablePath,
    });
}
