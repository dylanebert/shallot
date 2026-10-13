import { existsSync } from "node:fs";

export function assertChromiumInstalled(executablePath: string): void {
    if (!existsSync(executablePath))
        throw new Error(
            "Playwright Chromium is missing. Run `bunx playwright install chromium`, then `bunx playwright test`.",
        );
}
