import { existsSync } from "node:fs";

export function assertChromiumInstalled(executablePath: string): void {
    if (!existsSync(executablePath))
        throw new Error(
            "Playwright Chromium is missing. Run `bun run browser:install`, then `bun run test:browser`.",
        );
}
