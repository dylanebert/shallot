import { resolve } from "node:path";
import { runBrowserCheck } from "@dylanebert/shallot/harness/browser";
import { check } from "@dylanebert/shallot/harness/check";

type InputCheck = {
    held(): boolean;
    observations: { focused?: boolean; unfocused?: boolean };
};

declare global {
    interface Window {
        __inputCheck?: InputCheck;
    }
}

check(
    "a browser key press reaches the State-scoped input record only while the canvas is focused",
    {
        claim: "the browser input adapter fails to record a real key press on the focused canvas or accepts it after focus leaves the canvas",
        size: "integration",
        requires: ["browser"],
        subject: [
            "src/core/input/index.ts",
            "src/core/input/pages/browser-input.html",
            "src/core/input/pages/browser-input.ts",
        ],
        budget: 20_000,
    },
    () =>
        runBrowserCheck(resolve(import.meta.dir, "pages/browser-input.html"), async (page) => {
            await page.locator("#canvas").click();
            await page.locator("#canvas").focus();
            if (
                !(await page
                    .locator("#canvas")
                    .evaluate((canvas) => document.activeElement === canvas))
            )
                throw new Error("canvas did not receive browser focus");
            await page.keyboard.down("w");
            await page.evaluate(() => {
                const input = window.__inputCheck!;
                input.observations.focused = input.held();
            });
            await page.keyboard.up("w");

            await page.locator("#outside").click();
            await page.locator("#outside").focus();
            if (
                !(await page
                    .locator("#outside")
                    .evaluate((button) => document.activeElement === button))
            )
                throw new Error("focus did not leave the canvas");
            await page.keyboard.down("w");
            await page.evaluate(() => {
                const input = window.__inputCheck!;
                input.observations.unfocused = input.held();
            });
            await page.keyboard.up("w");
        }),
);
