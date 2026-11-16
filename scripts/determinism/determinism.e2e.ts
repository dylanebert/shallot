import { expect, test } from "playwright/test";
import { actions } from "./composition.fixture";
import { GAMEPLAY_HASH } from "./golden.fixture";

for (const workers of [1, 2, 4]) {
    test(`recorded stream matches the gameplay golden with ${workers} workers`, async ({
        page,
        browser,
    }) => {
        await page.goto("/");
        const result = await page.evaluate(async (count) => {
            const { replay } = await import("./main");
            return replay(count);
        }, workers);
        console.log(
            JSON.stringify({
                browser: browser.browserType().name(),
                version: browser.version(),
                ...result,
            }),
        );
        expect(result.isolated).toBe(true);
        expect(result.artifact).toBe("shared");
        expect(result.workers).toBe(workers);
        expect(result.ticks).toBe(actions.length);
        expect(result.hash).toBe(GAMEPLAY_HASH.toString());
    });
}
