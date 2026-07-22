import { test } from "@playwright/test";
import { type Assertion, boot, emit, region, shot } from "../../harness/lib";

// Positive claim: the cube starts white and a spacebar press turns it green. Reads the centre color
// before and after synthesizing the keystroke — the change must follow the input.
test("color-on-key", async ({ page }) => {
    test.setTimeout(80_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        await page.locator("canvas").first().click();
        const before = region(await shot(page), 0.4, 0.4, 0.6, 0.6);
        a.push({
            name: "cube starts white",
            ok: before.r > 150 && before.g > 150 && before.b > 150,
            detail: `centre rgb ${before.r | 0},${before.g | 0},${before.b | 0}`,
        });

        await page.keyboard.press("Space");
        await page.waitForTimeout(500);
        const after = region(await shot(page), 0.4, 0.4, 0.6, 0.6);
        a.push({
            name: "spacebar turns it green",
            ok: after.g > 110 && after.g > after.r + 40 && after.g > after.b + 40,
            detail: `centre rgb ${after.r | 0},${after.g | 0},${after.b | 0}`,
        });
    }

    emit({
        ok: booted && rendered && a.every((x) => x.ok),
        booted,
        rendered,
        assertions: a,
        errors: errors.length ? errors : undefined,
    });
});
