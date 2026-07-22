import { test } from "@playwright/test";
import { type Assertion, boot, diff, emit, region, shot } from "../../harness/lib";

// Positive claim: the centre of the frame is red, distinct from a dark background, and the frame
// holds still (no spin). Grades pixels, so it's blind to how the scene was authored.
test("red-box", async ({ page }) => {
    test.setTimeout(80_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        const first = await shot(page);
        const c = region(first, 0.4, 0.4, 0.6, 0.6);
        const corner = region(first, 0, 0, 0.12, 0.12);
        a.push({
            name: "centre is red",
            ok: c.r > 110 && c.r > c.g * 1.7 && c.r > c.b * 1.7,
            detail: `centre rgb ${c.r | 0},${c.g | 0},${c.b | 0}`,
        });
        a.push({
            name: "cube stands out from a dark background",
            ok: corner.r + corner.g + corner.b < c.r + c.g + c.b - 80,
            detail: `corner rgb ${corner.r | 0},${corner.g | 0},${corner.b | 0}`,
        });

        await page.waitForTimeout(1200);
        const second = await shot(page);
        a.push({
            name: "scene is static, not spinning",
            ok: diff(first, second) < 6,
            detail: `frame delta ${diff(first, second).toFixed(1)}`,
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
