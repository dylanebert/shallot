import { test } from "@playwright/test";
import { type Assertion, boot, emit, type Rgb, region, shot } from "../../harness/lib";

// Positive claim: a number key paints the cube, and the choice survives a page reload with no further
// input. To keep persistence from being faked by a lucky default, the gate picks a target colour that
// differs from whatever the scene starts on — only a project that actually saves the choice can show
// that colour again after a fresh load.
const isRed = (c: Rgb): boolean => c.r > 90 && c.r > c.g + 40 && c.r > c.b + 40;
const isBlue = (c: Rgb): boolean => c.b > 90 && c.b > c.r + 30 && c.b > c.g + 20;

test("persist-color", async ({ page }) => {
    test.setTimeout(120_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        await page.locator("canvas").first().click();
        const start = region(await shot(page), 0.4, 0.4, 0.6, 0.6);

        // aim for a colour the scene isn't already showing: if a reload returns to a fixed default
        // colour, only genuine persistence can match the target — coincidence can't
        const target = isBlue(start)
            ? { key: "Digit1", name: "red", reads: isRed }
            : { key: "Digit3", name: "blue", reads: isBlue };

        await page.keyboard.press(target.key);
        await page.waitForTimeout(500);
        const painted = region(await shot(page), 0.4, 0.4, 0.6, 0.6);
        a.push({
            name: `pressing the key paints the cube ${target.name}`,
            ok: target.reads(painted),
            detail: `centre rgb ${painted.r | 0},${painted.g | 0},${painted.b | 0}`,
        });

        // a fresh navigation to the same origin keeps localStorage — the colour must come back with no
        // key press. boot() re-waits for a settled (or animated) frame before the read.
        const reboot = await boot(page);
        const restored = region(await shot(page), 0.4, 0.4, 0.6, 0.6);
        a.push({
            name: `the cube is still ${target.name} after a reload`,
            ok: reboot.rendered && target.reads(restored),
            detail: `centre rgb ${restored.r | 0},${restored.g | 0},${restored.b | 0}`,
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
