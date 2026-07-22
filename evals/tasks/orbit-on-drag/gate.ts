import { test } from "@playwright/test";
import { type Assertion, boot, diff, emit, shot } from "../../harness/lib";

// Positive claim, isolating causation: with no input the view holds still, and a horizontal drag
// changes it. Comparing an idle interval against a drag interval separates orbit-on-drag from a scene
// that just animates on its own.
test("orbit-on-drag", async ({ page }) => {
    test.setTimeout(80_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        const before = await shot(page);
        await page.waitForTimeout(700);
        const idle = await shot(page);
        const idleDelta = diff(before, idle);

        const box = await page.locator("canvas").first().boundingBox();
        if (box) {
            const cy = box.y + box.height / 2;
            await page.mouse.move(box.x + box.width * 0.3, cy);
            await page.mouse.down();
            await page.mouse.move(box.x + box.width * 0.7, cy, { steps: 12 });
            await page.mouse.up();
        }
        await page.waitForTimeout(300);
        const dragged = await shot(page);
        const dragDelta = diff(idle, dragged);

        // derived thresholds (measured 2026-07-13 on the baseline orbit project): a settled idle
        // interval diffs exactly 0.00 (bit-static render), so idle < 0.5 is floor + epsilon; a real
        // orbit drag measures mean-abs 1.77 full-frame (the object is a small fraction of the frame,
        // centre region 215,174,136 → 154,127,103), so drag > 1 sits between floor and measurement.
        // The 3x-idle ratio keeps the drag signal causal on a scene with residual drift.
        a.push({
            name: "view holds still without input",
            ok: idleDelta < 0.5,
            detail: `idle delta ${idleDelta.toFixed(2)}`,
        });
        a.push({
            name: "dragging orbits the camera",
            ok: dragDelta > 1 && dragDelta > idleDelta * 3,
            detail: `drag delta ${dragDelta.toFixed(2)} vs idle ${idleDelta.toFixed(2)}`,
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
