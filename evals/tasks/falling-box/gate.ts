import { test } from "@playwright/test";
import { type Assertion, boot, centroidY, emit, type Rgb, region, shot } from "../../harness/lib";

// Positive claim: a blue box descends and then settles. The fall completes ~1.05s after scene build
// (measured 2026-07-13), so a settle-stable boot() returns after the transient is over — boot proves
// the app serves + settles, then a page.reload() replays the fall and the blue-pixel centroid is
// sampled as fast as screenshots allow from the moment the canvas exists. Descent shows as the
// centroid's highest observed point sitting above its resting point; settling as a stable tail.
const isBlue = (c: Rgb): boolean => c.b > 90 && c.b > c.r + 30 && c.b > c.g + 20;

test("falling-box", async ({ page }) => {
    test.setTimeout(80_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
        if (m.type() === "error") errors.push(m.text());
    });

    const { booted, rendered } = await boot(page);
    const a: Assertion[] = [];

    if (rendered) {
        await page.reload();
        await page.waitForSelector("canvas", { timeout: 20_000 });
        const t0 = Date.now();
        const ys: number[] = [];
        let lastShot = await shot(page);
        while (Date.now() - t0 < 6_000) {
            lastShot = await shot(page);
            const y = centroidY(lastShot, isBlue);
            if (y != null) ys.push(y);
        }

        const centre = region(lastShot, 0.4, 0.4, 0.6, 0.6);
        a.push({
            name: "a blue box is on screen",
            ok: ys.length > 0,
            detail:
                ys.length > 0
                    ? `${ys.length} blue samples, resting centroid y ${ys[ys.length - 1].toFixed(3)}`
                    : `no blue pixels found (centre rgb ${centre.r | 0},${centre.g | 0},${centre.b | 0})`,
        });

        const rest = ys[ys.length - 1];
        const peak = Math.min(...ys);
        a.push({
            name: "the box fell (moved down)",
            ok: ys.length > 1 && rest - peak > 0.03,
            detail: ys.length > 1 ? `descent ${(rest - peak).toFixed(3)}` : "missing samples",
        });
        // settled render is bit-static (measured 0.00 frame delta), so the last two centroid
        // samples of a resting box agree to well under a pixel; 0.01 is epsilon over that floor
        const tail = ys.slice(-2);
        a.push({
            name: "the box settled (stopped falling)",
            ok: tail.length === 2 && Math.abs(tail[1] - tail[0]) < 0.01,
            detail:
                tail.length === 2
                    ? `tail Δy ${Math.abs(tail[1] - tail[0]).toFixed(4)}`
                    : "missing samples",
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
