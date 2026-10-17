import { expect, test } from "playwright/test";

test("the built gallery presents its canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await expect(page.locator("canvas")).toBeVisible();
    await expect(page.locator("nav")).toHaveCount(0);
    expect(errors).toEqual([]);
});

test("world names cover pixels within separate projected boxes above the spheres", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4186/tests/stepped.html?captures");
    await page.waitForFunction(() => (window as any).ready);
    const boxes: { names: number[][]; spheres: number[][] } = await page.evaluate(() =>
        (window as any).boxes(),
    );
    const overlaps = (a: number[], b: number[]) =>
        a[0]! < b[2]! && a[2]! > b[0]! && a[1]! < b[3]! && a[3]! > b[1]!;
    const ink: number[] = await page.evaluate(
        (boxes) => (window as any).nameInk(boxes),
        boxes.names,
    );
    for (const [i, box] of boxes.names.entries()) {
        expect(box[0]).toBeGreaterThan(0);
        expect(box[1]).toBeGreaterThan(0);
        expect(box[2]).toBeLessThan(1280);
        for (const other of [...boxes.names.slice(i + 1), ...boxes.spheres])
            expect(overlaps(box, other)).toBe(false);
        expect(ink[i]).toBeGreaterThan(10);
    }
});

test("stepped draws show blended glass, repeating wood, animated and row-coloured checker pixels, wave silhouette and registered background", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const skippedDraws: string[] = [];
    page.on("console", (message) => {
        if (message.type() === "warning" && message.text().startsWith("standard: draw"))
            skippedDraws.push(message.text());
    });
    await page.goto("http://127.0.0.1:4186/tests/stepped.html");
    await page.waitForFunction(() => (window as any).ready);
    console.log("materials adapter", await page.evaluate(() => (window as any).adapter));
    const aim = (index: number) => page.evaluate((i) => (window as any).aim(i), index);
    const pixels = (): Promise<number[]> => page.evaluate(() => (window as any).pixels());
    const changed = (a: number[], b: number[], x0 = 90, y0 = 25, x1 = 230, y1 = 150) => {
        let count = 0;
        for (let y = y0; y < y1; y++)
            for (let x = x0; x < x1; x++) {
                const i = (y * 320 + x) * 4;
                if (Math.max(...[0, 1, 2].map((c) => Math.abs(a[i + c]! - b[i + c]!))) > 12)
                    count++;
            }
        return count;
    };
    await aim(3);
    const glass = await pixels();
    await page.evaluate(() => (window as any).hide(3, true));
    const behind = await pixels();
    await page.evaluate(() => {
        (window as any).hide(3, false);
        (window as any).glassOpaque();
    });
    const opaque = await pixels();
    const center = (a: number[]) => a.slice((90 * 320 + 160) * 4, (90 * 320 + 160) * 4 + 3);
    console.log("glass, behind, opaque", center(glass), center(behind), center(opaque));
    expect(changed(glass, behind)).toBeGreaterThan(100);
    expect(changed(glass, opaque)).toBeGreaterThan(100);
    for (let c = 0; c < 3; c++)
        expect(center(glass)[c]!).toBeGreaterThanOrEqual(
            Math.min(center(behind)[c]!, center(opaque)[c]!) - 3,
        );
    for (let c = 0; c < 3; c++)
        expect(center(glass)[c]!).toBeLessThanOrEqual(
            Math.max(center(behind)[c]!, center(opaque)[c]!) + 3,
        );
    await aim(4);
    const wood = await pixels();
    await page.evaluate(() => (window as any).woodUnlit(true));
    const tiled = await pixels();
    // The central wood patch has both UV coordinates above 1. Without lighting, clamp is flat.
    const grain = new Set<string>();
    for (let y = 80; y < 100; y++)
        for (let x = 150; x < 170; x++) {
            const i = (y * 320 + x) * 4;
            grain.add(
                tiled
                    .slice(i, i + 3)
                    .map((c) => Math.round(c / 4))
                    .join(","),
            );
        }
    expect(grain.size, "wood grain beyond [0, 1] must repeat rather than clamp").toBeGreaterThan(
        30,
    );
    await page.evaluate(() => (window as any).woodOff());
    expect(changed(wood, await pixels())).toBeGreaterThan(500);
    await aim(5);
    const checker = await pixels();
    await page.evaluate(() => (window as any).advance(0.37));
    const animatedChecker = await pixels();
    expect(changed(checker, animatedChecker)).toBeGreaterThan(500);
    await page.evaluate(() => (window as any).checkerBlue());
    expect(changed(animatedChecker, await pixels())).toBeGreaterThan(500);
    await aim(6);
    const wave = await pixels();
    await page.evaluate(() => (window as any).advance(0.48));
    const moved = await pixels();
    // The upper background is uniform: membership changes, not lit interior shading.
    const background = wave.slice(0, 3);
    const isBackground = (a: number[], i: number) =>
        [0, 1, 2].every((c) => Math.abs(a[i + c]! - background[c]!) < 3);
    let silhouette = 0;
    for (let y = 20; y < 140; y++)
        for (let x = 100; x < 220; x++) {
            const i = (y * 320 + x) * 4;
            if (isBackground(wave, i) !== isBackground(moved, i)) silhouette++;
        }
    expect(silhouette).toBeGreaterThan(40);
    // Declared linear backdrop, encoded through the default presentation.
    expect(background[0]).toBeGreaterThanOrEqual(25);
    expect(background[0]).toBeLessThanOrEqual(35);
    expect(background[1]).toBeGreaterThanOrEqual(23);
    expect(background[1]).toBeLessThanOrEqual(32);
    expect(background[2]).toBeGreaterThanOrEqual(21);
    expect(background[2]).toBeLessThanOrEqual(30);
    expect(skippedDraws).toEqual([]);
    expect(errors).toEqual([]);
});
