import { expect, type JSHandle, type Page, test } from "playwright/test";

test("the built gallery presents its canvas", async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto("/");
    await expect(page.locator("canvas")).toBeVisible();
    await expect(page.locator("nav")).toHaveCount(0);
    expect(errors).toEqual([]);
});

test.describe("one stepped gallery", () => {
    test.describe.configure({ mode: "serial" });
    let page: Page;
    const errors: string[] = [];
    const skippedDraws: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "warning" && message.text().startsWith("standard: draw"))
                skippedDraws.push(message.text());
        });
        await page.goto("http://127.0.0.1:4186/tests/stepped.html?names");
        await page.waitForFunction(() => (window as any).ready);
    });
    test.afterAll(() => page.close());
    test.afterEach(() => {
        expect(errors).toEqual([]);
        expect(skippedDraws).toEqual([]);
    });

    test("world names cover pixels within separate projected boxes above the spheres", async () => {
        const boxes: { names: number[][]; spheres: number[][] } = await page.evaluate(() =>
            (window as any).boxes(),
        );
        const overlaps = (a: number[], b: number[]) =>
            a[0]! < b[2]! && a[2]! > b[0]! && a[1]! < b[3]! && a[3]! > b[1]!;
        const { counts: ink, restored }: { counts: number[]; restored: boolean[] } =
            await page.evaluate((boxes) => (window as any).nameInk(boxes), boxes.names);
        for (const [i, box] of boxes.names.entries()) {
            expect(box[0]).toBeGreaterThan(0);
            expect(box[1]).toBeGreaterThan(0);
            expect(box[2]).toBeLessThan(640);
            for (const other of [...boxes.names.slice(i + 1), ...boxes.spheres])
                expect(overlaps(box, other)).toBe(false);
            expect(ink[i]).toBeGreaterThan(10);
            expect(restored[i], "showing a name again restores every authored Text value").toBe(
                true,
            );
        }
    });

    test("stepped draws show blended glass, repeating wood, animated and row-coloured checker pixels, wave silhouette and registered background", async () => {
        await page.evaluate(() => (window as any).resize(320));
        console.log("materials adapter", await page.evaluate(() => (window as any).adapter));
        const aim = (index: number) => page.evaluate((i) => (window as any).aim(i), index);
        const pixels = (): Promise<JSHandle<number[]>> =>
            page.evaluateHandle((): Promise<number[]> => (window as any).pixels());
        const changed = (a: JSHandle<number[]>, b: JSHandle<number[]>) =>
            page.evaluate(
                ({ a, b }) => {
                    let count = 0;
                    for (let y = 25; y < 150; y++)
                        for (let x = 90; x < 230; x++) {
                            const i = (y * 320 + x) * 4;
                            if (
                                Math.max(...[0, 1, 2].map((c) => Math.abs(a[i + c]! - b[i + c]!))) >
                                12
                            )
                                count++;
                        }
                    return count;
                },
                { a, b },
            );
        await aim(3);
        const glass = await pixels();
        await page.evaluate(() => (window as any).hide(3, true));
        const behind = await pixels();
        await page.evaluate(() => {
            (window as any).hide(3, false);
            (window as any).glassOpaque();
        });
        const opaque = await pixels();
        const center = (a: JSHandle<number[]>) =>
            a.evaluate((a) => a.slice((90 * 320 + 160) * 4, (90 * 320 + 160) * 4 + 3));
        const [glassCenter, behindCenter, opaqueCenter] = await Promise.all(
            [glass, behind, opaque].map(center),
        );
        console.log("glass, behind, opaque", glassCenter, behindCenter, opaqueCenter);
        expect(await changed(glass, behind)).toBeGreaterThan(100);
        expect(await changed(glass, opaque)).toBeGreaterThan(100);
        for (let c = 0; c < 3; c++)
            expect(glassCenter[c]!).toBeGreaterThanOrEqual(
                Math.min(behindCenter[c]!, opaqueCenter[c]!) - 3,
            );
        for (let c = 0; c < 3; c++)
            expect(glassCenter[c]!).toBeLessThanOrEqual(
                Math.max(behindCenter[c]!, opaqueCenter[c]!) + 3,
            );
        await aim(4);
        const wood = await pixels();
        await page.evaluate(() => (window as any).woodUnlit(true));
        const tiled = await pixels();
        // The central wood patch has both UV coordinates above 1. Without lighting, clamp is flat.
        const grain = await tiled.evaluate((tiled) => {
            const colors = new Set<string>();
            for (let y = 80; y < 100; y++)
                for (let x = 150; x < 170; x++) {
                    const i = (y * 320 + x) * 4;
                    colors.add(
                        tiled
                            .slice(i, i + 3)
                            .map((c) => Math.round(c / 4))
                            .join(","),
                    );
                }
            return colors.size;
        });
        expect(grain, "wood grain beyond [0, 1] must repeat rather than clamp").toBeGreaterThan(30);
        await page.evaluate(() => (window as any).woodOff());
        expect(await changed(wood, await pixels())).toBeGreaterThan(500);
        await aim(5);
        const checker = await pixels();
        await page.evaluate(() => (window as any).advance(0.37));
        const animatedChecker = await pixels();
        expect(await changed(checker, animatedChecker)).toBeGreaterThan(500);
        await page.evaluate(() => (window as any).checkerBlue());
        expect(await changed(animatedChecker, await pixels())).toBeGreaterThan(500);
        await aim(6);
        const wave = await pixels();
        await page.evaluate(() => (window as any).advance(0.48));
        const moved = await pixels();
        // The upper background is uniform: membership changes, not lit interior shading.
        const background = await wave.evaluate((wave) => wave.slice(0, 3));
        const silhouette = await page.evaluate(
            ({ wave, moved, background }) => {
                const isBackground = (a: number[], i: number) =>
                    [0, 1, 2].every((c) => Math.abs(a[i + c]! - background[c]!) < 3);
                let count = 0;
                for (let y = 20; y < 140; y++)
                    for (let x = 100; x < 220; x++) {
                        const i = (y * 320 + x) * 4;
                        if (isBackground(wave, i) !== isBackground(moved, i)) count++;
                    }
                return count;
            },
            { wave, moved, background },
        );
        expect(silhouette).toBeGreaterThan(40);
        // Declared linear backdrop, encoded through the default presentation.
        expect(background[0]).toBeGreaterThanOrEqual(25);
        expect(background[0]).toBeLessThanOrEqual(35);
        expect(background[1]).toBeGreaterThanOrEqual(23);
        expect(background[1]).toBeLessThanOrEqual(32);
        expect(background[2]).toBeGreaterThanOrEqual(21);
        expect(background[2]).toBeLessThanOrEqual(30);
    });
});
