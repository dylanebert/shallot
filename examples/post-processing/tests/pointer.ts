import { expect, type Locator, type Page } from "playwright/test";

export async function bodyHasFocus(page: Page): Promise<void> {
    expect(
        await page.evaluate(() => document.activeElement === document.body),
        "chrome must leave keyboard focus on body",
    ).toBe(true);
}

export async function pointerControl(page: Page, control: Locator): Promise<void> {
    await control.scrollIntoViewIfNeeded();
    const box = (await control.boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.mouse.down();
    await bodyHasFocus(page);
    await page.mouse.up();
    await bodyHasFocus(page);
}

export async function dragRange(page: Page, control: Locator): Promise<void> {
    await control.scrollIntoViewIfNeeded();
    const box = (await control.boundingBox())!;
    const before = await control.inputValue();
    await page.mouse.move(box.x + box.width * 0.25, box.y + box.height / 2);
    await page.mouse.down();
    await bodyHasFocus(page);
    await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2, { steps: 8 });
    await bodyHasFocus(page);
    await page.mouse.up();
    await bodyHasFocus(page);
    expect(await control.inputValue()).not.toBe(before);
}

export async function gameKeyMovesCamera(page: Page): Promise<void> {
    const step = () =>
        page.evaluate(() => (window as unknown as { controlsKey(): number }).controlsKey());
    const before = await step();
    await page.keyboard.down("ArrowRight");
    try {
        expect(await step()).not.toBe(before);
    } finally {
        await page.keyboard.up("ArrowRight");
        await step();
    }
}

export async function tabSkipsChrome(page: Page): Promise<void> {
    const controls = page.locator("button, input, summary, [tabindex]");
    for (let i = 0; i <= (await controls.count()); i++) {
        await page.keyboard.press("Tab");
        expect(
            await controls.evaluateAll((elements) =>
                elements.some((element) => element === document.activeElement),
            ),
        ).toBe(false);
    }
}
