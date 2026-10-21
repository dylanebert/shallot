import { holdFrames } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

const steps = (page: Page) => page.evaluate(() => (window as unknown as { steps(): number }).steps());

async function heldAcrossActions(page: Page, frames: { run(count?: number): Promise<void> }) {
    const held = await steps(page);
    await page.mouse.move(40, 40, { steps: 4 });
    await page.getByRole("button", { name: "Control" }).click();
    await page.keyboard.press("Tab");
    await page.evaluate(() => new Promise((resolve) => setTimeout(resolve, 200)));
    expect(await steps(page), "no frame runs between actions").toBe(held);
    await frames.run(3);
    expect(await steps(page), "run(3) runs three frames").toBe(held + 3);
    expect(await steps(page)).toBe(held + 3);
}

test("held frames run only on run(count), from a selector becoming ready", async ({ page }) => {
    const frames = await holdFrames(page, "button");
    await page.goto("/frames.html");
    await page.getByRole("button", { name: "Control" }).waitFor();
    await heldAcrossActions(page, frames);
});

test("held frames run only on run(count), from a predicate becoming ready", async ({ page }) => {
    const frames = await holdFrames(page, () => "steps" in window && document.querySelector("button") !== null);
    await page.goto("/frames.html");
    await page.getByRole("button", { name: "Control" }).waitFor();
    await heldAcrossActions(page, frames);
});

test("a reload holds frames again once ready holds", async ({ page }) => {
    const frames = await holdFrames(page, "button");
    await page.goto("/frames.html");
    await page.getByRole("button", { name: "Control" }).waitFor();
    await page.reload();
    await page.getByRole("button", { name: "Control" }).waitFor();
    await heldAcrossActions(page, frames);
});

test("a held frame skips a callback another cancels in the same frame", async ({ page }) => {
    const frames = await holdFrames(page, "button");
    await page.goto("/frames.html");
    await page.getByRole("button", { name: "Control" }).waitFor();
    await page.evaluate(() => {
        const ran: string[] = [];
        (window as unknown as { ran: string[] }).ran = ran;
        let b = 0;
        requestAnimationFrame(() => {
            ran.push("a");
            cancelAnimationFrame(b);
        });
        b = requestAnimationFrame(() => ran.push("b"));
    });
    await frames.run();
    expect(await page.evaluate(() => (window as unknown as { ran: string[] }).ran)).toEqual(["a"]);
});
