import { mkdirSync } from "node:fs";
import { pipelineWarnings } from "@dylanebert/shallot/testing";
import { expect, type Page, test } from "playwright/test";

type SteppedWindow = {
    ready: boolean;
    evidence(name: string): Promise<number | { overlap: number; exposed: number }>;
    frame(name: string): Promise<void>;
    adapter: { class: string; identity: string };
    jitter(): Promise<{ maxChanged: number; occupied: number; turn: number }>;
};
test.use(pipelineWarnings);

test("the first frame has no late-pipeline warning", async ({ page }) => {
    await page.goto("http://127.0.0.1:4196/");
    await page.waitForFunction(() =>
        performance
            .getEntriesByType("measure")
            .some((entry) => entry.name.endsWith(":first-update")),
    );
});

test("jitter: turning at 144 Hz leaves the stationary view model silhouette fixed", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4196/tests/stepped.html");
    await page.waitForFunction(() => (window as unknown as SteppedWindow).ready);
    const result = await page.evaluate(() => (window as unknown as SteppedWindow).jitter());
    console.log("jitter", result);
    expect(result.occupied).toBeGreaterThan(1000);
    expect(Math.abs(result.turn)).toBeGreaterThan(0.2);
    expect(result.maxChanged).toBe(0);
});

// Each evidence call places the player and restores what it changes, so the claims share one stepped page.
test.describe(() => {
    let page: Page;
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto("http://127.0.0.1:4196/tests/stepped.html");
        await page.waitForFunction(() => (window as unknown as SteppedWindow).ready);
    });
    test.afterAll(() => page.close());
    for (const [name, claim] of [
        ["contact", "the held tool remains visible at the wall"],
        ["depth", "overlapping parts retain their mutual depth ordering"],
        ["count", "three parts draw exactly once across both cameras"],
        ["fov", "world FOV does not move the held tool's pixels"],
        ["light", "dimming the shared sun changes the held tool"],
    ] as const) {
        test(`${name}: ${claim}`, async () => {
            errors.length = 0;
            console.log(
                "view-model adapter",
                await page.evaluate(() => (window as unknown as SteppedWindow).adapter),
            );
            const value = await page.evaluate(
                (name) => (window as unknown as SteppedWindow).evidence(name),
                name,
            );
            console.log(name, value);
            if (name === "contact" || name === "light") expect(value).toBeGreaterThan(500);
            if (name === "count") expect(value).toBe(3);
            if (name === "fov") expect(value).toBe(0);
            if (name === "depth") {
                expect((value as { overlap: number }).overlap).toBe(0);
                expect((value as { exposed: number }).exposed).toBeGreaterThan(20);
            }
            if (name === "contact") {
                const html = await (await page.request.get("http://127.0.0.1:4196/")).text();
                const elements = await page.evaluate((html) => {
                    const document = new DOMParser().parseFromString(html, "text/html");
                    return [...document.body.children].map((element) => element.tagName);
                }, html);
                expect(elements).toEqual(["CANVAS", "SCRIPT"]);
            }
            if (process.env.SHALLOT_CAPTURE_DIR && name === "contact") {
                mkdirSync(process.env.SHALLOT_CAPTURE_DIR, { recursive: true });
                for (const frame of ["held", "wall"]) {
                    await page.evaluate(
                        (name) => (window as unknown as SteppedWindow).frame(name),
                        frame,
                    );
                    await page
                        .locator("canvas")
                        .screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/${frame}.png` });
                }
            }
            expect(errors).toEqual([]);
        });
    }
});

test("evidence calls restore each part's own layers, so a hidden part stays hidden", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4196/tests/stepped.html?hidden");
    await page.waitForFunction(() => (window as unknown as SteppedWindow).ready);
    const evidence = (name: string) =>
        page.evaluate((name) => (window as unknown as SteppedWindow).evidence(name), name);
    await evidence("contact");
    await evidence("depth");
    expect(await evidence("count")).toBe(2);
});
