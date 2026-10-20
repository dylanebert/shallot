import { mkdirSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        ready: boolean;
        evidence(name: string): Promise<number | { overlap: number; exposed: number }>;
        frame(name: string): Promise<void>;
        adapter: unknown;
        jitter(): Promise<{ maxChanged: number; occupied: number; turn: number }>;
    }
}
test("jitter: turning at 144 Hz leaves the stationary view model silhouette fixed", async ({
    page,
}) => {
    await page.goto("http://127.0.0.1:4196/tests/stepped.html");
    await page.waitForFunction(() => window.ready);
    const result = await page.evaluate(() => window.jitter());
    console.log("jitter", result);
    expect(result.occupied).toBeGreaterThan(1000);
    expect(Math.abs(result.turn)).toBeGreaterThan(0.2);
    expect(result.maxChanged).toBe(0);
});

for (const [name, claim] of [
    ["contact", "the held tool remains visible at the wall"],
    ["depth", "overlapping parts retain their mutual depth ordering"],
    ["count", "three parts draw exactly once across both cameras"],
    ["fov", "world FOV does not move the held tool's pixels"],
    ["light", "dimming the shared sun changes the held tool"],
] as const) {
    test(`${name}: ${claim}`, async ({ page }) => {
        const errors: string[] = [];
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto("http://127.0.0.1:4196/tests/stepped.html");
        await page.waitForFunction(() => window.ready);
        console.log("view-model adapter", await page.evaluate(() => window.adapter));
        const value = await page.evaluate((name) => window.evidence(name), name);
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
                await page.evaluate((name) => window.frame(name), frame);
                await page
                    .locator("canvas")
                    .screenshot({ path: `${process.env.SHALLOT_CAPTURE_DIR}/${frame}.png` });
            }
        }
        expect(errors).toEqual([]);
    });
}
