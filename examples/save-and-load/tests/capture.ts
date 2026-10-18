import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { expect, type Page } from "playwright/test";
import type { Save } from "../src/save";

declare global {
    interface Window {
        capture(): Promise<string>;
        adapter: { class: string; identity: string };
        saveAndLoad: {
            state(): { count: number; save: Save };
            advance(ticks?: number): Promise<unknown>;
            hover(): void;
            click(): void;
            key(code: string): void;
        };
    }
}
export async function capture(page: Page, name: string): Promise<void> {
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    if (!directory) return;
    mkdirSync(dirname(`${directory}/${name}.png`), { recursive: true });
    const adapter = await page.evaluate(() => window.adapter);
    expect(adapter.class).toBe("real");
    console.log("save-and-load adapter", adapter);
    const image = await page.evaluate(() => window.capture());
    await page.evaluate(async (src) => {
        const img = new Image();
        img.id = "capture";
        img.style.cssText = "position:absolute;left:0;top:0;pointer-events:none";
        img.src = src;
        await img.decode();
        document.body.insertBefore(img, document.querySelector("canvas")!.nextSibling);
    }, image);
    await page.screenshot({ path: `${directory}/${name}.png` });
    await page.evaluate(() => document.getElementById("capture")!.remove());
}
