import { readdirSync } from "node:fs";
import { resolve } from "node:path";
import type { Capture } from "@dylanebert/shallot/rendering";
import { expect, test } from "playwright/test";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "../..");
const examples = readdirSync(resolve(root, "examples"), { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();

for (const example of examples) {
    test(`${example} draws with its overlay inside a phone`, async ({ page }, testInfo) => {
        const directory = resolve(root, "examples", example);
        const server = await createServer({
            root: directory,
            configFile: resolve(directory, "vite.config.ts"),
            server: { host: "127.0.0.1", port: 0 },
        });
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "error") errors.push(message.text());
        });
        try {
            await server.listen();
            const session = await page.context().newCDPSession(page);
            await session.send("Emulation.setHardwareConcurrencyOverride", {
                hardwareConcurrency: 1,
            });
            await page.goto(server.resolvedUrls!.local[0]!);
            await page.addScriptTag({
                type: "module",
                url: `/@fs/${resolve(root, "scripts/phone/capture-page.ts")}`,
            });
            await page.waitForFunction(
                () =>
                    "shallotCaptureFrame" in window &&
                    (document.querySelector("canvas")?.width ?? 0) > 300,
            );
            const frame = await page.evaluate(async () => {
                const capture = await (
                    window as unknown as {
                        shallotCaptureFrame(canvas: HTMLCanvasElement): Promise<Capture>;
                    }
                ).shallotCaptureFrame(document.querySelector("canvas")!);
                const first = capture.rgba.slice(0, 3);
                let varied = 0;
                for (let i = 0; i < capture.rgba.length; i += 4) {
                    if (
                        first.some(
                            (value, channel) => Math.abs(value - capture.rgba[i + channel]!) > 24,
                        )
                    )
                        varied++;
                }
                return { width: capture.width, height: capture.height, varied };
            });
            expect(frame.width).toBeGreaterThan(0);
            expect(frame.height).toBeGreaterThan(0);
            expect(frame.varied, "a drawn scene, not a uniform clear").toBeGreaterThan(100);
            const overflow = await page.locator("body *").evaluateAll((elements) =>
                elements.flatMap((element) => {
                    if (!(element instanceof HTMLElement) || element instanceof HTMLCanvasElement)
                        return [];
                    const style = getComputedStyle(element);
                    const bounds = element.getBoundingClientRect();
                    if (
                        style.visibility === "hidden" ||
                        style.display === "none" ||
                        style.opacity === "0" ||
                        !element.checkVisibility() ||
                        bounds.width === 0 ||
                        bounds.height === 0
                    )
                        return [];
                    if (
                        bounds.left >= -1 &&
                        bounds.top >= -1 &&
                        bounds.right <= innerWidth + 1 &&
                        bounds.bottom <= innerHeight + 1
                    )
                        return [];
                    return [
                        {
                            element: element.outerHTML.slice(0, 200),
                            x: bounds.x,
                            y: bounds.y,
                            width: bounds.width,
                            height: bounds.height,
                        },
                    ];
                }),
            );
            expect(overflow, "every visible overlay element fits the viewport").toEqual([]);
            expect(errors, "no page or console errors").toEqual([]);
            const captureDirectory = testInfo.project.metadata.captureDirectory;
            if (typeof captureDirectory === "string") {
                await page.screenshot({ path: resolve(root, captureDirectory, `${example}.png`) });
            }
        } finally {
            await page.close();
            await server.close();
        }
    });
}
