import { expect, type Page, test } from "playwright/test";
import { CEILING } from "../../scripts/test-tiers";

async function canvasImage(page: Page): Promise<string> {
    const canvas = page.locator("#canvas");
    const bounds = await canvas.boundingBox();
    expect(bounds, "the first-person canvas has a page rectangle").not.toBeNull();
    return (await page.screenshot({ clip: bounds! })).toString("base64");
}

async function changedFraction(page: Page, first: string, second: string): Promise<number> {
    return page.evaluate(
        async ({ first, second }) => {
            const pixels = async (base64: string) => {
                const image = await createImageBitmap(
                    await (await fetch(`data:image/png;base64,${base64}`)).blob(),
                );
                const surface = document.createElement("canvas");
                surface.width = image.width;
                surface.height = image.height;
                const context = surface.getContext("2d")!;
                context.drawImage(image, 0, 0);
                image.close();
                return context.getImageData(0, 0, surface.width, surface.height).data;
            };
            const a = await pixels(first);
            const b = await pixels(second);
            if (a.length !== b.length) return 1;
            let changed = 0;
            for (let offset = 0; offset < a.length; offset += 4) {
                const delta = Math.max(
                    Math.abs(a[offset]! - b[offset]!),
                    Math.abs(a[offset + 1]! - b[offset + 1]!),
                    Math.abs(a[offset + 2]! - b[offset + 2]!),
                );
                if (delta > 24) changed++;
            }
            return changed / (a.length / 4);
        },
        { first, second },
    );
}

test("the browser input adapter fails to record a real key press on the focused canvas or accepts it after focus leaves the canvas", async ({
    page,
}) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "warning" && message.text().startsWith("[shallot]"))
            console.info(`browser adapter: ${message.text()}`);
    });
    await page.addInitScript(() => {
        const events: string[] = [];
        let frames = 0;
        const requestFrame = window.requestAnimationFrame.bind(window);
        window.requestAnimationFrame = (callback) =>
            requestFrame((timestamp) => {
                frames++;
                callback(timestamp);
            });
        Object.defineProperty(window, "__shallotBrowserEvents", { value: events });
        Object.defineProperty(window, "__shallotBrowserFrames", { get: () => frames });
        const record = (event: Event) => {
            const keyboard = event as KeyboardEvent;
            const target = event.target as HTMLElement | null;
            events.push(
                [
                    event.type,
                    keyboard.code ?? "",
                    `active=${document.activeElement?.id || document.activeElement?.tagName}`,
                    `focus=${document.hasFocus()}`,
                    `lock=${document.pointerLockElement?.id || "none"}`,
                    `target=${target?.id || target?.tagName || "none"}`,
                ].join(" "),
            );
        };
        for (const type of ["keydown", "keyup", "blur", "focus"]) {
            window.addEventListener(type, record, true);
        }
        for (const type of ["pointerdown", "pointerlockchange"]) {
            document.addEventListener(type, record, true);
        }
    });
    await page.goto("/");
    const canvas = page.locator("#canvas");
    await expect(canvas).toBeVisible();
    try {
        await expect(page.locator("[data-recipe-controls]")).toContainText("WASD", {
            timeout: CEILING.browser,
        });
    } catch (cause) {
        throw new Error(
            `first-person did not reach its first draw; page errors: ${JSON.stringify(pageErrors)}`,
            { cause },
        );
    }
    expect(pageErrors, "first-person reaches its first draw without a runtime error").toEqual([]);
    await canvas.evaluate((element) => {
        element.tabIndex = 0;
    });
    await canvas.click();
    await canvas.focus();
    expect(
        await canvas.evaluate((element) => document.activeElement === element),
        "the real first-person canvas receives browser focus",
    ).toBe(true);

    // Give a software adapter time to present several frames for each input observation.
    const idleBefore = await canvasImage(page);
    await page.waitForTimeout(2000);
    const idleAfter = await canvasImage(page);
    const idleChange = await changedFraction(page, idleBefore, idleAfter);

    await page.evaluate(() => {
        const outside = document.createElement("button");
        outside.id = "focus-outside-canvas";
        outside.textContent = "outside";
        outside.style.cssText = "position:fixed;right:0;top:0;z-index:99999";
        document.body.append(outside);
    });
    const outside = page.locator("#focus-outside-canvas");
    await page.evaluate(() => document.exitPointerLock());
    await page.waitForFunction(() => document.pointerLockElement === null);
    await outside.click();
    await outside.focus();
    expect(
        await outside.evaluate((element) => document.activeElement === element),
        "focus can leave the real canvas",
    ).toBe(true);
    expect(
        await canvas.evaluate((element) => document.activeElement !== element),
        "the canvas is no longer the focused input target",
    ).toBe(true);

    const outsideBefore = await canvasImage(page);
    await page.keyboard.down("w");
    await page.waitForTimeout(2000);
    const outsideAfter = await canvasImage(page);
    await page.keyboard.up("w");
    const unfocusedChange = await changedFraction(page, outsideBefore, outsideAfter);
    const inputSignal = Math.max(idleChange * 3, 0.02);
    expect(
        unfocusedChange,
        "a W press after canvas focus leaves does not move the first-person scene",
    ).toBeLessThan(inputSignal);

    await canvas.click();
    await canvas.focus();
    const focusedBefore = await canvasImage(page);
    await page.keyboard.down("w");
    await page.waitForTimeout(2000);
    const focusedAfter = await canvasImage(page);
    await page.keyboard.up("w");
    const focusedChange = await changedFraction(page, focusedBefore, focusedAfter);
    if (focusedChange <= inputSignal) {
        const diagnostics = await page.evaluate(() => {
            const browser = window as Window & {
                __shallotBrowserEvents?: string[];
                __shallotBrowserFrames?: number;
            };
            return {
                events: browser.__shallotBrowserEvents ?? [],
                frames: browser.__shallotBrowserFrames,
                focused: document.hasFocus(),
                visible: document.visibilityState,
            };
        });
        console.info(
            "browser input diagnostics:",
            JSON.stringify({
                idleChange,
                unfocusedChange,
                focusedChange,
                inputSignal,
                ...diagnostics,
                pageErrors,
            }),
        );
    }
    expect(
        focusedChange,
        "a focused W press changes the rendered first-person scene beyond idle motion",
    ).toBeGreaterThan(inputSignal);
});
