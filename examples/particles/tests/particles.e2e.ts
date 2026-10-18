import { mkdirSync, writeFileSync } from "node:fs";
import { expect, type Page, test } from "playwright/test";

type Evidence = {
    increase: number;
    changed: number;
    floorBeam: number;
    floorShade: number;
    wallLuminance: number;
    floorLuminance: number;
    clearLuminance: number;
    hazeBeam: number;
    hazeShade: number;
};
declare global {
    interface Window {
        particles: {
            adapter: { class: string; identity: string };
            requests: { requiredLimits?: Record<string, number>; requiredFeatures?: string[] }[];
            evidence(): Promise<Evidence>;
            walk(count?: number): { before: number; after: number };
            profiler(): void;
            profile(): { gpuTiming: string; passes: string[] };
            capture(): Promise<string>;
        };
    }
}
test.beforeEach(async ({ page }) => {
    const session = await page.context().newCDPSession(page);
    await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
});

async function capture(page: Page) {
    return page.evaluate(async (save) => {
        const image = await window.particles.capture();
        const bitmap = await createImageBitmap(await (await fetch(image)).blob());
        const canvas = document.createElement("canvas");
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const context = canvas.getContext("2d")!;
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
        const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
        const colors = new Set<number>();
        for (let i = 0; i < rgba.length; i += 4)
            colors.add((rgba[i]! << 16) | (rgba[i + 1]! << 8) | rgba[i + 2]!);
        return { image: save ? image : undefined, colors: colors.size };
    }, Boolean(process.env.SHALLOT_CAPTURE_DIR));
}

test.describe("one composed view", () => {
    test.describe.configure({ mode: "serial" });
    let page: Page;
    let evidence: Evidence;
    let walk: { before: number; after: number };
    let requests: Window["particles"]["requests"];
    const frames: Awaited<ReturnType<typeof capture>>[] = [];
    const errors: string[] = [];
    test.beforeAll(async ({ browser }) => {
        page = await browser.newPage();
        const session = await page.context().newCDPSession(page);
        await session.send("Emulation.setHardwareConcurrencyOverride", { hardwareConcurrency: 1 });
        page.on("pageerror", (error) => errors.push(error.message));
        page.on("console", (message) => {
            if (message.type() === "error") errors.push(message.text());
        });
        await page.goto("http://127.0.0.1:4196/tests/stepped.html");
        await page.waitForFunction(() => window.particles);
        console.log(
            "particles browser adapter",
            await page.evaluate(() => window.particles.adapter),
        );
        requests = await page.evaluate(() => window.particles.requests);
        console.log("particles device requests", requests);
        frames.push(await capture(page));
        evidence = await page.evaluate(() => window.particles.evidence());
        walk = await page.evaluate(() => window.particles.walk(24));
        frames.push(await capture(page));
        await page.evaluate(() => window.particles.profiler());
        frames.push(await capture(page));
    });
    test.afterAll(() => page.close());
    test.afterEach(() => expect(errors).toEqual([]));

    test("the floor beam outshines adjacent shade, motes brighten the shared view, and injected walking draws on default device limits", () => {
        expect(requests).toHaveLength(1);
        const legacyLimits = new Set([
            "maxStorageBuffersInVertexStage",
            "maxStorageBuffersInFragmentStage",
            "maxStorageTexturesInVertexStage",
            "maxStorageTexturesInFragmentStage",
        ]);
        for (const [name, value] of Object.entries(requests[0].requiredLimits ?? {})) {
            expect(legacyLimits.has(name)).toBe(true);
            expect(value).toBe(0);
        }
        expect([...(requests[0].requiredFeatures ?? [])].sort()).toEqual(
            ["indirect-first-instance", "timestamp-query"]
                .filter(
                    (feature) =>
                        feature !== "timestamp-query" ||
                        requests[0].requiredFeatures?.includes(feature),
                )
                .sort(),
        );
        console.log("beam motes", evidence);
        expect(evidence.increase).toBeGreaterThan(0.02);
        expect(evidence.changed).toBeGreaterThan(5);
        expect(evidence.floorBeam).toBeGreaterThan(evidence.floorShade + 20);
        expect(evidence.hazeBeam).toBeGreaterThan(evidence.hazeShade + 0.04);
        expect(evidence.wallLuminance).toBeGreaterThan(evidence.clearLuminance);
        expect(evidence.wallLuminance).toBeLessThan(evidence.floorLuminance);
        expect(walk.after).toBeLessThan(walk.before - 1);
    });

    test("spawn, mid-stir and profiler captures use the game's composed first-person view", async ({
        page: manualPage,
    }) => {
        const directory = process.env.SHALLOT_CAPTURE_DIR;
        let captures = frames;
        if (directory) {
            await manualPage.goto("http://127.0.0.1:4196/tests/stepped.html?captures");
            await manualPage.waitForFunction(() => window.particles);
            captures = [await capture(manualPage)];
            await manualPage.evaluate(() => window.particles.walk(32));
            captures.push(await capture(manualPage));
            await manualPage.evaluate(() => window.particles.profiler());
            captures.push(await capture(manualPage));
        }
        for (const [i, name] of ["spawn-facing-beam", "mid-stir", "profiler-canvas"].entries()) {
            const { image, colors } = captures[i]!;
            expect(
                colors,
                `${name} captures the composed scene, not a blank canvas`,
            ).toBeGreaterThan(32);
            if (directory && image) {
                mkdirSync(directory, { recursive: true });
                writeFileSync(
                    `${directory}/${name}.png`,
                    Buffer.from(image.split(",")[1], "base64"),
                );
            }
        }
        if (directory) await manualPage.screenshot({ path: `${directory}/profiler-shown.png` });
    });
});

for (const timestamps of ["adapter", "withheld"] as const) {
    test(`the stepped particles page keeps its profiler closed until F3 (${timestamps} timestamps)`, async ({
        page,
    }) => {
        if (timestamps === "withheld") {
            await page.addInitScript(() => {
                const request = GPUAdapter.prototype.requestDevice;
                GPUAdapter.prototype.requestDevice = function (descriptor) {
                    return request.call(this, {
                        ...descriptor,
                        requiredFeatures: [...(descriptor?.requiredFeatures ?? [])].filter(
                            (feature) => feature !== "timestamp-query",
                        ),
                    });
                };
            });
        }
        const errors: string[] = [];
        page.on("pageerror", (error) => errors.push(error.message));
        await page.setViewportSize({ width: 320, height: 180 });
        await page.goto("http://127.0.0.1:4196/tests/stepped.html?profiler");
        await page.waitForFunction(() => window.particles);
        const profiler = page.locator("[data-shallot-profile]");
        await expect(profiler).toHaveCount(0);
        await page.keyboard.press("F3");
        const profile = await page.evaluate(() => window.particles.profile());
        await expect(profiler).toHaveAttribute(
            "data-gpu-timing",
            /available|requires timestamp-query/,
        );
        await expect(profiler).toHaveCSS("font-size", "10px");
        await expect(profiler).toHaveCSS("min-width", "220px");
        expect(
            await profiler.evaluate(
                (element) =>
                    element.getBoundingClientRect().width / (element as HTMLElement).offsetWidth,
            ),
        ).toBeCloseTo(1, 2);
        if (timestamps === "withheld") {
            await expect(profiler).toHaveAttribute("data-gpu-timing", "requires timestamp-query");
        }
        if (profile.gpuTiming === "available") {
            expect(profile.passes).toContain("particles:advance");
        } else {
            await expect(
                profiler.getByText("requires timestamp-query", { exact: true }),
            ).toBeVisible();
            await expect(profiler.getByText("particles:advance", { exact: true })).toHaveCount(0);
        }
        expect(errors).toEqual([]);
    });
}
