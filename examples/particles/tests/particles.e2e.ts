import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "playwright/test";

declare global {
    interface Window {
        particles: {
            adapter: { class: string; identity: string };
            requests: { requiredLimits?: Record<string, number>; requiredFeatures?: string[] }[];
            evidence(): Promise<{
                increase: number;
                changed: number;
                floorBeam: number;
                floorShade: number;
                wallLuminance: number;
                floorLuminance: number;
                clearLuminance: number;
                hazeBeam: number;
                hazeShade: number;
            }>;
            walk(count?: number): { before: number; after: number };
            profiler(): void;
            profile(): { gpuTiming: string; passes: string[] };
            capture(): Promise<string>;
        };
    }
}
test("the floor beam outshines adjacent shade, motes brighten the shared view, and injected walking draws on default device limits", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
        if (message.type() === "error") errors.push(message.text());
    });
    await page.goto("http://127.0.0.1:4196/tests/stepped.html");
    await page.waitForFunction(() => window.particles);
    console.log("particles browser adapter", await page.evaluate(() => window.particles.adapter));
    console.log("particles device requests", await page.evaluate(() => window.particles.requests));
    const requests = await page.evaluate(() => window.particles.requests);
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
    const evidence = await page.evaluate(() => window.particles.evidence());
    console.log("beam motes", evidence);
    expect(evidence.increase).toBeGreaterThan(0.02);
    expect(evidence.changed).toBeGreaterThan(5);
    expect(evidence.floorBeam).toBeGreaterThan(evidence.floorShade + 20);
    expect(evidence.hazeBeam).toBeGreaterThan(evidence.hazeShade + 0.04);
    expect(evidence.wallLuminance).toBeGreaterThan(evidence.clearLuminance);
    expect(evidence.wallLuminance).toBeLessThan(evidence.floorLuminance);
    const walk = await page.evaluate(() => window.particles.walk());
    expect(walk.after).toBeLessThan(walk.before - 1);
    await page.evaluate(() => window.particles.evidence());
    expect(errors).toEqual([]);
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
        await page.goto("http://127.0.0.1:4196/tests/stepped.html");
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
test("spawn, mid-stir and profiler captures use the game's composed first-person view", async ({
    page,
}) => {
    const directory = process.env.SHALLOT_CAPTURE_DIR;
    await page.goto(`http://127.0.0.1:4196/tests/stepped.html${directory ? "?captures" : ""}`);
    await page.waitForFunction(() => window.particles);
    const capture = async (name: string) => {
        const image = await page.evaluate(() => window.particles.capture());
        if (!directory) return;
        mkdirSync(directory, { recursive: true });
        writeFileSync(`${directory}/${name}.png`, Buffer.from(image.split(",")[1], "base64"));
    };
    await capture("spawn-facing-beam");
    await page.evaluate(() => window.particles.walk(32));
    await capture("mid-stir");
    await page.evaluate(() => window.particles.profiler());
    await capture("profiler-canvas");
    if (directory) await page.screenshot({ path: `${directory}/profiler-shown.png` });
});
