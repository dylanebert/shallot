import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as harness from "@dylanebert/shallot/harness";
import { assertMotion, isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { pixelProbePass, probePixels } from "@dylanebert/shallot/harness/pixels";
import { expect, type Locator, type Page, test } from "@playwright/test";
import { PNG } from "pngjs";
import { adapterName, SOFTWARE } from "./gpu-adapter";

const KNOWN_PREEXISTING_WARNING = /\[implicit-conversion\][\s\S]*struct:vertexVsOut/;
const EXCLUDE_DOM =
    "body * { visibility: hidden !important; } canvas { visibility: visible !important; }";
const PARKED = /^samples are parked \(mean absolute difference [\d.]+, need > 0\.1\)$/;
const SWATCH_MESSAGE = "the composited cell grid should contain the cube swatch";
const swatch = {
    name: "glyph-colored cube reaches the compositor",
    minPixels: 500,
    minSpan: 40,
    r: [70, 255] as [number, number],
    g: [30, 230] as [number, number],
    b: [5, 190] as [number, number],
};
const hash = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");
const bindings = Object.fromEntries(
    [
        "@dylanebert/shallot/harness",
        "@dylanebert/shallot/harness/pixels",
        "./gpu-adapter.ts",
        "@playwright/test",
    ].map((id) => {
        const path = realpathSync(fileURLToPath(import.meta.resolve(id)));
        return [id, { path, sha256: hash(readFileSync(path)) }];
    }),
);
const readerPath = realpathSync(fileURLToPath(import.meta.url));
const barrelPath = realpathSync(fileURLToPath(import.meta.resolve("@dylanebert/shallot/harness")));
const motionPath = realpathSync(join(dirname(barrelPath), "motion.ts"));
const barrelSource = readFileSync(barrelPath, "utf8");
expect(barrelSource.match(/export \{ assertMotion \} from "\.\/motion";/g)).toHaveLength(1);
expect(harness.assertMotion).toBe(assertMotion);

async function gesture(page: Page, canvas: Locator) {
    const box = await canvas.boundingBox();
    expect(box).not.toBeNull();
    if (!box) throw new Error("missing gesture canvas");
    const x = box.x + box.width * 0.4;
    const y = box.y + box.height * 0.5;
    await page.mouse.move(x, y);
    await page.mouse.down();
    try {
        for (let move = 1; move <= 6; move++) {
            await page.mouse.move(x + box.width * 0.03 * move, y + box.height * 0.01 * move);
        }
    } finally {
        await page.mouse.up();
    }
}

// The compositor population includes the full canvas, but not overlaid DOM controls.
test("ascii showcase — the cell grid reaches the compositor", async ({
    browser,
    baseURL,
}, info) => {
    const record = (kind: string, data: object) => {
        console.log(
            `ASCII_PROBE ${JSON.stringify({ utc: new Date().toISOString(), monoNs: process.hrtime.bigint().toString(), kind, ...data })}`,
        );
    };
    record("identity", {
        bindings,
        reader: { path: readerPath, sha256: hash(readFileSync(readerPath)) },
        motion: {
            path: motionPath,
            sha256: hash(readFileSync(motionPath)),
            barrelPath: realpathSync(barrelPath),
            export: "assertMotion",
            callerIdentity: true,
        },
        node: process.version,
        browser: browser.version(),
    });
    for (const phase of ["positive", "locked", "absent"] as const) {
        const context = await browser.newContext({ baseURL });
        const errors: string[] = [];
        const warnings: string[] = [];
        let responses = 0;
        let captures = 0;
        record("phase-start", { phase });
        try {
            const page = await context.newPage();
            page.on("pageerror", (error) => errors.push(String(error)));
            page.on("console", (message) => {
                if (message.type() === "error" || isDegradedBootMessage(message.text())) {
                    errors.push(`[console.${message.type()}] ${message.text()}`);
                }
                if (
                    message.type() === "warning" &&
                    !KNOWN_PREEXISTING_WARNING.test(message.text())
                ) {
                    warnings.push(`[console.warning] ${message.text()}`);
                }
            });
            if (phase !== "positive") {
                await page.route("**/scenes/ascii.scene", async (route) => {
                    responses++;
                    expect(responses, `${phase} scene response count`).toBe(1);
                    const response = await route.fetch();
                    expect(response.ok()).toBe(true);
                    const original = await response.body();
                    const source = original.toString("utf8");
                    const pattern =
                        phase === "locked"
                            ? /<a\b[^>]*\bid="camera"[^>]*\/>/g
                            : /<a\b[^>]*\bid="box"[^>]*\/>/g;
                    const targets = source.match(pattern) ?? [];
                    expect(targets, `${phase} intended scene target`).toHaveLength(1);
                    const target = targets[0];
                    if (targets.length !== 1 || target === undefined)
                        throw new Error("scene target is not unique");
                    let replacement = "";
                    if (phase === "locked") {
                        const orbits = target.match(/\borbit="[^"]*"/g) ?? [];
                        expect(orbits).toHaveLength(1);
                        const orbit = orbits[0];
                        if (orbits.length !== 1 || orbit === undefined)
                            throw new Error("camera orbit is not unique");
                        expect(orbit).not.toMatch(/\bmode\s*:/);
                        replacement = target.replace(orbit, orbit.replace(/"$/, '; mode: 1"'));
                    }
                    const mutated = Buffer.from(source.replace(target, replacement));
                    await info.attach(`${phase}-scene-original`, {
                        body: original,
                        contentType: "application/xml",
                    });
                    await info.attach(`${phase}-scene-mutated`, {
                        body: mutated,
                        contentType: "application/xml",
                    });
                    record("mutation", {
                        phase,
                        responses,
                        targets: targets.length,
                        originalBytes: original.length,
                        mutatedBytes: mutated.length,
                        originalSha256: hash(original),
                        mutatedSha256: hash(mutated),
                    });
                    await route.fulfill({ response, body: mutated });
                });
            }
            const sceneResponse = page.waitForResponse(
                (response) => new URL(response.url()).pathname === "/scenes/ascii.scene",
            );
            await page.goto("/");
            const served = await sceneResponse;
            const servedBytes = await served.body();
            await info.attach(`${phase}-scene-served`, {
                body: servedBytes,
                contentType: "application/xml",
            });
            record("served", {
                phase,
                url: served.url(),
                sha256: hash(servedBytes),
                bytes: servedBytes.length,
            });
            const adapter = await adapterName(page);
            record("adapter", { phase, adapter });
            test.skip(
                adapter === "" || SOFTWARE.test(adapter),
                `no real-GPU adapter (${adapter || "none offered"})`,
            );
            const canvas = page.locator("canvas").first();
            await expect(canvas).toBeVisible();
            await expect
                .poll(() =>
                    canvas.evaluate((element) => {
                        const cells = element as HTMLCanvasElement & {
                            cellCols?: number;
                            cellRows?: number;
                        };
                        return (cells.cellCols ?? 0) * (cells.cellRows ?? 0);
                    }),
                )
                .toBeGreaterThan(0);
            const metadata = await canvas.evaluate((element) => {
                const cells = element as HTMLCanvasElement & {
                    cellCols?: number;
                    cellRows?: number;
                };
                return { cols: cells.cellCols, rows: cells.cellRows };
            });
            if (phase !== "positive") expect(responses).toBe(1);
            const box = await canvas.boundingBox();
            expect(box).not.toBeNull();
            if (!box) throw new Error("missing phase canvas box");
            record("ready", { phase, metadata, responses, box });
            const panel = page
                .getByRole("button", { name: "+", exact: true })
                .first()
                .locator("..")
                .locator("..");
            await expect(panel).toBeVisible();
            const capture = async (name: string, exclude = true) => {
                const beforeBox = await canvas.boundingBox();
                expect(beforeBox).toEqual(box);
                const visibility = await panel.evaluate(
                    (element) => getComputedStyle(element).visibility,
                );
                const bytes = await canvas.screenshot({
                    timeout: 5_000,
                    style: exclude ? EXCLUDE_DOM : undefined,
                });
                const png = PNG.sync.read(bytes);
                expect(png.width).toBeGreaterThan(0);
                expect(png.height).toBeGreaterThan(0);
                expect(png.data.length).toBe(png.width * png.height * 4);
                expect(await canvas.boundingBox()).toEqual(box);
                expect(
                    await panel.evaluate((element) => getComputedStyle(element).visibility),
                ).toBe(visibility);
                await expect(panel).toBeVisible();
                const id = `${phase}-${captures++}-${name}`;
                await info.attach(`${id}.png`, { body: bytes, contentType: "image/png" });
                await info.attach(`${id}.rgba`, {
                    body: png.data,
                    contentType: "application/octet-stream",
                });
                record("capture", {
                    phase,
                    id,
                    exclude,
                    width: png.width,
                    height: png.height,
                    count: png.data.length,
                    box: beforeBox,
                    visibility,
                    pngSha256: hash(bytes),
                    rgbaSha256: hash(png.data),
                });
                return png;
            };
            let result = { pixels: 0, width: 0, height: 0 };
            let captureError: unknown;
            const swatches = async () => {
                await expect
                    .poll(
                        async () => {
                            const png = await capture("swatch").catch((error: unknown) => {
                                captureError = error;
                                throw error;
                            });
                            result = probePixels(png.data, png.width, png.height, swatch);
                            record("swatch", {
                                phase,
                                ...result,
                                pass: pixelProbePass(result, swatch),
                            });
                            return pixelProbePass(result, swatch);
                        },
                        { message: SWATCH_MESSAGE, timeout: 15_000 },
                    )
                    .toBe(true);
                expect(
                    pixelProbePass(result, swatch),
                    `matched ${result.pixels} px, span ${result.width}x${result.height}`,
                ).toBe(true);
            };
            if (phase === "absent") {
                await expect(
                    swatches(),
                    "absent cube must reject at the swatch consumer",
                ).rejects.toThrow(SWATCH_MESSAGE);
                expect(
                    captureError,
                    "absent control must not consume a capture error",
                ).toBeUndefined();
                expect(captures).toBeGreaterThan(0);
                expect(pixelProbePass(result, swatch)).toBe(false);
                record("absent-rejected", { phase, ...result });
            } else {
                await swatches();
                const rawBefore = phase === "locked" ? await capture("raw-before", false) : null;
                const before = await capture("before");
                if (phase === "positive") {
                    await gesture(page, canvas);
                } else {
                    await gesture(page, canvas);
                    await panel.evaluate((element) => {
                        (element as HTMLElement).style.backgroundColor = "rgb(255, 255, 255)";
                    });
                }
                await page.waitForTimeout(500);
                const after = await capture("after");
                const motion = (first: PNG, second: PNG, population: string) => {
                    expect([second.width, second.height]).toEqual([first.width, first.height]);
                    expect(second.data.length).toBe(first.data.length);
                    let sum = 0;
                    for (let i = 0; i < first.data.length; i++)
                        sum += Math.abs(first.data[i] - second.data[i]);
                    record("motion", {
                        phase,
                        population,
                        sum,
                        count: first.data.length,
                        mean: sum / first.data.length,
                    });
                    return () => assertMotion(first.data, second.data, 0.1);
                };
                if (phase === "positive") {
                    motion(before, after, "excluded")();
                } else {
                    expect(rawBefore).not.toBeNull();
                    if (!rawBefore) throw new Error("missing DOM foil capture");
                    const rawAfter = await capture("raw-after", false);
                    motion(rawBefore, rawAfter, "raw")();
                    expect(
                        motion(before, after, "excluded"),
                        "locked camera must reject as parked",
                    ).toThrow(PARKED);
                    record("locked-rejected", { phase });
                }
            }
            expect(errors, `page errors: ${errors.join("\n")}`).toEqual([]);
            expect(warnings, `page console warnings: ${warnings.join("\n")}`).toEqual([]);
            if (phase !== "positive") expect(responses).toBe(1);
            record("phase-pass", { phase, responses, captures });
        } finally {
            await context.close();
            record("phase-closed", { phase, responses, captures, errors, warnings });
        }
    }
});
