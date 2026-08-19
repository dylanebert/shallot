import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

// where every run's capture lands — a fixed, git-ignored path (`test-results/`, this project's
// `.gitignore`) rather than a per-run timestamped name, so "the latest capture" always has one findable
// location for the spec's human release-look gate to read.
const CAPTURE_PATH = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test-results",
    "roads-capture.png",
);

// Drive the terrain generator's device gate: load the app, wait for it to warm and expose
// `window.__roadsGate`, run it on the real GPU, and assert every check passes (and the page raised no
// error). The checks themselves live in `src/gate.ts` against the published surface — this driver is the
// only part Playwright touches. One session, phases within one test (the Playwright structure rule).

interface Check {
    name: string;
    pass: boolean;
    detail: string;
}

interface ScreenPoint {
    x: number;
    y: number;
    depth: number;
}

// Software rasterizers by the name they report in `GPUAdapterInfo`: Chromium's SwiftShader, Mesa's two,
// and D3D's WARP. Deliberately a name list rather than a capability probe — SwiftShader clears shallot's
// whole base floor (all three `BASE_FEATURES` plus 10 storage buffers per stage, measured under WSL
// 2026-08-10), so nothing about the floor distinguishes it, and it is only the *execution* that dies.
// Bias narrow: an unlisted software adapter re-crashes loudly, while an over-broad pattern would skip this
// gate on real hardware and report green having tested nothing.
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

/** the adapter's self-reported identity, or `""` when the browser hands out none. */
const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

test("terrain generator gate — sized, deterministic, reseeds, not flat (real GPU)", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    // standalone runtime — the gate only touches GPU buffers, no editor/scene writes.
    await page.goto("/");

    // Full device testing: the generator needs a real GPU. Probe before waiting on the boot hook, because
    // on a software adapter the app never installs it and the wait burns its full 120 s before failing —
    // display-gated the same way `shallot verify`'s callers are, so a GPU-less host skips instead of reds.
    const adapter = await adapterName(page);
    console.log(`roads gate adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    // the boot plugin installs the hook once the terrain mesh is registered (see boot.ts). A cold vite
    // build is slower than the warm path, so allow longer than steady-state needs.
    await page.waitForFunction(() => typeof window.__roadsGate === "function", null, {
        timeout: 120_000,
    });

    const checks = (await page.evaluate(() =>
        (window as unknown as { __roadsGate: () => Promise<Check[]> }).__roadsGate(),
    )) as Check[];

    expect(errors, errors.join("\n")).toEqual([]);
    expect(checks.length).toBeGreaterThan(0);
    for (const c of checks) {
        expect(c.pass, `${c.name}: ${c.detail}`).toBe(true);
    }

    // Phase 2: the overlay substrate's own flagged-risk validation (spec's Locked decision) — a real
    // captured frame, machine-read, over the full procedural network (overlay/network.ts). The compute-write
    // half is proven device-free (overlay/stroke.test.ts's seeded-tile readback oracle); this is the fs
    // composite's own arm, observable only in the rendered frame — the compute-write and the sampled-pixel
    // output are different granularities, and only this arm sees the second one.
    await page.waitForFunction(
        () => (window as unknown as { __roadsOverlayIdle: () => boolean }).__roadsOverlayIdle(),
        null,
        { timeout: 10_000 },
    );

    const { onRoad, offRoad } = (await page.evaluate(() =>
        (
            window as unknown as {
                __roadsCapturePoints: () => Promise<{
                    onRoad: [number, number, number];
                    offRoad: [number, number, number];
                }>;
            }
        ).__roadsCapturePoints(),
    )) as { onRoad: [number, number, number]; offRoad: [number, number, number] };

    const [onRoadScreen, offRoadScreen] = (await page.evaluate(
        (points) =>
            (
                window as unknown as {
                    __roadsProbe: (pts: [number, number, number][]) => ScreenPoint[];
                }
            ).__roadsProbe(points),
        [onRoad, offRoad],
    )) as ScreenPoint[];

    const screenshot = await page.screenshot();
    mkdirSync(dirname(CAPTURE_PATH), { recursive: true });
    writeFileSync(CAPTURE_PATH, screenshot);
    const capture = await page.evaluate(
        async ({ base64, onRoadScreen, offRoadScreen, tolerancePx }) => {
            const binary = atob(base64);
            const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
            const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
            const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
            const context = canvas.getContext("2d");
            if (!context) throw new Error("roads capture: 2D screenshot context unavailable");
            context.drawImage(bitmap, 0, 0);
            const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;

            const luminanceAt = (fracX: number, fracY: number): number => {
                const x = Math.min(bitmap.width - 1, Math.max(0, Math.round(fracX * bitmap.width)));
                const y = Math.min(
                    bitmap.height - 1,
                    Math.max(0, Math.round(fracY * bitmap.height)),
                );
                const at = (y * bitmap.width + x) * 4;
                return 0.3 * data[at] + 0.59 * data[at + 1] + 0.11 * data[at + 2];
            };

            const onRoadLum = luminanceAt(onRoadScreen.x, onRoadScreen.y);
            const offRoadLum = luminanceAt(offRoadScreen.x, offRoadScreen.y);

            // walk pixels along the on-road → off-road screen segment; the transition band is the run of
            // consecutive samples whose luminance sits strictly between the two endpoints (not at either
            // plateau) — its pixel length is what the derived tolerance bounds.
            const dxPx = (offRoadScreen.x - onRoadScreen.x) * bitmap.width;
            const dyPx = (offRoadScreen.y - onRoadScreen.y) * bitmap.height;
            const spanPx = Math.hypot(dxPx, dyPx);
            const steps = Math.max(8, Math.ceil(spanPx * 2)); // ≥2 samples/pixel along the segment
            const lo = Math.min(onRoadLum, offRoadLum);
            const hi = Math.max(onRoadLum, offRoadLum);
            const band = hi - lo;
            const plateau = band * 0.1; // within 10% of an endpoint counts as "at" that endpoint
            let transitionSamples = 0;
            for (let i = 0; i <= steps; i++) {
                const t = i / steps;
                const fx = onRoadScreen.x + (offRoadScreen.x - onRoadScreen.x) * t;
                const fy = onRoadScreen.y + (offRoadScreen.y - onRoadScreen.y) * t;
                const lum = luminanceAt(fx, fy);
                if (lum > lo + plateau && lum < hi - plateau) transitionSamples++;
            }
            const transitionPx = (transitionSamples / steps) * spanPx;

            return { onRoadLum, offRoadLum, transitionPx, spanPx, tolerancePx };
        },
        {
            base64: screenshot.toString("base64"),
            onRoadScreen,
            offRoadScreen,
            tolerancePx: await page.evaluate(
                () =>
                    (window as unknown as { __roadsTransitionTolerancePx: number })
                        .__roadsTransitionTolerancePx,
            ),
        },
    );

    // on-road classifies as road albedo (dark, low-chroma asphalt), off-road as terrain (brighter grass) —
    // a luminance ratio, not a fixed RGB band, since ambient/directional lighting scales both pixels by
    // roughly the same local factor: the albedo constants alone put road/grass luminance at ≈0.40 (asphalt
    // [0.05,0.05,0.06] vs grass [0.09,0.16,0.05] through the same 0.3/0.59/0.11 weights terrain.ts's `lit`
    // uses), so 0.75 leaves comfortable margin while still ruling out "no overlay reached the frame".
    expect(
        capture.onRoadLum,
        `on-road ${capture.onRoadLum.toFixed(1)} vs off-road ${capture.offRoadLum.toFixed(1)}`,
    ).toBeLessThan(capture.offRoadLum * 0.75);

    expect(
        capture.transitionPx,
        `boundary transition ${capture.transitionPx.toFixed(2)}px over a ${capture.spanPx.toFixed(2)}px probe span (tolerance ${capture.tolerancePx}px)`,
    ).toBeLessThanOrEqual(capture.tolerancePx);

    expect(errors, errors.join("\n")).toEqual([]);

    // Phase 3: stage 14's reseed-integrity device arm (spec Validation, "Reseed integrity" — device half).
    // F9 twice via `__roadsRegenerate` (a deterministic bridge onto the same `regenerate()` the real F9
    // handler calls, `boot.ts`), rather than real random keypresses: a live F9 draws a fresh
    // `Math.random()` seed every press, and this stage's `overlay/queue.test.ts` already covers the
    // reset mechanics device-free — what only the device can show is that the *composite* actually reads
    // the reset indirection. Two fixed seeds chosen so neither reseed's own network coincidentally
    // re-touches the boot network's on-road tile (device-free, `overlay/network.test.ts`'s "stage 14's
    // device-arm reseed seeds" pin) — a coincidental real road there would make a still-stale read pass
    // by accident.
    const ReseedSeedA = 111111;
    const ReseedSeedB = 222222;

    for (const seed of [ReseedSeedA, ReseedSeedB]) {
        await page.evaluate(
            (s) =>
                (
                    window as unknown as { __roadsRegenerate: (seed: number) => Promise<void> }
                ).__roadsRegenerate(s),
            seed,
        );
        await page.waitForFunction(
            () => (window as unknown as { __roadsOverlayIdle: () => boolean }).__roadsOverlayIdle(),
            null,
            { timeout: 10_000 },
        );
    }

    // the boot network's on-road world (x, z) is fixed, but its surface height isn't — the old network's
    // flatten target is gone once the live document swaps twice, so re-derive the real generated height
    // there (`__roadsHeightAt`) rather than reusing the stale flattened point from Phase 2.
    const staleWorldPoint = (await page.evaluate(
        (xz) =>
            (
                window as unknown as {
                    __roadsHeightAt: (x: number, z: number) => Promise<[number, number, number]>;
                }
            ).__roadsHeightAt(xz[0], xz[1]),
        [onRoad[0], onRoad[2]] as [number, number],
    )) as [number, number, number];

    const [staleScreen] = (await page.evaluate(
        (points) =>
            (
                window as unknown as {
                    __roadsProbe: (pts: [number, number, number][]) => ScreenPoint[];
                }
            ).__roadsProbe(points),
        [staleWorldPoint],
    )) as ScreenPoint[];

    const staleScreenshot = await page.screenshot();
    const staleCapture = await page.evaluate(
        async ({ base64, staleScreen, offRoadScreen }) => {
            const binary = atob(base64);
            const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
            const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
            const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
            const context = canvas.getContext("2d");
            if (!context) throw new Error("roads capture: 2D screenshot context unavailable");
            context.drawImage(bitmap, 0, 0);
            const data = context.getImageData(0, 0, bitmap.width, bitmap.height).data;

            const luminanceAt = (fracX: number, fracY: number): number => {
                const x = Math.min(bitmap.width - 1, Math.max(0, Math.round(fracX * bitmap.width)));
                const y = Math.min(
                    bitmap.height - 1,
                    Math.max(0, Math.round(fracY * bitmap.height)),
                );
                const at = (y * bitmap.width + x) * 4;
                return 0.3 * data[at] + 0.59 * data[at + 1] + 0.11 * data[at + 2];
            };

            return {
                staleLum: luminanceAt(staleScreen.x, staleScreen.y),
                offRoadLum: luminanceAt(offRoadScreen.x, offRoadScreen.y),
            };
        },
        {
            base64: staleScreenshot.toString("base64"),
            staleScreen,
            offRoadScreen,
        },
    );

    // the same road-vs-terrain luminance ratio the Phase 2 assertion above uses, inverted: a tile still
    // stuck with the old network's road albedo would read dark, < offRoadLum * 0.75 (asserted true for a
    // real road, above); a correctly invalidated tile reads as bare terrain, at or above that ratio.
    expect(
        staleCapture.staleLum,
        `stale boot-network on-road point ${staleCapture.staleLum.toFixed(1)} vs off-road ${staleCapture.offRoadLum.toFixed(1)} — still reads as road after two reseeds`,
    ).toBeGreaterThanOrEqual(staleCapture.offRoadLum * 0.75);

    expect(errors, errors.join("\n")).toEqual([]);
});
