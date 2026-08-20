import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";
import { captureProbePoints, generateNetwork } from "../src/overlay/network";
import { makePermutation } from "../src/terrain/noise";
import { buildPolylineProfile, heightAtCpu } from "../src/terrain/profile";

// The boot seed (`terrain.ts`'s `SEED = 1337`) — the network the corridor pose is derived from and
// the terrain that must be live when the corridor capture is taken. Not imported from `terrain.ts`
// because that module imports `@dylanebert/shallot` at module top level, which Node ≥26 rejects
// (bare `package.json` import) — the Playwright driver stays clear of that package on the Node side.
const BOOT_SEED = 1337;

// where every run's capture lands — a fixed, git-ignored path (`test-results/`, this project's
// `.gitignore`) rather than a per-run timestamped name, so "the latest capture" always has one findable
// location for the spec's human release-look gate to read.
const CAPTURE_PATH = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test-results",
    "roads-capture.png",
);

// Stage 24a's corridor-pose capture — a second file alongside the gate's own, written at the derived
// corridor pose (corridorPose.ts). The default-orbit capture above stays unchanged in pose; this one
// is the admissible artifact for 24b's release look.
const CORRIDOR_CAPTURE_PATH = join(
    dirname(fileURLToPath(import.meta.url)),
    "..",
    "test-results",
    "roads-corridor-capture.png",
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

    // In-frame assertion: each probe's fractional screen coordinates must be strictly inside (0, 1)
    // before any pixel is read — `luminanceAt` clamps out-of-range coordinates to the border, so an
    // off-screen probe silently reads a border pixel and passes instead of reding (the same class as
    // the unreachable null control, `checks.md`'s audit-check-vacuity). Measured on the shipped default
    // (distance 120 m): onRoad sits at 0.905/0.464 and offRoad at 0.929/0.492 of frame — on-screen, but
    // one small camera change from the clamp. Red-first demonstrated: tightening the x-bound to
    // `toBeLessThan(0.9)` reds on the offRoad probe (x=0.917), exit 1 — the assertion fires before
    // `luminanceAt` can clamp.
    for (const [name, sp] of [
        ["onRoad", onRoadScreen],
        ["offRoad", offRoadScreen],
    ] as const) {
        expect(sp.x, `${name} probe x=${sp.x} is not in-frame (0, 1)`).toBeGreaterThan(0);
        expect(sp.x, `${name} probe x=${sp.x} is not in-frame (0, 1)`).toBeLessThan(1);
        expect(sp.y, `${name} probe y=${sp.y} is not in-frame (0, 1)`).toBeGreaterThan(0);
        expect(sp.y, `${name} probe y=${sp.y} is not in-frame (0, 1)`).toBeLessThan(1);
    }

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
    //
    // Two complementary probes, not one: the old boot network's on-road point must stop reading road
    // (`invalidate()` actually released it), *and* the final reseeded network's own on-road point must
    // start reading road (the swapped-in document actually queued and drained). An arm that only checked
    // the first half passed even when `regenerate()`'s two calls ran in the wrong order — `markDirty`
    // then `invalidate` wipes the very pending-queue entries `markDirty` just pushed, so the old road
    // vanishes for the wrong reason (nothing ever redraws again) and the first probe alone can't tell the
    // difference (adversarial review, 2026-08-19). The final network's on-road point is derived the same
    // way `overlay/network.test.ts` derives its disjointness pin — real `captureProbePoints` against the
    // real reseed seed, not a hand-picked coordinate.
    const ReseedSeedA = 111111;
    const ReseedSeedB = 233332;

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

    const { onRoad: newRoad } = captureProbePoints(ReseedSeedB);

    // both probed world (x, z) are fixed, but their surface heights aren't — the old point's flatten
    // target is gone and the new point's has just been baked — so re-derive the real generated height at
    // each (`__roadsHeightAt`) rather than reusing Phase 2's stale flattened point.
    const [staleWorldPoint, newRoadWorldPoint] = (await page.evaluate(
        (points) =>
            Promise.all(
                points.map((xz) =>
                    (
                        window as unknown as {
                            __roadsHeightAt: (
                                x: number,
                                z: number,
                            ) => Promise<[number, number, number]>;
                        }
                    ).__roadsHeightAt(xz[0], xz[1]),
                ),
            ),
        [
            [onRoad[0], onRoad[2]],
            [newRoad[0], newRoad[1]],
        ] as [number, number][],
    )) as [number, number, number][];

    const [staleScreen, newRoadScreen] = (await page.evaluate(
        (points) =>
            (
                window as unknown as {
                    __roadsProbe: (pts: [number, number, number][]) => ScreenPoint[];
                }
            ).__roadsProbe(points),
        [staleWorldPoint, newRoadWorldPoint],
    )) as ScreenPoint[];

    // In-frame assertion for the reseed probes (same rationale as Phase 2's pair above).
    for (const [name, sp] of [
        ["stale", staleScreen],
        ["newRoad", newRoadScreen],
    ] as const) {
        expect(sp.x, `${name} probe x=${sp.x} is not in-frame (0, 1)`).toBeGreaterThan(0);
        expect(sp.x, `${name} probe x=${sp.x} is not in-frame (0, 1)`).toBeLessThan(1);
        expect(sp.y, `${name} probe y=${sp.y} is not in-frame (0, 1)`).toBeGreaterThan(0);
        expect(sp.y, `${name} probe y=${sp.y} is not in-frame (0, 1)`).toBeLessThan(1);
    }

    const reseedScreenshot = await page.screenshot();
    const reseedCapture = await page.evaluate(
        async ({ base64, staleScreen, newRoadScreen, offRoadScreen }) => {
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
                newRoadLum: luminanceAt(newRoadScreen.x, newRoadScreen.y),
                offRoadLum: luminanceAt(offRoadScreen.x, offRoadScreen.y),
            };
        },
        {
            base64: reseedScreenshot.toString("base64"),
            staleScreen,
            newRoadScreen,
            offRoadScreen,
        },
    );

    // complement 1: a tile still stuck with the old network's road albedo would read dark, < offRoadLum *
    // 0.75 (the same ratio Phase 2 asserts true for a real road, above); a correctly invalidated tile
    // reads as bare terrain, at or above that ratio.
    expect(
        reseedCapture.staleLum,
        `stale boot-network on-road point ${reseedCapture.staleLum.toFixed(1)} vs off-road ${reseedCapture.offRoadLum.toFixed(1)} — still reads as road after two reseeds`,
    ).toBeGreaterThanOrEqual(reseedCapture.offRoadLum * 0.75);

    // complement 2: the final reseeded network's own road must actually be there — the arm this finding
    // was missing. A `markDirty`-then-`invalidate` ordering bug passes complement 1 (the old road is gone)
    // while failing this one (the new road never queued, so nothing redrew after the swap).
    expect(
        reseedCapture.newRoadLum,
        `final-network on-road point ${reseedCapture.newRoadLum.toFixed(1)} vs off-road ${reseedCapture.offRoadLum.toFixed(1)} — doesn't read as road after the reseed that should have drawn it`,
    ).toBeLessThan(reseedCapture.offRoadLum * 0.75);

    expect(errors, errors.join("\n")).toEqual([]);

    // Restore the boot seed after Phase 3's reseeds. Phase 3 reseeds to 111111 then 233332 (both
    // deliberately chosen so neither network has a road at the probe point — that property is what
    // makes Phase 3's arm valid, and it is exactly what guarantees the corridor capture would be empty
    // if the boot seed were not restored). The corridor pose (corridorPose.ts / corridorCapture.ts) is
    // derived from `generateNetwork(1337)` via `roadFrame()` (boundaryAnchors.ts), so the camera points
    // at seed 1337's road-0 midpoint. The capture must be taken with the live terrain on the same
    // network the pose was derived from — restore the boot seed, then reposition and capture.
    await page.evaluate(
        (s) =>
            (
                window as unknown as { __roadsRegenerate: (seed: number) => Promise<void> }
            ).__roadsRegenerate(s),
        BOOT_SEED,
    );
    await page.waitForFunction(
        () => (window as unknown as { __roadsOverlayIdle: () => boolean }).__roadsOverlayIdle(),
        null,
        { timeout: 10_000 },
    );

    // Phase 4: corridor content arm — assert against the live device scene that the corridor is
    // actually in frame and set into terrain. This closes the gate the blocker proved missing: nothing
    // previously checked the corridor capture's content — the screenshot was written to a file and
    // never inspected, so a capture of empty terrain (seed 233332's, with no road at the probe point)
    // passed every gate. The arm checks two things: (1) the live height at the corridor centre matches
    // the chord's flatten target (the terrain is flattened there — a road is present), and (2) the
    // live height ~30 m to the side does not match the flatten target (unflattened terrain flanks the
    // corridor — it reads set into terrain, not flat everywhere). Both checks use the boot seed's
    // network geometry and natural heights, so a wrong-seed terrain reds both.
    const bootDoc = generateNetwork(BOOT_SEED);
    const [[roadAx, roadAz], [roadBx, roadBz]] = bootDoc.polylines[0].points;
    const roadDx = roadBx - roadAx;
    const roadDz = roadBz - roadAz;
    const roadLen = Math.hypot(roadDx, roadDz) || 1;
    const roadUx = roadDx / roadLen;
    const roadUz = roadDz / roadLen;
    const roadNx = -roadUz; // unit normal (network.ts's convention)
    const roadNz = roadUx;

    // corridor centre: road-0 midpoint (t = 0.5) — the same point corridorCapture.ts targets
    const centreX = roadAx + roadUx * roadLen * 0.5;
    const centreZ = roadAz + roadUz * roadLen * 0.5;

    // chord target at the centre: linear interpolation of the endpoints' natural heights
    const bootPerm = makePermutation(BOOT_SEED);
    const profile = buildPolylineProfile(bootDoc.polylines[0].points, bootPerm);
    const chordTarget = profile[0].height + (profile[1].height - profile[0].height) * 0.5;

    // flank: 30 m perpendicular to the road from the centre (well beyond halfWidth + falloff ≈ 10.8 m)
    const FlankOffset = 30;
    const flankX = centreX + roadNx * FlankOffset;
    const flankZ = centreZ + roadNz * FlankOffset;
    const naturalFlank = heightAtCpu(flankX, flankZ, bootPerm);

    const [liveCentre, liveFlank] = (await page.evaluate(
        (points) =>
            Promise.all(
                points.map((xz) =>
                    (
                        window as unknown as {
                            __roadsHeightAt: (
                                x: number,
                                z: number,
                            ) => Promise<[number, number, number]>;
                        }
                    ).__roadsHeightAt(xz[0], xz[1]),
                ),
            ),
        [
            [centreX, centreZ],
            [flankX, flankZ],
        ] as [number, number][],
    )) as [number, number, number][];

    // (1) The corridor centre is flattened: the live height matches the chord target (within the
    // nearest-vertex approximation — `withHeight` reads the nearest grid vertex, at most SPACING/2
    // from the centre, and the chord target varies by < 0.3 m over that distance).
    expect(
        Math.abs(liveCentre[1] - chordTarget),
        `corridor centre live height ${liveCentre[1].toFixed(2)} vs chord target ${chordTarget.toFixed(2)} — corridor is not flattened at the capture point (wrong seed?)`,
    ).toBeLessThan(0.5);

    // (2) The flank is unflattened: the live height matches the natural terrain height (within the
    // nearest-vertex approximation on natural terrain — at most ~3 m of gradient over SPACING/2).
    expect(
        Math.abs(liveFlank[1] - naturalFlank),
        `flank live height ${liveFlank[1].toFixed(2)} vs natural ${naturalFlank.toFixed(2)} — flank is not natural terrain`,
    ).toBeLessThan(3.0);

    // (3) The corridor is set into terrain: the chord target differs from the natural flank height
    // (the flatten operation cut into terrain, not flat ground).
    expect(
        Math.abs(chordTarget - naturalFlank),
        `chord target ${chordTarget.toFixed(2)} vs natural flank ${naturalFlank.toFixed(2)} — corridor is not set into terrain`,
    ).toBeGreaterThan(0.5);

    expect(errors, errors.join("\n")).toEqual([]);

    // Phase 5: stage 24a's corridor-pose capture. Reposition the orbit to the derived corridor pose
    // (corridorCapture.ts via window.__roadsCorridorCapture), wait for it to settle, and save the
    // screenshot to a second file. The default-orbit capture (Phase 2) is already saved and must not
    // move — it feeds the fs-composite pixel probes. This capture is the admissible artifact for 24b's
    // human release look: the earthwork's 1.4404 m vertical excursion projects to ≥5 px at this pose
    // (asserted device-free by corridorPose.test.ts), while ≥30 m of unflattened terrain flanks the
    // corridor in frame.
    await page.evaluate(() =>
        (
            window as unknown as { __roadsCorridorCapture: () => Promise<void> }
        ).__roadsCorridorCapture(),
    );
    await page.waitForTimeout(500); // let the render loop paint the re-posed frame
    const corridorScreenshot = await page.screenshot();
    mkdirSync(dirname(CORRIDOR_CAPTURE_PATH), { recursive: true });
    writeFileSync(CORRIDOR_CAPTURE_PATH, corridorScreenshot);

    expect(errors, errors.join("\n")).toEqual([]);
});
