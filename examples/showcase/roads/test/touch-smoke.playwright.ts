import { assertMotion, isDegradedBootMessage } from "@dylanebert/shallot/harness";
import { expect, type Page, test } from "@playwright/test";
import { PNG } from "pngjs";
import { oneFingerDrag } from "./touch-drag";

test.use({ hasTouch: true, isMobile: true, viewport: { width: 390, height: 844 } });
const FrameChangeThreshold = 3;

async function sampleCanvas(page: Page) {
    const png = await page.locator("canvas").first().screenshot();
    const channels: number[] = await page.evaluate(async (base64) => {
        const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
        const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
        const surface = new OffscreenCanvas(32, 32);
        const ctx = surface.getContext("2d");
        if (!ctx) throw new Error("touch smoke: 2D context unavailable");
        ctx.drawImage(bitmap, 0, 0, 32, 32);
        bitmap.close();
        return Array.from(ctx.getImageData(0, 0, 32, 32).data);
    }, png.toString("base64"));
    return { png, channels };
}

const snapshot = (page: Page) =>
    page.evaluate(async () => {
        const h = await import("/src/" + "harness.ts");
        const edit = await import("/src/" + "edit.ts");
        const terrain = await import("/src/terrain/" + "terrain.ts");
        return { ...h.cameraSnapshot(), pick: edit.editSnapshot(), doc: terrain.getDocument() };
    });

test("roads touch motion rejects no-input, disabled-handler and frozen-camera controls", async ({
    page,
}, info) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));
    page.on("console", (m) => {
        if (m.type() === "error" || isDegradedBootMessage(m.text())) errors.push(m.text());
    });
    const results: any[] = [];
    for (const control of ["drag", "no-input", "disabled-handler", "frozen-camera"]) {
        await page.goto("/");
        await page.waitForFunction(() => window.__roadsOverlayIdle?.() === true);
        const adapter = await page.evaluate(async () => {
            const a = await navigator.gpu?.requestAdapter();
            return a
                ? [a.info.vendor, a.info.architecture, a.info.device, a.info.description].join(" ")
                : "";
        });
        expect(adapter).not.toBe("");
        expect(adapter).not.toMatch(/swiftshader|llvmpipe|lavapipe|warp|basic render/i);
        await page.evaluate(async (control) => {
            const h = await import("/src/" + "harness.ts");
            if (control === "frozen-camera") h.poseCamera({ smoothness: 0 });
            const receipts: unknown[] = [];
            (window as any).__touchReceipts = receipts;
            for (const type of [
                "pointerdown",
                "pointermove",
                "pointerup",
                "pointercancel",
                "touchstart",
                "touchmove",
                "touchend",
            ]) {
                window.addEventListener(
                    type,
                    (event) => {
                        receipts.push({
                            type: event.type,
                            trusted: event.isTrusted,
                            target: (event.target as Element)?.tagName,
                            time: event.timeStamp,
                        });
                    },
                    { capture: true },
                );
                if (control === "disabled-handler") {
                    document.addEventListener(type, (event) => event.stopImmediatePropagation(), {
                        capture: true,
                    });
                }
            }
            await h.frames(3);
        }, control);
        const beforeState = await snapshot(page);
        const before = await sampleCanvas(page);
        const canvas = page.locator("canvas").first();
        const box = (await canvas.boundingBox())!;
        const from = { x: box.x + box.width / 2 - 80, y: box.y + box.height / 2 };
        const to = { x: from.x + 160, y: from.y + 40 };
        const target = await page.evaluate(
            (p) => document.elementFromPoint(p.x, p.y)?.tagName,
            from,
        );
        expect(target).toBe("CANVAS");
        let firstFrame: Awaited<ReturnType<typeof snapshot>> | null = null;
        if (control !== "no-input") {
            const cdp = await page.context().newCDPSession(page);
            try {
                await oneFingerDrag(cdp, from, to, 10, async () => {
                    await page.evaluate(async () =>
                        (await import("/src/" + "harness.ts")).frames(2),
                    );
                    firstFrame = await snapshot(page);
                });
            } finally {
                await cdp.detach();
            }
        }
        // A fixed frame observation window is shared by positives and negatives, not a retry-to-pass.
        await page.evaluate(async () => (await import("/src/" + "harness.ts")).frames(30));
        const after = await sampleCanvas(page);
        const afterState = await snapshot(page);
        const events = await page.evaluate(() => (window as any).__touchReceipts);
        const numerator = before.channels.reduce(
            (sum, v, i) => sum + Math.abs(v - after.channels[i]),
            0,
        );
        const denominator = before.channels.length;
        let motion = false;
        try {
            assertMotion(before.channels, after.channels, FrameChangeThreshold);
            motion = true;
        } catch (error) {
            if (!String(error).includes("samples are parked")) throw error;
        }
        // Independent immutable-frame truth: full-resolution PNG decode, not the 32x32 predicate.
        const a = PNG.sync.read(before.png),
            b = PNG.sync.read(after.png);
        expect([a.width, a.height]).toEqual([b.width, b.height]);
        let changedChannels = 0,
            fullDifference = 0;
        for (let i = 0; i < a.data.length; i++) {
            const d = Math.abs(a.data[i] - b.data[i]);
            if (d) changedChannels++;
            fullDifference += d;
        }
        const cameraBefore = beforeState.cameras.find((c: any) => c.canvas);
        const cameraAfter = afterState.cameras.find((c: any) => c.canvas);
        const cameraMoved =
            JSON.stringify([cameraBefore.pos, cameraBefore.rot]) !==
            JSON.stringify([cameraAfter.pos, cameraAfter.rot]);
        const yawMoved = cameraBefore.yaw !== cameraAfter.yaw;
        const result = {
            control,
            adapter,
            box,
            from,
            to,
            target,
            firstFrame,
            numerator,
            denominator,
            mean: numerator / denominator,
            motion,
            cameraMoved,
            yawMoved,
            changedChannels,
            fullDifference,
            beforeState,
            afterState,
            events,
        };
        results.push(result);
        await info.attach(`${control}-before.png`, { body: before.png, contentType: "image/png" });
        await info.attach(`${control}-after.png`, { body: after.png, contentType: "image/png" });
        await info.attach(`${control}.json`, {
            body: JSON.stringify({ ...result, before: before.channels, after: after.channels }),
            contentType: "application/json",
        });
        console.log("TOUCH_CONTROL", JSON.stringify(result));
    }
    for (const r of results) {
        expect(r.denominator).toBe(4096);
        expect(r.afterState.doc, `${r.control}: orbit must not edit the road`).toEqual(
            r.beforeState.doc,
        );
        if (r.firstFrame) {
            expect(
                r.firstFrame.pick.grab.dragging,
                `${r.control}: first press is not a handle hit`,
            ).toBe(false);
            expect(
                r.firstFrame.pick.ray.origin
                    .concat(r.firstFrame.pick.ray.dir)
                    .every(Number.isFinite),
            ).toBe(true);
        }
        expect(r.motion, `${r.control}: ${r.numerator}/${r.denominator}`).toBe(
            r.control === "drag",
        );
        expect(r.cameraMoved, r.control).toBe(r.control === "drag");
        expect(r.yawMoved, r.control).toBe(r.control === "drag" || r.control === "frozen-camera");
        if (r.control === "no-input") expect(r.events).toEqual([]);
        else {
            expect(
                r.events.some(
                    (e: any) => e.type === "pointerdown" && e.target === "CANVAS" && e.trusted,
                ),
                r.control,
            ).toBe(true);
            expect(
                r.events.some((e: any) => e.type === "pointermove" && e.trusted),
                r.control,
            ).toBe(true);
        }
        if (r.control === "drag") expect(r.changedChannels).toBeGreaterThan(0);
        else expect(r.fullDifference, r.control).toBe(0);
    }
    expect(errors, errors.join("\n")).toEqual([]);
});
