import { test } from "@playwright/test";
import { PNG } from "pngjs";
import { defineFlow } from "../runner";
import { S } from "../selectors";

// Regression: an app's UI must stay inside the canvas region, never spill into the host chrome (the bug
// where orrstead's HUD painted over the editor viewport). The fixture (scripts/capture/ui) mounts a
// deliberately invalid HUD through `config.ui` — `position: fixed`, 4000×4000 — and the engine's sandboxed
// overlay (`contain: layout paint` + `overflow: hidden`, engine/app/index.ts `mountOverlay`) must bound +
// clip it to the canvas. The canvas is inset 64px from the window on all sides, so any pixel in that
// border is host chrome. We assert the chrome stays teal (not magenta) and that the UI did render inside
// the frame. Strip the `contain`/`overflow` from `mountOverlay` and the magenta covers the chrome → red.

const UI_PORT = process.env.CAPTURE_UI_PORT || "3009";
const MAGENTA: [number, number, number] = [255, 0, 255];
const near = (a: number, b: number, t = 40): boolean => Math.abs(a - b) <= t;
const isMagenta = (r: number, g: number, b: number): boolean =>
    near(r, MAGENTA[0]) && near(g, MAGENTA[1]) && near(b, MAGENTA[2]);

defineFlow(
    { name: "ui-containment", scene: "ui", target: "manual" },
    async ({ page, step, assert }) => {
        test.setTimeout(30_000);
        await page.goto(`http://localhost:${UI_PORT}`);
        await page.waitForFunction(
            () => {
                const c = document.querySelector("canvas");
                return (
                    (window as Window & { __uiReady?: boolean }).__uiReady === true &&
                    c != null &&
                    c.width > 0 &&
                    c.height > 0
                );
            },
            null,
            { timeout: 10000 },
        );
        await page.waitForTimeout(300);

        const { width: w, height: h, data } = PNG.sync.read(await page.screenshot());
        const at = (x: number, y: number): [number, number, number] => {
            const i = (y * w + x) * 4;
            return [data[i], data[i + 1], data[i + 2]];
        };

        // points well inside the 64px chrome border (and the top-center strip directly above the canvas) —
        // all host chrome the contained UI must not reach
        const chrome: [number, number][] = [
            [8, 8],
            [w - 9, 8],
            [8, h - 9],
            [w - 9, h - 9],
            [Math.floor(w / 2), 8],
        ];
        for (const [x, y] of chrome) {
            const [r, g, b] = at(x, y);
            assert(
                `host chrome at (${x},${y}) is untouched by the app UI (rgb ${r},${g},${b})`,
                !isMagenta(r, g, b),
            );
        }

        // positive: the UI actually mounted + rendered, inside the canvas frame (window center)
        const [cr, cg, cb] = at(Math.floor(w / 2), Math.floor(h / 2));
        assert(
            `the app UI rendered inside the canvas region (rgb ${cr},${cg},${cb})`,
            isMagenta(cr, cg, cb),
        );

        await step("contained", { highlight: [S.canvas] });
    },
);
