import type { CDPSession } from "@playwright/test";

// Minimal CDP touch-dispatch, trimmed to the one gesture a showcase touch smoke needs (one-finger
// drag — the primary orbit interaction every demo in this sweep shares). Chromium-only by construction:
// `Input.dispatchTouchEvent` is a CDP-only surface, and `page.touchscreen` only exposes `tap()` —
// neither Firefox/WebKit nor a synthetic `dispatchEvent` PointerEvent can drive the real
// touch-action/listener path the input substrate depends on (`shallot-mobile-controls` spec, Locked
// decision). Full gesture coverage (pinch, two-finger pan/drag, the 2→1 finger transition) lives in
// gym's own copy (`examples/gym/test/touch-dispatch.ts`) — duplicated here, not imported, matching this
// corpus's own precedent for small shared shapes across example projects (`.claude/rules/examples.md`:
// recipes "never import from each other... a small shared shape duplicates rather than coupling
// entries").

const STEP_MS = 32;

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

async function wait(ms: number): Promise<void> {
    await new Promise((r) => setTimeout(r, ms));
}

/** one-finger drag: touchStart at `from`, `steps` touchMoves toward `to`, then touchEnd. */
export async function oneFingerDrag(
    cdp: CDPSession,
    from: { x: number; y: number },
    to: { x: number; y: number },
    steps = 10,
): Promise<void> {
    const id = 1;
    await cdp.send("Input.dispatchTouchEvent", {
        type: "touchStart",
        touchPoints: [{ x: from.x, y: from.y, id }],
    });
    await wait(STEP_MS);
    for (let i = 1; i <= steps; i++) {
        const x = lerp(from.x, to.x, i / steps);
        const y = lerp(from.y, to.y, i / steps);
        await cdp.send("Input.dispatchTouchEvent", {
            type: "touchMove",
            touchPoints: [{ x, y, id }],
        });
        await wait(STEP_MS);
    }
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}
