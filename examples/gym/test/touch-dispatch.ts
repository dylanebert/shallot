import type { CDPSession } from "@playwright/test";

// Chromium-only, by construction: this dispatches real touch input through the CDP `Input` domain
// (`Input.dispatchTouchEvent`), which only Chromium-family browsers expose over Playwright's CDP
// session — Firefox and WebKit have no equivalent wire protocol this module could target. Playwright's
// own `page.touchscreen` only exposes `tap()`, and synthetic `dispatchEvent` PointerEvents bypass the
// browser's native touch→pointer translation entirely, so neither can prove the
// `touch-action`/listener setup the engine's input substrate depends on actually works
// (`shallot-mobile-controls` spec, Locked decision). CDP is the integration-honest instrument instead —
// acceptable because the verify harness this gate extends is already Chromium-only.
//
// CDP's touch model, empirically probed (the protocol doc's "TouchEnd/TouchCancel must not contain any
// touch points" describes only the ends-everyone shape, not the whole contract): `touchStart`/
// `touchMove` list every point CURRENTLY down (a point missing from `touchMove` relative to the prior
// event is NOT released — probed directly, `touch.count` stayed at 2 after a `touchMove` naming only
// one of two active points). A `touchEnd` names the point(s) that are LIFTING — `changedTouches`, not
// `touches` — so `touchEnd` with every active point (or `[]`, Playwright's own `RawTouchscreenImpl.tap`
// shape) ends the whole gesture, while `touchEnd` naming only ONE of several active points lifts that
// finger alone and leaves the rest down — the mechanism a 2→1 finger transition dispatches through.

export interface TouchPoint {
    x: number;
    y: number;
    id: number;
}

async function dispatch(
    cdp: CDPSession,
    type: "touchStart" | "touchMove" | "touchEnd",
    points: TouchPoint[],
): Promise<void> {
    await cdp.send("Input.dispatchTouchEvent", {
        type,
        touchPoints: points.map((p) => ({ x: p.x, y: p.y, id: p.id })),
    });
}

/** begin contact for every point in `points` (fresh `id`s — a gesture's whole finger set). */
export async function touchStart(cdp: CDPSession, points: TouchPoint[]): Promise<void> {
    await dispatch(cdp, "touchStart", points);
}

/** move every currently-down point to its new position (every `id` still held must be listed — a
 *  missing `id` here does NOT release it, probed above). */
export async function touchMove(cdp: CDPSession, points: TouchPoint[]): Promise<void> {
    await dispatch(cdp, "touchMove", points);
}

/** end contact for `lifting` (the point(s) going up, at their last known position) — omit it (or pass
 *  `[]`) to end the whole gesture at once. Naming only SOME of the currently-active points lifts just
 *  those fingers, leaving the rest down (the 2→1 transition's release step). */
export async function touchEnd(cdp: CDPSession, lifting: TouchPoint[] = []): Promise<void> {
    await dispatch(cdp, "touchEnd", lifting);
}

// small pause between dispatch steps so the page's rAF loop gets a chance to consume each frame's
// accumulated pointer deltas (Inputs.mouse/touch reset their deltas at frame end, standard/input) —
// dispatching every step back-to-back with no yield can coalesce a whole gesture into events the
// browser's own event loop hasn't drained between animation frames.
const STEP_MS = 32;

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

function interpolate(
    from: { x: number; y: number },
    to: { x: number; y: number },
    steps: number,
): { x: number; y: number }[] {
    const out: { x: number; y: number }[] = [];
    for (let i = 1; i <= steps; i++) {
        out.push({ x: lerp(from.x, to.x, i / steps), y: lerp(from.y, to.y, i / steps) });
    }
    return out;
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
    await touchStart(cdp, [{ ...from, id }]);
    await wait(STEP_MS);
    for (const p of interpolate(from, to, steps)) {
        await touchMove(cdp, [{ ...p, id }]);
        await wait(STEP_MS);
    }
    await touchEnd(cdp);
}

/** two-finger pinch about `center`: both fingers move radially from `startRadius` to `endRadius` along
 *  opposite directions on the same axis, then lift together. Positive `endRadius - startRadius` spreads
 *  (zoom in); negative pinches together (zoom out) — matches `Touch.pinchDelta`'s sign convention. */
export async function pinch(
    cdp: CDPSession,
    center: { x: number; y: number },
    startRadius: number,
    endRadius: number,
    steps = 10,
): Promise<void> {
    const idA = 1;
    const idB = 2;
    const at = (r: number): TouchPoint[] => [
        { x: center.x - r, y: center.y, id: idA },
        { x: center.x + r, y: center.y, id: idB },
    ];
    await touchStart(cdp, at(startRadius));
    await wait(STEP_MS);
    for (let i = 1; i <= steps; i++) {
        const r = lerp(startRadius, endRadius, i / steps);
        await touchMove(cdp, at(r));
        await wait(STEP_MS);
    }
    await touchEnd(cdp);
}

/** two-finger drag: both fingers move together by the same delta, then lift together. */
export async function twoFingerDrag(
    cdp: CDPSession,
    center: { x: number; y: number },
    spacing: number,
    delta: { x: number; y: number },
    steps = 10,
): Promise<void> {
    const idA = 1;
    const idB = 2;
    const at = (cx: number, cy: number): TouchPoint[] => [
        { x: cx - spacing, y: cy, id: idA },
        { x: cx + spacing, y: cy, id: idB },
    ];
    await touchStart(cdp, at(center.x, center.y));
    await wait(STEP_MS);
    for (const p of interpolate(center, { x: center.x + delta.x, y: center.y + delta.y }, steps)) {
        await touchMove(cdp, at(p.x, p.y));
        await wait(STEP_MS);
    }
    await touchEnd(cdp);
}

/**
 * two fingers drag together, then ONE lifts mid-gesture (a `touchEnd` naming only that finger — probed
 * above — leaves the other down) and drags on alone before lifting too — the 2→1 finger
 * `setPointerCapture` handoff S3's review flagged as spec-plausible but unprovable in the bun mock
 * harness (`shallot-mobile-controls` spec, Residue). `lift` names which finger goes UP ("a" or "b"),
 * the other is the survivor — pass "a" to exercise the regression S3 fixed (the input substrate's
 * pointer capture always starts on the first-down finger, `touchStart`'s array order, so lifting "a"
 * forces the `recaptureTouch` handoff onto "b"; lifting "b" instead leaves the still-captured "a"
 * untouched and never exercises the handoff).
 */
export async function twoToOneFingerDrag(
    cdp: CDPSession,
    center: { x: number; y: number },
    spacing: number,
    twoFingerDelta: { x: number; y: number },
    oneFingerDelta: { x: number; y: number },
    lift: "a" | "b" = "a",
    steps = 8,
): Promise<void> {
    const idA = 1;
    const idB = 2;
    const at = (cx: number, cy: number): TouchPoint[] => [
        { x: cx - spacing, y: cy, id: idA },
        { x: cx + spacing, y: cy, id: idB },
    ];
    await touchStart(cdp, at(center.x, center.y));
    await wait(STEP_MS);

    let last = center;
    for (const p of interpolate(
        center,
        { x: center.x + twoFingerDelta.x, y: center.y + twoFingerDelta.y },
        steps,
    )) {
        await touchMove(cdp, at(p.x, p.y));
        await wait(STEP_MS);
        last = p;
    }

    // lift the chosen finger — a `touchEnd` naming just that point's `id` — leaving the other down.
    // `a` sits left of center, `b` right; the survivor keeps its own side's last position.
    const liftedX = lift === "a" ? last.x - spacing : last.x + spacing;
    const liftedId = lift === "a" ? idA : idB;
    const survivorX = lift === "a" ? last.x + spacing : last.x - spacing;
    const survivorId = lift === "a" ? idB : idA;
    await touchEnd(cdp, [{ x: liftedX, y: last.y, id: liftedId }]);
    await wait(STEP_MS);

    const survivor: TouchPoint = { x: survivorX, y: last.y, id: survivorId };
    for (const p of interpolate(
        survivor,
        { x: survivor.x + oneFingerDelta.x, y: survivor.y + oneFingerDelta.y },
        steps,
    )) {
        await touchMove(cdp, [{ x: p.x, y: p.y, id: survivorId }]);
        await wait(STEP_MS);
    }
    await touchEnd(cdp);
}
