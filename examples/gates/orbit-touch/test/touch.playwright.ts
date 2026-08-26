import { expect, type Page, test } from "@playwright/test";
import { oneFingerDrag, pinch, twoFingerDrag, twoToOneFingerDrag } from "../touch-dispatch";

// S4's touch verification gate (spec: `shallot-mobile-controls`). Drives real CDP touch against the
// fixture app (`../src/main.ts`: an Orbit camera targeting a box) and reads the orbit pose back through
// `window.__orbitPose()`, which folds `harness.read()`'s camera Transform pose together with the public
// `Orbit` component's own target pose (yaw/pitch/distance/pan) — the direct-component half is what lets
// each assertion isolate its gesture's own quantity: `harness.read()`'s Transform pos/quat is the
// *smoothed* rendered pose (`smoothLerp`, extras/orbit) and lags a live drag by several frames, while
// `Orbit.yaw`/`distance`/`pan` are the immediate per-frame target the drag/pinch/pan math writes.
//
// Each phase asserts one gesture moves its own quantity/families and holds every other one: pinch →
// distance, one-finger drag → yaw AND pitch (a diagonal drag, so pitch is exercised as a moving quantity
// too, not just a held one), two-finger drag → pan — matching three.js OrbitControls' / Babylon's touch
// map (spec Locked decision). Phase 4 covers S3's residue: the capturing finger lifts mid-gesture (2→1
// transition) — a `setPointerCapture` handoff review flagged as spec-plausible but unprovable in the bun
// mock harness, confirmed here on real touch.
//
// Run by path — `cd examples/gates/orbit-touch && bunx playwright test` (or `bun run gate` in that
// dir), display-gated (WSL bridge, `playwright.global-setup.ts`) — never part of the default
// `bun run test` sweep.

// Software rasterizers by the name they report in `GPUAdapterInfo` — the same display-gate pattern
// `examples/showcase/roads/test/roads.playwright.ts` uses (that file's header has the full rationale).
const SOFTWARE = /swiftshader|llvmpipe|lavapipe|warp|basic render/i;

const adapterName = (page: Page): Promise<string> =>
    page.evaluate(async () => {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter) return "";
        const { vendor, architecture, device, description } = adapter.info;
        return [vendor, architecture, device, description].filter(Boolean).join(" ");
    });

interface OrbitPose {
    yaw: number;
    pitch: number;
    distance: number;
    pan: [number, number, number];
    pos: [number, number, number] | null;
    quat: [number, number, number, number] | null;
}

const readPose = (page: Page): Promise<OrbitPose> =>
    page.evaluate(() => {
        const pose = (window as unknown as { __orbitPose: () => OrbitPose }).__orbitPose();
        if (!pose) throw new Error("__orbitPose() returned null — no Orbit entity in the scene");
        return pose;
    });

// how far a family's own quantity must move to count as "moved" — comfortably above the smoothed
// re-projection noise `harness.read()`'s pos/quat can carry between frames, well below a single real
// gesture's travel (the drags/pinch below cover tens of CSS pixels).
const MoveEps = 0.02;
// how little a family's OTHER two quantities may drift for a gesture to count as isolated — nonzero
// because floating-point orbit math on a live camera pose (world → ortho projection, smoothLerp) isn't
// bit-exact across frames, but far under the deliberate movement above.
const HoldEps = 0.01;

test("orbit touch gate — pinch/drag/pan isolation, 2→1 finger transition (real touch)", async ({
    page,
}) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    await page.goto("/");

    const adapter = await adapterName(page);
    console.log(`orbit-touch gate adapter: ${adapter || "none offered"}`);
    test.skip(
        adapter === "" || SOFTWARE.test(adapter),
        `no real-GPU adapter (${adapter || "none offered"})`,
    );

    await page.waitForFunction(
        () => typeof (window as unknown as { __orbitPose?: unknown }).__orbitPose === "function",
        null,
        {
            timeout: 60_000,
        },
    );

    const viewport = page.viewportSize();
    if (!viewport) throw new Error("no viewport — mobile device context did not apply");
    const center = { x: viewport.width / 2, y: viewport.height / 2 };

    const cdp = await page.context().newCDPSession(page);

    // --- Phase 1: one-finger drag rotates (yaw AND pitch move — a diagonal drag, so pitch is
    // exercised as a moving quantity here rather than only ever asserted held; distance/pan hold) ---
    const before1 = await readPose(page);
    await oneFingerDrag(cdp, center, { x: center.x + 80, y: center.y - 50 }, 12);
    const after1 = await readPose(page);

    expect(
        Math.abs(after1.yaw - before1.yaw),
        `one-finger drag: yaw ${before1.yaw} → ${after1.yaw}`,
    ).toBeGreaterThan(MoveEps);
    expect(
        Math.abs(after1.pitch - before1.pitch),
        `one-finger drag: pitch ${before1.pitch} → ${after1.pitch}`,
    ).toBeGreaterThan(MoveEps);
    expect(
        Math.abs(after1.distance - before1.distance),
        `one-finger drag: distance moved (${before1.distance} → ${after1.distance}) — should hold`,
    ).toBeLessThan(HoldEps);
    expect(
        Math.hypot(...after1.pan.map((v, i) => v - before1.pan[i])),
        `one-finger drag: pan moved (${before1.pan} → ${after1.pan}) — should hold`,
    ).toBeLessThan(HoldEps);

    expect(errors, errors.join("\n")).toEqual([]);

    // --- Phase 2: pinch zooms (distance moves; yaw/pitch/pan hold) ---
    const before2 = await readPose(page);
    await pinch(cdp, center, 40, 140, 12); // spreading — zooms in (Touch.pinchDelta positive)
    const after2 = await readPose(page);

    expect(
        Math.abs(after2.distance - before2.distance),
        `pinch: distance ${before2.distance} → ${after2.distance}`,
    ).toBeGreaterThan(MoveEps);
    expect(
        Math.abs(after2.yaw - before2.yaw),
        `pinch: yaw moved (${before2.yaw} → ${after2.yaw}) — should hold`,
    ).toBeLessThan(HoldEps);
    expect(
        Math.abs(after2.pitch - before2.pitch),
        `pinch: pitch moved (${before2.pitch} → ${after2.pitch}) — should hold`,
    ).toBeLessThan(HoldEps);
    expect(
        Math.hypot(...after2.pan.map((v, i) => v - before2.pan[i])),
        `pinch: pan moved (${before2.pan} → ${after2.pan}) — should hold`,
    ).toBeLessThan(HoldEps);

    expect(errors, errors.join("\n")).toEqual([]);

    // --- Phase 3: two-finger drag pans (pan moves; yaw/pitch/distance hold) ---
    const before3 = await readPose(page);
    await twoFingerDrag(cdp, center, 60, { x: 90, y: 40 }, 12);
    const after3 = await readPose(page);

    expect(
        Math.hypot(...after3.pan.map((v, i) => v - before3.pan[i])),
        `two-finger drag: pan ${before3.pan} → ${after3.pan}`,
    ).toBeGreaterThan(MoveEps);
    expect(
        Math.abs(after3.yaw - before3.yaw),
        `two-finger drag: yaw moved (${before3.yaw} → ${after3.yaw}) — should hold`,
    ).toBeLessThan(HoldEps);
    expect(
        Math.abs(after3.pitch - before3.pitch),
        `two-finger drag: pitch moved (${before3.pitch} → ${after3.pitch}) — should hold`,
    ).toBeLessThan(HoldEps);
    expect(
        Math.abs(after3.distance - before3.distance),
        `two-finger drag: distance moved (${before3.distance} → ${after3.distance}) — should hold`,
    ).toBeLessThan(HoldEps);

    expect(errors, errors.join("\n")).toEqual([]);

    // --- Phase 4: 2→1 finger transition — pan with two fingers, lift one, keep rotating with the
    // survivor. The pre-fix defect (S3 review, `standard/input/`): the capturing finger's release never
    // re-captured the survivor, freezing the gesture instead of handing off to a one-finger rotate. Both
    // halves must show real movement — the pan from the two-finger portion, then a FURTHER yaw change
    // from the one-finger portion — proving the handoff continued rather than froze.
    const before4 = await readPose(page);
    // lift "a" — the first-down, capturing finger (`touch-dispatch.ts`'s docblock) — so the handoff
    // actually exercises `recaptureTouch` rather than leaving the still-captured survivor untouched.
    await twoToOneFingerDrag(cdp, center, 60, { x: 70, y: 0 }, { x: 70, y: 0 }, "a", 8);
    const afterPan4 = await readPose(page);

    expect(
        Math.hypot(...afterPan4.pan.map((v, i) => v - before4.pan[i])),
        `2→1 transition: pan didn't move from the two-finger half (${before4.pan} → ${afterPan4.pan})`,
    ).toBeGreaterThan(MoveEps);

    // the survivor's one-finger rotate lands a few animation frames after the drag helper's own last
    // dispatch — poll the observable (yaw's own delta from before4) rather than a fixed sleep (kex
    // coding.md Testing → Forbidden: no waitForTimeout). A frozen handoff (the pre-fix defect) never
    // clears this poll and times out rather than silently reading a stale pose.
    await expect
        .poll(async () => Math.abs((await readPose(page)).yaw - before4.yaw), {
            message:
                `2→1 transition: yaw froze after the finger lift (baseline ${before4.yaw}) — the ` +
                `survivor's one-finger rotate never continued the gesture`,
            timeout: 2_000,
        })
        .toBeGreaterThan(MoveEps);

    expect(errors, errors.join("\n")).toEqual([]);
});
