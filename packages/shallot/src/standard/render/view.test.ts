import { describe, expect, spyOn, test } from "bun:test";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Camera } from "./camera";
import { attachView, backingSize, devEnabled, pruneViews, trackCanvasOwner, Views } from "./view";

// backingSize derives a view's render backing (device px) from its CSS display size, a Resolution pin
// (resW/resH, 0 = that axis unset), and the pixelRatio fit-ratio. The aspect derivation + the
// upscale-only `pixelated` flag are the logic worth pinning.
describe("backingSize", () => {
    test("no pin (both 0) renders at display × ratio, smooth at ratio ≥ 1", () => {
        expect(backingSize(0, 0, 1920, 1080, 1)).toEqual({ w: 1920, h: 1080, pixelated: false });
        expect(backingSize(0, 0, 1920, 1080, 2)).toEqual({ w: 3840, h: 2160, pixelated: false });
    });

    test("no pin with ratio < 1 is the pixel-art downscale (pixelated)", () => {
        expect(backingSize(0, 0, 1920, 1080, 0.5)).toEqual({ w: 960, h: 540, pixelated: true });
    });

    test("height pin derives width from the display aspect, point-sampled up", () => {
        // 360 × (1920/1080) = 640 exactly; 640 < 1920 → upscaling → pixelated
        expect(backingSize(0, 360, 1920, 1080, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("width pin derives height from the display aspect", () => {
        // 640 × (1080/1920) = 360 exactly
        expect(backingSize(640, 0, 1920, 1080, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("both axes pinned are exact, even off the display aspect", () => {
        expect(backingSize(320, 240, 1920, 1080, 1)).toEqual({ w: 320, h: 240, pixelated: true });
    });

    test("aspect derivation rounds to the nearest pixel", () => {
        // 360 × (1366/768) = 640.3125 → 640
        expect(backingSize(0, 360, 1366, 768, 1)).toEqual({ w: 640, h: 360, pixelated: true });
    });

    test("a pin above the display is a smooth supersample, not pixelated", () => {
        // 2160 × (1920/1080) = 3840; backing ≥ display on both axes → no nearest upscale
        expect(backingSize(0, 2160, 1920, 1080, 1)).toEqual({ w: 3840, h: 2160, pixelated: false });
    });
});

// pruneViews drops a View whose camera despawned (membership) or whose eid was recycled to a new camera
// (create-stamp). The stamp arm is the same-update destroy+create realias membership can't see: the new
// camera keeps Camera membership, so without the stamp the recycled eid would inherit the dead camera's
// canvas + leaked observer. attachView is device-free (no canvas), so the sweep logic runs under bun test.
describe("pruneViews", () => {
    test("drops a View whose camera despawned (membership)", () => {
        clear();
        register("Camera", Camera);
        const state = new State();
        const eid = state.create();
        state.add(eid, Camera);
        attachView(eid);
        Views.get(eid)!.stamp = state.stamp(eid); // as BeginFrameSystem's pack records it

        state.destroy(eid); // Camera membership drops, but the View lingers
        pruneViews(state);

        expect(Views.has(eid)).toBe(false);
        Views.clear();
    });

    test("keeps a live camera whose stamp still matches", () => {
        clear();
        register("Camera", Camera);
        const state = new State();
        const eid = state.create();
        state.add(eid, Camera);
        attachView(eid);
        Views.get(eid)!.stamp = state.stamp(eid);

        pruneViews(state);

        expect(Views.has(eid)).toBe(true);
        Views.clear();
    });

    test("drops a View whose eid was recycled to a new camera (same-update realias)", () => {
        clear();
        register("Camera", Camera);
        const state = new State();
        const a = state.create();
        state.add(a, Camera);
        attachView(a);
        Views.get(a)!.stamp = state.stamp(a); // the packed stamp of camera A

        // same update: destroy A, recycle its eid with a new camera B — both carry Camera, so membership
        // alone never drops. Only the bumped create-stamp flags the realias.
        state.destroy(a);
        const b = state.create();
        expect(b).toBe(a); // LIFO recycle hands back the same slot
        state.add(b, Camera);

        pruneViews(state);

        // the stale View (A's canvas/observer) must be dropped so bindCamera re-binds a fresh one for B.
        // Without the stamp arm this stays true (B keeps Camera membership) — the red state.
        expect(Views.has(a)).toBe(false);
        Views.clear();
    });
});

// trackCanvasOwner is the dev-only rebuild guard's pure core (the import.meta.env.DEV gate lives at
// attachCanvas, which needs a real GPU device — so the warn logic is tested here directly, device-free).
// A canvas re-bound by a live undisposed *different* State warns; distinct canvases and a disposed prior
// owner stay silent — the rebuild-without-dispose leak class the State-owned teardown closes.
describe("trackCanvasOwner (dev rebuild guard)", () => {
    test("warns when a canvas is re-bound by a live, undisposed different State", () => {
        const canvas = {} as HTMLCanvasElement;
        const a = new State();
        const b = new State();
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            trackCanvasOwner(canvas, a); // first bind records the owner
            expect(warn).not.toHaveBeenCalled();
            trackCanvasOwner(canvas, b); // same canvas, a still live → warn
            expect(warn).toHaveBeenCalledTimes(1);
        } finally {
            warn.mockRestore();
        }
    });

    test("silent on distinct canvases (legit multi-app embed)", () => {
        const one = {} as HTMLCanvasElement;
        const two = {} as HTMLCanvasElement;
        const a = new State();
        const b = new State();
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            trackCanvasOwner(one, a);
            trackCanvasOwner(two, b);
            expect(warn).not.toHaveBeenCalled();
        } finally {
            warn.mockRestore();
        }
    });

    test("silent after the prior owner disposed (a clean rebuild)", () => {
        const canvas = {} as HTMLCanvasElement;
        const a = new State();
        const b = new State();
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            trackCanvasOwner(canvas, a);
            a.dispose(); // proper teardown before rebuild
            trackCanvasOwner(canvas, b);
            expect(warn).not.toHaveBeenCalled();
        } finally {
            warn.mockRestore();
        }
    });

    // the prod gate: reading `import.meta.env.DEV` outside a vite dev build must resolve false, never throw —
    // a regression to a bare `import.meta.env.DEV` would throw where `import.meta.env` is undefined. bun test
    // is not a vite dev build, so this pins the safe-read contract.
    test("devEnabled returns false (never throws) off a vite dev build", () => {
        expect(devEnabled()).toBe(false);
    });

    test("re-binding the same State is silent (idempotent auto-bind retry)", () => {
        const canvas = {} as HTMLCanvasElement;
        const a = new State();
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        try {
            trackCanvasOwner(canvas, a);
            trackCanvasOwner(canvas, a);
            expect(warn).not.toHaveBeenCalled();
        } finally {
            warn.mockRestore();
        }
    });
});
