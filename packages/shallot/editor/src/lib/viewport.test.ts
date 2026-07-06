import { beforeEach, describe, expect, test } from "bun:test";
import { Camera, Depth, State, Tag } from "@dylanebert/shallot";
import { clear, register } from "@dylanebert/shallot/ecs/core";
import { GizmosPlugin, Overlay, Overlays, overlayCameras, setOverlays } from "./viewport";

describe("overlayCameras — the per-view category gate", () => {
    beforeEach(() => {
        clear();
        register("Camera", Camera);
        register("Overlays", Overlays);
    });

    test("selects exactly the cameras whose set enables the category", () => {
        const s = new State();
        const grid = s.create();
        s.add(grid, Camera as never);
        s.add(grid, Overlays as never);
        Overlays.enabled.set(grid, Overlay.Grid);

        const both = s.create();
        s.add(both, Camera as never);
        s.add(both, Overlays as never);
        Overlays.enabled.set(both, Overlay.Grid | Overlay.Outline);

        const off = s.create();
        s.add(off, Camera as never);
        s.add(off, Overlays as never);
        Overlays.enabled.set(off, 0);

        const byEid = (a: number, b: number) => a - b;
        expect(overlayCameras(s, Overlay.Grid).sort(byEid)).toEqual([grid, both].sort(byEid));
        expect(overlayCameras(s, Overlay.Outline)).toEqual([both]);
        expect(overlayCameras(s, Overlay.Grid)).not.toContain(off);
    });

    test("a camera without the Overlays component enables nothing", () => {
        const s = new State();
        const cam = s.create();
        s.add(cam, Camera as never);
        expect(overlayCameras(s, Overlay.Grid)).toEqual([]);
    });
});

describe("setOverlays — pulls in each category's prepass lane", () => {
    beforeEach(() => {
        clear();
        register("Camera", Camera);
        register("Overlays", Overlays);
        register("Tag", Tag);
        register("Depth", Depth);
    });

    function camera(s: State): number {
        const eid = s.create();
        s.add(eid, Camera as never);
        return eid;
    }

    test("enabling a category adds its lane marker — grid→Depth, outline→Tag", () => {
        const s = new State();
        const cam = camera(s);
        setOverlays(s, cam, Overlay.Grid | Overlay.Outline);
        expect(Overlays.enabled.get(cam)).toBe(Overlay.Grid | Overlay.Outline);
        expect(s.has(cam, Depth as never)).toBe(true);
        expect(s.has(cam, Tag as never)).toBe(true);
    });

    test("a category left off pulls no lane; the one on pulls only its own", () => {
        const s = new State();
        const cam = camera(s);
        setOverlays(s, cam, Overlay.Grid);
        expect(s.has(cam, Depth as never)).toBe(true);
        expect(s.has(cam, Tag as never)).toBe(false);
    });

    test("mask 0 enables the component but pulls no lane (the play-mode default)", () => {
        const s = new State();
        const cam = camera(s);
        setOverlays(s, cam, 0);
        expect(s.has(cam, Overlays as never)).toBe(true);
        expect(Overlays.enabled.get(cam)).toBe(0);
        expect(s.has(cam, Depth as never)).toBe(false);
        expect(s.has(cam, Tag as never)).toBe(false);
    });

    test("toggling a category off keeps its lane — additive, no target churn", () => {
        const s = new State();
        const cam = camera(s);
        setOverlays(s, cam, Overlay.Grid);
        setOverlays(s, cam, 0);
        expect(Overlays.enabled.get(cam)).toBe(0);
        expect(s.has(cam, Depth as never)).toBe(true);
    });
});

describe("overlay systems declare the layer + category axes", () => {
    test("every Gizmos system is tooling-layer; overlays carry a numeric category", () => {
        expect(GizmosPlugin.systems?.length).toBeGreaterThan(0);
        // every gizmos system is editor tooling — stripped from a shipped game (the layer axis)
        for (const sys of GizmosPlugin.systems ?? []) {
            expect(sys.annotations?.layer).toBe("tooling");
        }
        // the overlays (grid / outline) are the category-bearing subset: each declares a numeric category
        // and runs in both edit and play. Handles (the transform manipulator) are tooling too but not an
        // overlay — edit-only, driven by the active tool + selection, so they carry no category.
        const overlays = (GizmosPlugin.systems ?? []).filter(
            (s) => s.annotations?.category !== undefined,
        );
        expect(overlays.length).toBeGreaterThan(0);
        for (const sys of overlays) {
            expect(typeof sys.annotations?.category).toBe("number");
            expect(sys.annotations?.mode).toBe("always");
        }
    });
});
