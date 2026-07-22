// The overlay channel's own output, headless (spec tumble-inline stage 4 — the demonstration-layer restore).
// The engine fold's first pass dropped every sample's `render()` overlay, so event/collision scenes read as
// bare "boxes falling". This pins the mechanism the standing visual probe's `overlay-labels` check reads: a
// sample's `render()` emits the labels + HUD lines its demonstration needs. events-joint-break is the gated
// case — six boxes on rising-threshold breakable joints, each labelled with its threshold + a live load HUD.
//
// GPU-free + DOM-free: the render bodies only read the world and call the Overlay surface, so a counting stub
// (the same string3d/text output `overlayLayer.labelCount()` counts) exercises them without a device or a
// canvas — exactly what the headless `bun run test:gym` tier can run.
//
// Red-first (spec gate): the check is nonzero labels from the render. With the render disconnected (a no-op,
// the "adapter disconnected" shape) the count is 0 and every assertion below fails; wired, it is 6. Proven
// red by stubbing renderJointBreak to a no-op before implementing it.
//
// Outside bunfig's `bun test` scope (rooted at packages/shallot) — run via `bun run test:gym`, or explicitly:
//   bun test ./examples/gym/src/tumble-overlay.test.ts

import { expect, test } from "bun:test";
import { init, World } from "@dylanebert/shallot/tumble/core";
import { buildJointBreak, renderJointBreak } from "./tumble-joint-break";
import type { Overlay } from "./tumble-overlay";

// a counting Overlay stub: records string3d labels + text lines (the render's own output), no-ops the
// geometry (its emission is validated on the real device by the interaction probe).
function counter(): { draw: Overlay; labels: { s: string }[]; texts: string[] } {
    const labels: { s: string }[] = [];
    const texts: string[] = [];
    const draw: Overlay = {
        line: () => {},
        point: () => {},
        aabb: () => {},
        solidSphere: () => {},
        solidCapsule: () => {},
        string3d: (_p, s) => {
            labels.push({ s });
        },
        text: (line) => {
            texts.push(line);
        },
    };
    return { draw, labels, texts };
}

test("events-joint-break render emits a label per hung box + a load HUD line", async () => {
    await init({ threads: 0 });
    const world = new World({
        gravity: { x: 0, y: -10, z: 0 },
        enableSleep: true,
        enableContinuous: true,
    });
    try {
        buildJointBreak(world);
        const { draw, labels, texts } = counter();
        renderJointBreak(draw, world, {}, 0);

        // one label per hung box — the presence invariant the visual probe's `overlay-labels` asserts nonzero;
        // a disconnected render (no-op) emits 0 and trips this.
        expect(labels.length).toBe(6);
        // labelled by threshold before any break (300, 600, …, 1800 — the sample's rising thresholds).
        expect(labels.map((l) => l.s)).toEqual(["300", "600", "900", "1200", "1500", "1800"]);
        // the live load / broken HUD line (the sample's drawText).
        expect(texts).toEqual(["load 0 N   broken 0/6"]);
    } finally {
        world.destroy();
    }
});
