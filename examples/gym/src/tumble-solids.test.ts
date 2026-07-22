// The stage-4b solid-layer derivation gate, proven red-first (spec tumble-inline — manual-pass finding:
// a solid vanishing from the rendered layer under mouse-grab interaction). The native solid layer derives
// one instanced Part per world solid shape each frame; the locked clause requires that derivation to be
// TOTAL — every solid body in the world is derived, always. This test pins that invariant across a
// grab/release cycle that drives a body far from the origin (an aggressive drag, the reported interaction):
//
//   every world solid body has exactly one live solid Part — before, during, and after the cycle.
//
// Red-first: the layer walked the world through `defaultDebugDraw()`'s ±100 m `drawingBounds`, so a body
// driven past ±100 by the grab was silently CLIPPED from the walk — its Part removed — while it still
// simulated in the world (the manual pass's "disappears under interaction"). A body's screen visibility is
// the Part pack's frustum cull's job, never the derivation's, so the walk must derive every body regardless
// of position. With the clip the live-Part count drops below the world's solid-body count (RED); with a
// total walk they stay equal (GREEN). Visuals never feed the gold oracle, so this is orthogonal to it.
//
// A minimal two-body world (a static ground + one dynamic box) is the reproduction — one live world keeps
// the wasm kernel singleton clean (the sequential-world trap, `tumble-golds.test.ts`); the invariant is
// scene-independent, so the smallest scene that can drive a body out of bounds pins it.
//
// Outside bunfig's `bun test` scope (rooted at `packages/shallot`) — run via `bun run test:gym`, or:
//   bun test ./examples/gym/src/tumble-solids.test.ts

import { expect, test } from "bun:test";
import { Color, Part, Slab, State, Transform } from "@dylanebert/shallot";
import { register } from "@dylanebert/shallot/ecs/core";
import {
    BodyType,
    type DebugDraw,
    defaultDebugDraw,
    init,
    makeBoxHull,
    World,
} from "@dylanebert/shallot/tumble/core";
import { beginGrab, driveGrab, type Grab, updateGrab } from "./tumble-grab";
import { solidPool } from "./tumble-sample";
import { collectSolids } from "./tumble-solids";

// the world's solid-shape count read UNBOUNDED — the source of truth the derived layer must match. Reading
// it with a huge draw box means a far body still counts, so a clipped derivation shows up as a shortfall.
function worldSolidCount(world: World): number {
    let n = 0;
    const H = 1e9;
    const dd: DebugDraw = {
        ...defaultDebugDraw(),
        drawingBounds: { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } },
        drawShapes: true,
        drawSolidHull: () => {
            n++;
        },
        drawSolidSphere: () => {
            n++;
        },
        drawSolidCapsule: () => {
            n++;
        },
        drawSolidMesh: () => {
            n++;
        },
    };
    world.draw(dd);
    return n;
}

test("the solid layer derives every world body, even one dragged far from the origin", async () => {
    await init({ threads: 0 });

    register("Transform", Transform);
    register("Color", Color);
    register("Part", Part);
    const state = new State({ capacity: 1024 });
    Slab.collect(); // allocate the slab CPU columns at the now-fixed capacity (SlabPlugin's job in a real build)

    const world = new World({ gravity: { x: 0, y: -10, z: 0 } });
    const ground = world.createBody({ type: BodyType.Static });
    ground.createHull({}, makeBoxHull(50, 1, 50));
    const box = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 } });
    box.createHull({ density: 1 }, makeBoxHull(0.5, 0.5, 0.5));

    // synthetic mesh ids (one per unique geometry) — the pool only keys on the map, and no GPU device is
    // needed to exercise its membership logic (registerSolids is the device-bound half, out of scope here).
    const solids = collectSolids(world);
    const keyToMesh = new Map<object, number>();
    let id = 0;
    for (const key of solids.keyToName.keys()) keyToMesh.set(key, id++);

    const pool = solidPool(state, keyToMesh);
    const drawFrame = (): void => {
        pool.begin();
        world.draw(pool.adapter);
        pool.end();
    };
    const liveParts = (): number => {
        let n = 0;
        for (const _ of state.query([Part])) n++;
        return n;
    };

    // settle: every world solid body has exactly one live Part.
    for (let i = 0; i < 30; i++) world.step(1 / 60, 4);
    drawFrame();
    expect(liveParts()).toBe(worldSolidCount(world)); // 2: ground + box

    // grab the box and drive the anchor far — an aggressive drag flings the box past the origin. The pool
    // walk runs every step so the invariant is checked THROUGH the motion, not just at the end.
    const r = world.castRayClosest({ x: 0, y: 40, z: 0 }, { x: 0, y: -1000, z: 0 });
    expect(r.hit).toBe(true);
    const grab: Grab = beginGrab(world, { x: 0, y: 40, z: 0 }, { x: 0, y: -1000, z: 0 }) as Grab;
    expect(grab).not.toBeNull();

    const Far = 5000;
    for (let f = 0; f < 400 && box.getPosition().x < 300; f++) {
        updateGrab(grab, { x: Far, y: 5, z: 0 }, { x: 1, y: 0, z: 0 });
        driveGrab(grab, 1 / 60);
        world.step(1 / 60, 4);
        drawFrame();
        // the derivation must stay total every step — no world solid body ever loses its Part.
        expect(liveParts()).toBe(worldSolidCount(world));
    }

    // the drag actually pushed the box well past ±100, so the ±100 clip WOULD have dropped it (the red case).
    expect(box.getPosition().x).toBeGreaterThan(100);
    // final: still one Part per world solid body.
    expect(liveParts()).toBe(worldSolidCount(world));

    state.dispose();
    world.destroy();
});
