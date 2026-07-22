// Regression: the resident broad/manifold columns are a kernel singleton reused across sequential worlds
// (the gym host runs a throwaway oracle world, destroys it, then builds the render world; the fixture/gold
// suites run per-world). Each resident region is grow-only so a later world reuses the earlier high-water;
// `reserveManifolds` was the one region that missed that guard, so a fresh world's small caps SHRANK
// `MANIFOLD_END` below the broad region anchored at it. The next contact-driven manifold grow then memmoved
// the broad region from a stale anchor and orphaned the static tree pool's bytes — the static bodies
// silently dropped out of every broadphase query (`overlapAABB` / `world.draw`) while still simulating.
//
// The kernel is a process-wide singleton whose region high-water leaks across sibling test files, and this
// scenario's whole point is a stale high-water — so it runs the two-world sequence in an isolated child
// process (the fixture/gold suites' own recipe for the sequential-world traps, `tumble.md`).

import { describe, expect, test } from "bun:test";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

// Runs in the isolated child (`bun -e`): a throwaway world grows every resident region to its high-water,
// then the reused world flings its planks so the broadphase fattens each AABB into overlap
// with everything, spiking the manifold count past the reused high-water and forcing the mid-step
// relocation the bug corrupts. Prints the static-shape count each step; the parent asserts it stays 3.
const child = `
import { BodyType, makeBoxHull, World } from ${JSON.stringify(resolve(here, "index.ts"))};
const IDENT = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const H = 1e9, DT = 1 / 60;
const worldBox = { lowerBound: { x: -H, y: -H, z: -H }, upperBound: { x: H, y: H, z: H } };
const pin = (b, w) => ({ p: b.getLocalPoint(w), q: IDENT });
function buildBridge(world, count) {
    world.createBody({ type: BodyType.Static }).createHull({}, makeBoxHull(50, 1, 50));
    const px = 0.125, deckY = 6, startX = -(count * 2 * px) / 2;
    const post = (x) => { const b = world.createBody({ type: BodyType.Static, position: { x, y: deckY, z: 0 } }); b.createHull({}, makeBoxHull(0.15, 0.4, 0.7)); return b; };
    const hinge = (a, b, ex) => { for (const z of [-0.5, 0.5]) world.createSphericalJoint(a, b, { localFrameA: pin(a, { x: ex, y: deckY, z }), localFrameB: pin(b, { x: ex, y: deckY, z }), constraintHertz: 1000, enableSpring: true, hertz: 2, dampingRatio: 1 }); };
    let prev = post(startX - 2 * px); const right = post(startX + count * 2 * px); let lastX = startX;
    for (let i = 0; i < count; ++i) { const cx = startX + i * 2 * px + px; const p = world.createBody({ type: BodyType.Dynamic, position: { x: cx, y: deckY, z: 0 } }); p.createHull({ density: 20 }, makeBoxHull(px, px, 0.5)); hinge(prev, p, cx - px); prev = p; lastX = cx; }
    hinge(prev, right, lastX + px);
    // four loose boxes dropped on the deck (the sample's extra dynamic-dynamic contacts — needed to spike
    // the reused world's manifold count past the stale high-water and force the corrupting relocation).
    for (let i = 0; i < 4; ++i) world.createBody({ type: BodyType.Dynamic, position: { x: startX + ((i + 1) * count * 2 * px) / 5, y: deckY + 2 + i, z: 0 } }).createHull({ density: 1 }, makeBoxHull(0.4, 0.4, 0.4));
}
const countStatics = (world) => { let n = 0; world.overlapAABB(worldBox, (s) => { if (s.getBody().getType() === BodyType.Static) n++; return true; }); return n; };

const warm = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: true, enableContinuous: true });
buildBridge(warm, 24);
for (let i = 0; i < 60; ++i) warm.step(DT, 4);
warm.destroy();

const world = new World({ gravity: { x: 0, y: -10, z: 0 }, enableSleep: true, enableContinuous: true });
buildBridge(world, 24);
for (let i = 0; i < 60; ++i) world.step(DT, 4);
let min = countStatics(world);
const planks = [];
world.overlapAABB(worldBox, (s) => { const b = s.getBody(); if (b.getType() === BodyType.Dynamic) planks.push(b); return true; });
for (const p of planks) p.setLinearVelocity({ x: 1e7, y: 1e7, z: 1e7 });
for (let i = 0; i < 60; ++i) { world.step(DT, 4); min = Math.min(min, countStatics(world)); }
world.destroy();
process.stdout.write("STATICS_MIN=" + min);
`;

describe("sequential-world resident-region reuse", () => {
    test("static bodies survive contact-driven relocation in a reused kernel region", () => {
        const r = spawnSync("bun", ["-e", child], { encoding: "utf8", timeout: 60_000 });
        expect(r.status).toBe(0);
        // 3 = ground + 2 posts; the fling must never drop the static tree out of the broadphase.
        expect(r.stdout).toContain("STATICS_MIN=3");
    });
});
