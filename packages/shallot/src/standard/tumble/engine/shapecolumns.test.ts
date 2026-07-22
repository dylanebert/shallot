// T3 — the resident shape column (kernel/src/shapes.rs) mirrors every live shape's type, geometry and
// `nextShapeId`, and the body record's headShapeId lane mirrors the head of each awake body's shape
// list. Nothing reads the column yet (T4 does), so nothing else in the suite can see it go stale: these
// tests are the whole gate on the write sites. Layout, the create/destroy patches, and both relocation
// directions (a lower region's grow moving this one, and its own grow moving the manifold region above
// it) are exercised against the real wasm — the region modules are wasm-only, so this is the tier a
// cargo test would otherwise hold.

import { describe, expect, test } from "bun:test";
import { NULL_INDEX } from "./array";
import { IDENT_RECORDS, N_BODY } from "./bodycolumns";
import { S2_HEAD_SHAPE, SIM2_STRIDE } from "./columns";
import { SetType } from "./core";
import type { Capsule, Sphere } from "./geometry";
import type { HullData } from "./hull";
import { makeBoxHull } from "./hull";
import { BodyType, createGridMesh, World } from "./index";
import { kernel } from "./kernel";
import { isConvexRefit, S_GEOM, S_NEXT, S_TYPE, SHAPE_STRIDE } from "./shapecolumns";
import { ShapeType } from "./types";
import type { WorldState } from "./world";

/** exactly-representable helper so the test positions aren't f64 literals feeding the engine. */
function f(x: number): number {
    return Math.fround(x);
}

/** The column's null sentinel — TS's `NULL_INDEX` (-1) through the u32 view. */
const NULL_SHAPE = 0xffffffff;

/**
 * Assert the column record of every live shape matches the shape, and every awake body's head lane
 * matches its shape list. The invariant this stage owns.
 */
function checkColumn(state: WorldState): void {
    checkRegionsStacked();
    state.shapeStore.refreshViews();
    const u = state.shapeStore.shapeU;
    const g = state.shapeStore.shapeF;

    for (const shape of state.shapes) {
        if (shape.id === NULL_INDEX) continue; // freed slot; its record is dead
        const o = shape.id * SHAPE_STRIDE;
        expect(u[o + S_TYPE]).toBe(shape.type);
        expect(u[o + S_NEXT]).toBe(shape.nextShapeId >>> 0);

        const p = o + S_GEOM;
        if (shape.type === ShapeType.Sphere) {
            const s = shape.sphere as Sphere;
            expect([g[p], g[p + 1], g[p + 2], g[p + 3]]).toEqual([
                s.center.x,
                s.center.y,
                s.center.z,
                s.radius,
            ]);
        } else if (shape.type === ShapeType.Capsule) {
            const c = shape.capsule as Capsule;
            expect([g[p], g[p + 1], g[p + 2], g[p + 3], g[p + 4], g[p + 5], g[p + 6]]).toEqual([
                c.center1.x,
                c.center1.y,
                c.center1.z,
                c.center2.x,
                c.center2.y,
                c.center2.z,
                c.radius,
            ]);
        } else if (shape.type === ShapeType.Hull) {
            const box = (shape.hull as HullData).aabb;
            expect([g[p], g[p + 1], g[p + 2], g[p + 3], g[p + 4], g[p + 5]]).toEqual([
                box.lowerBound.x,
                box.lowerBound.y,
                box.lowerBound.z,
                box.upperBound.x,
                box.upperBound.y,
                box.upperBound.z,
            ]);
        }
    }

    state.bodyStore.refreshViews();
    const sim2 = state.bodyStore.sim2U;
    for (const body of state.bodies) {
        if (body.id === NULL_INDEX || body.setIndex !== SetType.Awake) continue;
        expect(sim2[body.localIndex * SIM2_STRIDE + S2_HEAD_SHAPE]).toBe(body.headShapeId >>> 0);
    }
}

/**
 * Assert the persistent regions are still stacked body < fat-AABB < shape < manifold in linear memory.
 * The mirror check above cannot see a dropped relocation — writer and reader both go through the same
 * layout header, so a stale base is self-consistent — but it silently aliases the region below. This is
 * the check that goes red when `reserveBodies` forgets `shapes::relocate`, or when the shape region's
 * own grow forgets to move the manifold region above it.
 */
function checkRegionsStacked(): void {
    const k = kernel();
    const buf = k.memory.buffer;

    const bodyLayout = new Uint32Array(buf, k.bodyLayoutPtr(), N_BODY);
    // sim2 is the body region's top column; it holds `cap` bodies plus the trailing identity records.
    const bodyEnd = bodyLayout[N_BODY - 1] + (k.bodyCap() + IDENT_RECORDS) * SIM2_STRIDE * 4;

    // The fat-AABB region is sized at shape create, so it may not exist yet when no enabled-body shape has
    // been made; the shape region then anchors straight onto the body region (its first grow relocates it).
    const fatCap = k.fatAabbCap();
    const fatBase = new Uint32Array(buf, k.fatAabbLayoutPtr(), 1)[0];
    const fatEnd = fatCap === 0 ? bodyEnd : fatBase + fatCap * 6 * 4;
    if (fatCap !== 0) expect(fatBase).toBeGreaterThanOrEqual(bodyEnd);

    const shapeCap = k.shapeCap();
    const shapeBase = new Uint32Array(buf, k.shapeLayoutPtr(), 1)[0];
    expect(shapeCap).toBeGreaterThan(0);
    expect(shapeBase).toBeGreaterThanOrEqual(fatEnd);

    const shapeEnd = shapeBase + shapeCap * SHAPE_STRIDE * 4;
    const manifoldDir = new Uint32Array(buf, k.manifoldLayoutPtr(), 2)[0];
    if (manifoldDir !== 0) expect(manifoldDir).toBeGreaterThanOrEqual(shapeEnd);
}

/** Every region base above the body region, in memory order — the geometry pools included. */
function regionBases(): number[] {
    const k = kernel();
    const buf = k.memory.buffer;
    const manifold = new Uint32Array(buf, k.manifoldLayoutPtr(), 2);
    const geo = new Uint32Array(buf, k.geoLayoutPtr(), 6);
    return [manifold[0], manifold[1], ...geo];
}

/** The body's shape list as the *column* sees it: head lane → `next` chain. */
function chainFromColumn(state: WorldState, bodyId: number): number[] {
    state.shapeStore.refreshViews();
    const u = state.shapeStore.shapeU;
    const ids: number[] = [];
    let id = state.bodies[bodyId].headShapeId;
    while (id !== NULL_INDEX) {
        ids.push(id);
        const next = u[id * SHAPE_STRIDE + S_NEXT];
        id = next === NULL_SHAPE ? NULL_INDEX : next;
    }
    return ids;
}

describe("shape column residency", () => {
    test("mirrors type, geometry and the shape list across create / destroy", () => {
        const world = new World();
        const ground = world.createBody({ type: BodyType.Static, position: { x: 0, y: 0, z: 0 } });
        ground.createHull({}, makeBoxHull(20, f(0.5), 20)); // shapeId 0
        ground.createMesh({}, createGridMesh(4, 4, 1, 1, false), { x: 1, y: 1, z: 1 }); // 1 — fallback type

        // A multi-shape dynamic body. The list is head-inserted, so the chain is reverse-creation order.
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: f(0.5) }); // 2
        const capsule = body.createCapsule(
            { density: 1 },
            {
                center1: { x: f(-0.5), y: 0, z: 0 },
                center2: { x: f(0.5), y: 0, z: 0 },
                radius: f(0.25),
            },
        ); // 3
        const hull = body.createHull({ density: 1 }, makeBoxHull(f(0.5), f(0.5), f(0.5))); // 4

        const state = world.state;
        checkColumn(state);
        expect(chainFromColumn(state, 1)).toEqual([4, 3, 2]);
        expect(chainFromColumn(state, 0)).toEqual([1, 0]);

        // Destroying a middle shape re-points its predecessor's `next` slot in the column.
        capsule.destroy();
        checkColumn(state);
        expect(chainFromColumn(state, 1)).toEqual([4, 2]);

        // Destroying the head re-points the awake body's head lane.
        world.step(1 / 60, 4);
        hull.destroy();
        checkColumn(state);
        expect(chainFromColumn(state, 1)).toEqual([2]);

        world.destroy();
    });

    // A sleeping body has no resident record (bodycolumns.ts syncHeadShape: "a body outside the awake
    // set has no resident record — its lane is written when it enters one, from the body's
    // then-current headShapeId"). Destroying one of a sleeping body's shapes still patches the shape
    // column's own next-pointer (unlinkShape, unconditional) and the JS-side body.headShapeId
    // (destroyShapeInternal, unconditional), but the resident head-shape lane sync is skipped while
    // asleep. This pins that the stale lane self-heals from body.headShapeId when the body wakes and
    // resident-pushes again — the refit/broadphase walk the correct remaining shape chain, not a
    // dangling one.
    test("destroying a shape on a sleeping body heals the resident chain on wake", () => {
        const world = new World();
        const ground = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        ground.createHull({}, makeBoxHull(10, f(0.5), 10));

        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: f(0.5), z: 0 },
        });
        const left = body.createSphere(
            { density: 1 },
            { center: { x: f(-0.6), y: 0, z: 0 }, radius: f(0.5) },
        );
        const right = body.createSphere(
            { density: 1, enableContactEvents: true },
            { center: { x: f(0.6), y: 0, z: 0 }, radius: f(0.5) },
        );

        const state = world.state;
        const bodyId = body.id.index1 - 1;
        checkColumn(state);
        expect(chainFromColumn(state, bodyId)).toEqual([right.id.index1 - 1, left.id.index1 - 1]);

        // Settle both spheres onto the ground and let the body sleep.
        let asleep = false;
        for (let i = 0; i < 300 && asleep === false; ++i) {
            world.step(1 / 60);
            for (const move of world.getBodyEvents().moveEvents) {
                if (move.body.id.index1 === body.id.index1 && move.fellAsleep) asleep = true;
            }
        }
        expect(asleep).toBe(true);
        expect(body.isAwake()).toBe(false);

        // Destroy the right sphere while asleep. Its contact with the ground is touching, so the
        // contact teardown inside destroyShapeInternal wakes the body as a side effect (contact.ts
        // destroyContact: `wakeBodies && touching`) — but that happens *after* unlinkShape/syncHeadShape
        // already ran with the body still asleep, so the resident-lane skip is exercised regardless.
        right.destroy();
        // Force-awake explicitly too, matching the scenario regardless of the auto-wake above.
        body.setAwake(true);
        expect(body.isAwake()).toBe(true);

        world.step(1 / 60, 4);

        checkColumn(state);
        expect(chainFromColumn(state, bodyId)).toEqual([left.id.index1 - 1]);
        expect(right.isValid()).toBe(false);
        expect(left.isValid()).toBe(true);

        // The remaining shape still collides with the ground; nothing references the destroyed one.
        const leftIndex = left.id.index1 - 1;
        let contactKey = state.bodies[bodyId].headContactKey;
        let contactCount = 0;
        while (contactKey !== NULL_INDEX) {
            const contactId = contactKey >> 1;
            const edgeIndex = contactKey & 1;
            const contact = state.contacts[contactId];
            expect(contact.shapeIdA === leftIndex || contact.shapeIdB === leftIndex).toBe(true);
            contactCount += 1;
            contactKey = contact.edges[edgeIndex].nextKey;
        }
        expect(contactCount).toBeGreaterThan(0);

        // Finite, resting transform — no NaN corruption from the stale-lane path.
        const xf = body.getTransform();
        expect(Number.isFinite(xf.p.x)).toBe(true);
        expect(Number.isFinite(xf.p.y)).toBe(true);
        expect(Number.isFinite(xf.p.z)).toBe(true);

        world.destroy();
    });

    test("survives its own grow and a body-region grow below it", () => {
        const world = new World();
        const box = makeBoxHull(f(0.5), f(0.5), f(0.5));
        const ground = world.createBody({ type: BodyType.Static, position: { x: 0, y: 0, z: 0 } });
        ground.createHull({}, makeBoxHull(30, f(0.5), 30));

        const state = world.state;

        // Step a small scene first so the manifold + geometry regions above the shape region are live:
        // a shape-region grow has to memmove them, not just re-anchor itself.
        for (let i = 0; i < 4; ++i) {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: f(i * 1.5), y: 2, z: 0 },
            });
            b.createHull({ density: 1 }, box);
        }
        for (let step = 0; step < 10; ++step) world.step(1 / 60, 4);
        checkColumn(state);

        // Past the 16-record initial capacities: the shape region grows (16 → 32 → 64), relocating the
        // manifold + geometry regions above it, and the body region grows too — which relocates the
        // fat-AABB + shape + manifold + geometry regions above *it*. Both directions of the chain, with
        // live records on either side of every move.
        for (let i = 0; i < 40; ++i) {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: f((i % 8) * 1.5), y: f(4 + i * 1.2), z: f(Math.trunc(i / 8) * 1.5) },
            });
            b.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: f(0.5) });
        }
        expect(state.shapes.length).toBeGreaterThan(32);
        checkColumn(state);

        // And they hold as the scene settles: refits, island sleep, and the resident-record swap-remove
        // migrations the head lane has to ride.
        for (let step = 0; step < 60; ++step) {
            world.step(1 / 60, 4);
            checkColumn(state);
        }

        // A body-region grow with *no* shape create after it. A scene that adds a shape per body closes
        // this window by accident — the next `reserveShapes` grow re-anchors the region from the (moved)
        // fat-AABB top — so shapeless bodies are what actually pins `reserveBodies`'s relocation of the
        // shape region. Shape count is unchanged here; only the body high-water moves.
        const shapeCount = state.shapes.length;
        for (let i = 0; i < 100; ++i) {
            world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: f(200 + i), z: 0 } });
        }
        expect(state.shapes.length).toBe(shapeCount);
        checkColumn(state);
        world.step(1 / 60, 4);
        checkColumn(state);

        world.destroy();
    });

    test("its own grow shifts every region above it by one delta", () => {
        const world = new World();
        const box = makeBoxHull(f(0.5), f(0.5), f(0.5));
        const ground = world.createBody({ type: BodyType.Static, position: { x: 0, y: 0, z: 0 } });
        ground.createHull({}, makeBoxHull(30, f(0.5), 30));
        for (let i = 0; i < 8; ++i) {
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: f(i * 1.5), y: 2, z: 0 },
            });
            b.createHull({ density: 1 }, box);
        }
        for (let step = 0; step < 10; ++step) world.step(1 / 60, 4);

        // Stacking alone is a weak gate on the regions *above* this one: a stale base still tests as
        // "above the region below it" whenever the growth delta is smaller than the region it forgot to
        // move. What a relocation actually promises is that one memmove shifts every region above by the
        // same delta, and that this region's own base does not move (it is base-anchored on the fat-AABB
        // top). Assert exactly that across a grow of the shape region itself.
        const k = kernel();
        const shapeBase = () => new Uint32Array(k.memory.buffer, k.shapeLayoutPtr(), 1)[0];
        const capBefore = k.shapeCap();
        const baseBefore = shapeBase();
        const above = regionBases();

        // Grow the shape region in isolation: a disabled body has no proxy, so its shapes skip the fat-AABB
        // write (fataabbcolumns.ts) — otherwise that region, which sits below this one, would grow at the
        // same shape count and relocate the shape base out from under the base-anchored invariant.
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 40, z: 0 },
            isEnabled: false,
        });
        while (k.shapeCap() === capBefore) {
            body.createSphere({ density: 1 }, { center: { x: 0, y: 0, z: 0 }, radius: f(0.1) });
        }

        expect(shapeBase()).toBe(baseBefore);
        const moved = regionBases();
        const delta = moved[0] - above[0];
        expect(delta).toBeGreaterThan(0);
        expect(moved).toEqual(above.map((b) => b + delta));
        checkColumn(world.state);

        world.destroy();
    });
});

describe("finalize refit partition", () => {
    test("isConvexRefit selects sphere/capsule/hull, skips mesh/height-field/compound", () => {
        // The kernel refits sphere/capsule/hull in place; the rest fall back to the TS AABB path at their
        // list position (mirrors kernel `is_convex_refit`, kernel/src/finalize.rs).
        expect(isConvexRefit(ShapeType.Sphere)).toBe(true);
        expect(isConvexRefit(ShapeType.Capsule)).toBe(true);
        expect(isConvexRefit(ShapeType.Hull)).toBe(true);
        expect(isConvexRefit(ShapeType.Mesh)).toBe(false);
        expect(isConvexRefit(ShapeType.HeightField)).toBe(false);
        expect(isConvexRefit(ShapeType.Compound)).toBe(false);
    });
});
