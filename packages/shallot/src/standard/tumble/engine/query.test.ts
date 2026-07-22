// Stage 13 (queries) tests: per-body ray cast, shape cast, and overlap. Ported from
// test_body_query.c (CastRay*/CastShape*/Overlap* subtests). The per-body query functions take an
// explicit world origin and an explicit body transform (not the body's stored pose): everything is
// re-centered on the origin so the float collision math stays exact far from the world origin. These
// pin that framing — results come back in world space, the supplied transform drives the geometry,
// and a large origin offset must not change a hit fraction or normal. Tolerances match the C test's
// ENSURE_SMALL bounds (the underlying rayCast/shapeCast/overlap primitives are bit-exact per gold).
//
// The Mover* subtests (from test_body_query.c) live at the bottom, plus a world-level collideMover /
// castMover pair the upstream unit suite lacks (they exercise the World wiring the gold can't reach).

import { describe, expect, test } from "bun:test";
import {
    BodyType,
    type Capsule,
    type CollisionPlane,
    clipVector,
    defaultQueryFilter,
    makeBoxHull,
    type ShapeProxy,
    solvePlanes,
    type Transform,
    type Vec3,
    World,
} from "./index";
import { quat } from "./math";

const PI = Math.PI;

function identityAt(x: number, y: number, z: number): Transform {
    return { p: { x, y, z }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
}

describe("body cast ray", () => {
    test("hits sphere", () => {
        const world = new World();
        const body = world.createBody();
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });

        // Body sphere at world (5,0,0), ray straight at it along +X.
        const r = body.castRay({ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, identityAt(5, 0, 0));

        expect(r.hit).toBe(true);
        expect(r.shape?.isValid()).toBe(true);
        expect(Math.abs(r.fraction - 0.4)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.x + 1)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.y)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.z)).toBeLessThan(1e-5);
        expect(Math.abs(r.point.x - 4)).toBeLessThan(1e-4);
        expect(Math.abs(r.point.y)).toBeLessThan(1e-4);
        expect(Math.abs(r.point.z)).toBeLessThan(1e-4);

        world.destroy();
    });

    test("miss", () => {
        const world = new World();
        const body = world.createBody();
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });

        // Ray runs parallel to the body, never reaching it.
        const r = body.castRay({ x: 0, y: 0, z: 0 }, { x: 0, y: 10, z: 0 }, identityAt(5, 0, 0));
        expect(r.hit).toBe(false);

        world.destroy();
    });

    test("closest shape", () => {
        const world = new World();
        const body = world.createBody();
        const near = body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        body.createSphere({}, { center: { x: 4, y: 0, z: 0 }, radius: 1 });

        // Ray crosses both spheres; the loop must shrink maxFraction to the nearer hit.
        const r = body.castRay({ x: -5, y: 0, z: 0 }, { x: 10, y: 0, z: 0 }, identityAt(0, 0, 0));

        expect(r.hit).toBe(true);
        expect(r.shape?.id.index1).toBe(near.id.index1);
        expect(r.shape?.id.generation).toBe(near.id.generation);
        expect(Math.abs(r.fraction - 0.4)).toBeLessThan(1e-5);

        world.destroy();
    });

    test("rotated body", () => {
        const world = new World();
        const body = world.createBody();
        // Local center (0,2,0) rotated +90 deg about Z lands at world (-2,0,0).
        body.createSphere({}, { center: { x: 0, y: 2, z: 0 }, radius: 0.5 });

        const bodyTransform: Transform = {
            p: { x: 0, y: 0, z: 0 },
            q: quat.fromAxisAngle({ x: 0, y: 0, z: 1 }, 0.5 * PI),
        };
        const r = body.castRay({ x: 0, y: 0, z: 0 }, { x: -4, y: 0, z: 0 }, bodyTransform);

        expect(r.hit).toBe(true);
        expect(Math.abs(r.fraction - 0.375)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.x - 1)).toBeLessThan(1e-5);
        expect(Math.abs(r.point.x + 1.5)).toBeLessThan(1e-4);

        world.destroy();
    });

    test("far from origin", () => {
        const world = new World();
        const body = world.createBody();
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });

        // Same geometry as "hits sphere", shifted far from the world origin. Relative framing keeps
        // the subtraction exact, so fraction and normal must be unchanged.
        const origin = { x: 1.0e6, y: -2.0e6, z: 5.0e5 };
        const bodyTransform: Transform = {
            p: { x: origin.x + 5, y: origin.y, z: origin.z },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        };
        const r = body.castRay(origin, { x: 10, y: 0, z: 0 }, bodyTransform);

        expect(r.hit).toBe(true);
        expect(Math.abs(r.fraction - 0.4)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.x + 1)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.y)).toBeLessThan(1e-5);
        expect(Math.abs(r.normal.z)).toBeLessThan(1e-5);

        world.destroy();
    });
});

describe("body cast shape", () => {
    const point: ShapeProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.5 };

    test("hits box", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(1, 1, 1));

        // Sphere proxy radius 0.5 cast along +X into a box whose front face is at world x = 4.
        const r = body.castShape(
            { x: 0, y: 0, z: 0 },
            point,
            { x: 10, y: 0, z: 0 },
            identityAt(5, 0, 0),
        );

        expect(r.hit).toBe(true);
        expect(r.shape?.isValid()).toBe(true);
        // The fraction carries a small shape-cast skin; the point and normal do not.
        expect(Math.abs(r.fraction - 0.35)).toBeLessThan(1e-2);
        expect(Math.abs(r.normal.x + 1)).toBeLessThan(1e-4);
        expect(Math.abs(r.point.x - 4)).toBeLessThan(1e-3);

        world.destroy();
    });

    test("miss", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(1, 1, 1));

        const r = body.castShape(
            { x: 0, y: 0, z: 0 },
            point,
            { x: 0, y: 10, z: 0 },
            identityAt(5, 0, 0),
        );
        expect(r.hit).toBe(false);

        world.destroy();
    });

    test("rotated body", () => {
        const world = new World();
        const body = world.createBody();
        // Body sphere local center (0,2,0) rotated +90 deg about Z lands at world (-2,0,0).
        body.createSphere({}, { center: { x: 0, y: 2, z: 0 }, radius: 1 });

        const bodyTransform: Transform = {
            p: { x: 0, y: 0, z: 0 },
            q: quat.fromAxisAngle({ x: 0, y: 0, z: 1 }, 0.5 * PI),
        };
        const r = body.castShape({ x: 0, y: 0, z: 0 }, point, { x: -4, y: 0, z: 0 }, bodyTransform);

        expect(r.hit).toBe(true);
        expect(Math.abs(r.fraction - 0.125)).toBeLessThan(1e-2);
        expect(Math.abs(r.normal.x - 1)).toBeLessThan(1e-4);
        expect(Math.abs(r.point.x + 1)).toBeLessThan(1e-3);

        world.destroy();
    });

    test("far from origin", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(1, 1, 1));

        const origin = { x: 1.0e6, y: -2.0e6, z: 5.0e5 };
        const bodyTransform: Transform = {
            p: { x: origin.x + 5, y: origin.y, z: origin.z },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        };
        const r = body.castShape(origin, point, { x: 10, y: 0, z: 0 }, bodyTransform);

        expect(r.hit).toBe(true);
        expect(Math.abs(r.fraction - 0.35)).toBeLessThan(1e-2);
        expect(Math.abs(r.normal.x + 1)).toBeLessThan(1e-4);

        world.destroy();
    });
});

describe("body overlap shape", () => {
    const proxy: ShapeProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.5 };

    function boxBody(): { world: World; body: ReturnType<World["createBody"]> } {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(1, 1, 1));
        return { world, body };
    }

    test("true when proxy sits at the box center", () => {
        const { world, body } = boxBody();
        expect(body.overlapShape({ x: 5, y: 0, z: 0 }, proxy, identityAt(5, 0, 0))).toBe(true);
        world.destroy();
    });

    test("false when separated", () => {
        const { world, body } = boxBody();
        expect(body.overlapShape({ x: 20, y: 0, z: 0 }, proxy, identityAt(5, 0, 0))).toBe(false);
        world.destroy();
    });

    test("respects the supplied body transform", () => {
        const { world, body } = boxBody();
        const origin = { x: 0, y: 0, z: 0 };
        expect(body.overlapShape(origin, proxy, identityAt(0, 0, 0))).toBe(true);
        expect(body.overlapShape(origin, proxy, identityAt(20, 0, 0))).toBe(false);
        world.destroy();
    });

    test("a zero mask rejects every category", () => {
        const { world, body } = boxBody();
        const filter = defaultQueryFilter();
        filter.maskBits = 0n;
        expect(body.overlapShape({ x: 0, y: 0, z: 0 }, proxy, identityAt(0, 0, 0), filter)).toBe(
            false,
        );
        world.destroy();
    });
});

describe("body get closest point", () => {
    // Not a test_body_query subtest, but a small invariant guard for the closest-point query.
    // The query runs GJK with useRadii = false, so it measures to the sphere *center* (distance 3),
    // and the closest point is the center itself.
    test("distance to a unit sphere at the origin", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Static });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 1 });

        const { point, distance } = body.getClosestPoint({ x: 3, y: 0, z: 0 });
        expect(Math.abs(distance - 3)).toBeLessThan(1e-4);
        expect(Math.abs(point.x)).toBeLessThan(1e-4);

        world.destroy();
    });
});

// The world-level queries walk all three broad-phase trees, re-centering each candidate on the
// query origin, then run the same per-shape dispatch the body queries use. A shape's proxy is in
// the tree the moment it is created, so no step is needed. These smoke-test the tree-walk plumbing.
describe("world queries", () => {
    function sphereAt(world: World, p: Vec3, radius: number) {
        const body = world.createBody({ type: BodyType.Static, position: p });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius });
        return body;
    }

    test("castRayClosest picks the nearer of two bodies", () => {
        const world = new World();
        sphereAt(world, { x: 5, y: 0, z: 0 }, 1);
        const near = sphereAt(world, { x: 2, y: 0, z: 0 }, 0.5);

        const r = world.castRayClosest({ x: 0, y: 0, z: 0 }, { x: 10, y: 0, z: 0 });
        expect(r.hit).toBe(true);
        expect(r.shape?.getBody().id.index1).toBe(near.id.index1);
        // Front of the r=0.5 sphere at x=2 is x=1.5 → fraction 0.15.
        expect(Math.abs(r.fraction - 0.15)).toBeLessThan(1e-5);
        expect(Math.abs(r.point.x - 1.5)).toBeLessThan(1e-4);

        world.destroy();
    });

    test("castRayClosest misses when nothing is in the path", () => {
        const world = new World();
        sphereAt(world, { x: 5, y: 0, z: 0 }, 1);
        const r = world.castRayClosest({ x: 0, y: 10, z: 0 }, { x: 10, y: 0, z: 0 });
        expect(r.hit).toBe(false);
        expect(r.shape).toBeNull();
        world.destroy();
    });

    test("overlapAABB reports only shapes whose fat AABB overlaps the box", () => {
        const world = new World();
        sphereAt(world, { x: 0, y: 0, z: 0 }, 1);
        sphereAt(world, { x: 20, y: 0, z: 0 }, 1);

        const hits: number[] = [];
        world.overlapAABB(
            { lowerBound: { x: -2, y: -2, z: -2 }, upperBound: { x: 2, y: 2, z: 2 } },
            (shape) => {
                hits.push(shape.getBody().getPosition().x);
                return true;
            },
        );
        expect(hits.length).toBe(1);
        expect(Math.abs(hits[0])).toBeLessThan(1e-4);
        world.destroy();
    });

    test("overlapShape re-centers the proxy on the body transform", () => {
        const world = new World();
        sphereAt(world, { x: 3, y: 0, z: 0 }, 1);
        const proxy: ShapeProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.5 };

        let count = 0;
        world.overlapShape({ x: 3, y: 0, z: 0 }, proxy, () => {
            count += 1;
            return true;
        });
        expect(count).toBe(1);

        count = 0;
        world.overlapShape({ x: 10, y: 0, z: 0 }, proxy, () => {
            count += 1;
            return true;
        });
        expect(count).toBe(0);
        world.destroy();
    });

    test("castShape sweeps a proxy and clips to the closest hit", () => {
        const world = new World();
        const body = world.createBody({ type: BodyType.Static, position: { x: 5, y: 0, z: 0 } });
        body.createHull({}, makeBoxHull(1, 1, 1));
        const proxy: ShapeProxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: 0.5 };

        let best = 1;
        const stats = world.castShape({ x: 0, y: 0, z: 0 }, proxy, { x: 10, y: 0, z: 0 }, (hit) => {
            best = Math.min(best, hit.fraction);
            return hit.fraction;
        });
        // Box front face at x=4, proxy radius 0.5 → contact near fraction 0.35.
        expect(Math.abs(best - 0.35)).toBeLessThan(1e-2);
        expect(stats.leafVisits).toBeGreaterThan(0);
        world.destroy();
    });
});

describe("body collide mover", () => {
    test("touches a box's +Y face, depth 0.1", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));

        const mover: Capsule = {
            center1: { x: -0.3, y: 0.6, z: 0 },
            center2: { x: 0.3, y: 0.6, z: 0 },
            radius: 0.2,
        };
        const planes = body.collideMover({ x: 0, y: 0, z: 0 }, mover, identityAt(0, 0, 0));

        expect(planes.length).toBe(1);
        expect(planes[0].shape.isValid()).toBe(true);
        expect(planes[0].plane.plane.normal.y).toBeGreaterThan(0.99);
        expect(Math.abs(planes[0].plane.plane.offset - 0.1)).toBeLessThan(1e-4);
        world.destroy();
    });

    test("separated returns no plane", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));

        const mover: Capsule = {
            center1: { x: -0.3, y: 5, z: 0 },
            center2: { x: 0.3, y: 5, z: 0 },
            radius: 0.2,
        };
        const planes = body.collideMover({ x: 0, y: 0, z: 0 }, mover, identityAt(0, 0, 0));
        expect(planes.length).toBe(0);
        world.destroy();
    });

    test("a rotated body returns the plane normal in world space", () => {
        const world = new World();
        const body = world.createBody();
        body.createHull({}, makeBoxHull(0.5, 0.5, 0.5));

        // +90° about X turns the local +Y face toward world +Z; the mover sits above the world +Z face.
        const bodyTransform: Transform = {
            p: { x: 0, y: 0, z: 0 },
            q: quat.fromAxisAngle({ x: 1, y: 0, z: 0 }, 0.5 * PI),
        };
        const mover: Capsule = {
            center1: { x: -0.3, y: 0, z: 0.6 },
            center2: { x: 0.3, y: 0, z: 0.6 },
            radius: 0.2,
        };
        const planes = body.collideMover({ x: 0, y: 0, z: 0 }, mover, bodyTransform);

        expect(planes.length).toBe(1);
        expect(planes[0].plane.plane.normal.z).toBeGreaterThan(0.99);
        expect(Math.abs(planes[0].plane.plane.offset - 0.1)).toBeLessThan(1e-4);
        world.destroy();
    });

    test("capacity caps the returned planes", () => {
        const world = new World();
        const body = world.createBody();
        body.createSphere({}, { center: { x: -0.4, y: 0.6, z: 0 }, radius: 0.5 });
        body.createSphere({}, { center: { x: 0.4, y: 0.6, z: 0 }, radius: 0.5 });

        const mover: Capsule = {
            center1: { x: -1, y: 0, z: 0 },
            center2: { x: 1, y: 0, z: 0 },
            radius: 0.2,
        };
        const capped = body.collideMover({ x: 0, y: 0, z: 0 }, mover, identityAt(0, 0, 0), 1);
        expect(capped.length).toBe(1);
        const full = body.collideMover({ x: 0, y: 0, z: 0 }, mover, identityAt(0, 0, 0), 4);
        expect(full.length).toBe(2);
        world.destroy();
    });
});

describe("world collide mover / cast mover", () => {
    test("collideMover gathers a floor plane that solvePlanes pushes the mover up out of", () => {
        const world = new World();
        // A wide static floor whose top face sits at y=0.
        const floor = world.createBody({
            type: BodyType.Static,
            position: { x: 0, y: -0.5, z: 0 },
        });
        floor.createHull({}, makeBoxHull(5, 0.5, 5));

        // A mover sunk 0.1 into the floor (core at y=0.1, radius 0.2).
        const mover: Capsule = {
            center1: { x: -0.3, y: 0.1, z: 0 },
            center2: { x: 0.3, y: 0.1, z: 0 },
            radius: 0.2,
        };

        const planes: CollisionPlane[] = [];
        world.collideMover({ x: 0, y: 0, z: 0 }, mover, (_shape, results) => {
            for (const pr of results) {
                planes.push({ plane: pr.plane, pushLimit: 3.4e38, push: 0, clipVelocity: true });
            }
            return true;
        });
        expect(planes.length).toBeGreaterThan(0);

        // Resolving a downward target lifts the mover up along +Y and clips the downward velocity.
        const result = solvePlanes({ x: 0, y: -1, z: 0 }, planes, planes.length);
        expect(result.delta.y).toBeGreaterThan(-1);
        const clipped = clipVector({ x: 0, y: -1, z: 0 }, planes, planes.length);
        expect(clipped.y).toBeGreaterThan(-1e-4);
        world.destroy();
    });

    test("castMover stops before a wall but reports 1 on a clear path", () => {
        const world = new World();
        const wall = world.createBody({ type: BodyType.Static, position: { x: 5, y: 0, z: 0 } });
        wall.createHull({}, makeBoxHull(0.5, 5, 5));

        const mover: Capsule = {
            center1: { x: 0, y: -1, z: 0 },
            center2: { x: 0, y: 1, z: 0 },
            radius: 0.3,
        };
        const hit = world.castMover({ x: 0, y: 0, z: 0 }, mover, { x: 10, y: 0, z: 0 });
        expect(hit).toBeGreaterThan(0);
        expect(hit).toBeLessThan(1);

        const clear = world.castMover({ x: 0, y: 0, z: 0 }, mover, { x: 0, y: 10, z: 0 });
        expect(clear).toBe(1);
        world.destroy();
    });
});
