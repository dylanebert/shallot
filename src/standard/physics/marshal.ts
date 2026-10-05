import { Body, type BodyType, Hulls, ShapeKind } from "../../core/physics";
import type { World } from "../../engine";
import {
    createHull,
    defaultSurfaceMaterial,
    type HullData,
    makeBoxHull,
    type Body as SolverBody,
    type PhysicsWorld as SolverWorld,
} from "./api";

// ECS → physics marshaling — the ONLY place a Body's authored fields become a physics rigid body, so the
// dual-run hash gate (physics.test.ts) and StandardPhysicsPlugin's sync system read this one path.
// Constraint definitions are composed in authoring.ts and reconciled in joints.ts.

// null (not throw) on a missing/unbuildable hull: an unregistered hull id must not take down the whole
// SyncSystem frame loop — the caller warns + skips that one body, mirroring joints.ts's skip-a-bad-
// constraint convention. Point the warning at the diagnostic (the missing id), never the brand.
function hullFromRegistry(world: World, hullId: number): HullData | null {
    const hulls = world.resource(Hulls);
    const name = hulls.name(hullId);
    const entry = name ? hulls.get(name) : undefined;
    if (!entry) {
        console.warn(`[physics] no hull registered for id ${hullId} — skipping body`);
        return null;
    }
    const points = entry.verts.map(([x, y, z]) => ({ x, y, z }));
    const built = createHull(points, points.length);
    if (!built) {
        console.warn(
            `[physics] createHull failed for registered hull "${entry.name}" (id ${hullId}) — skipping body`,
        );
        return null;
    }
    return built;
}

/** attach `Body`'s collider to a freshly-created physics body, deriving the shape density from the authored
 *  `mass` (physics computes body mass FROM shape density × volume; a static/kinematic body's density is
 *  irrelevant — physics never derives mass for a non-dynamic body). */
function attachShape(
    world: World,
    tb: SolverBody,
    kind: number,
    hx: number,
    hy: number,
    hz: number,
    w: number,
    mass: number,
    friction: number,
): boolean {
    const baseMaterial = { ...defaultSurfaceMaterial(), friction };
    const density = (volume: number) => (mass > 0 && volume > 0 ? mass / volume : 0);

    if (kind === ShapeKind.Sphere) {
        const volume = (4 / 3) * Math.PI * w ** 3;
        tb.createSphere(
            { baseMaterial, density: density(volume) },
            { center: { x: 0, y: 0, z: 0 }, radius: w },
        );
        return true;
    }
    if (kind === ShapeKind.Capsule) {
        // the capsule core is local-Y: a segment of length 2·hy capped by radius w.
        const volume = Math.PI * w * w * (2 * hy) + (4 / 3) * Math.PI * w ** 3;
        tb.createCapsule(
            { baseMaterial, density: density(volume) },
            { center1: { x: 0, y: -hy, z: 0 }, center2: { x: 0, y: hy, z: 0 }, radius: w },
        );
        return true;
    }
    const hull = kind === ShapeKind.Hull ? hullFromRegistry(world, w) : makeBoxHull(hx, hy, hz);
    if (!hull) return false; // unregistered/unbuildable hull — the caller skips this body
    tb.createHull({ baseMaterial, density: density(hull.volume) }, hull);
    return true;
}

/** marshal `eid` (a live `Body`) into a fresh physics body in `world`: read the authored shape/
 *  pose/mass/friction off the `Body` slab and create the matching physics body + collider. `userData` carries
 *  `eid` so a `BodyMoveEvent` round-trips back to the entity without a reverse map. Deterministic given `eid`
 *  and the current `Body` field values — the dual-run marshaling gate (physics.test.ts) exercises this
 *  directly, both through a live `World` and by hand-authoring the same field values. Returns `null` when the
 *  body references an unregistered/unbuildable hull (the collider can't attach): it warns, destroys the empty
 *  body, and the caller skips this eid rather than letting the throw take down the frame loop. */
export function marshalBody(
    world: World,
    physicsWorld: SolverWorld,
    eid: number,
): SolverBody | null {
    const kind = world.storage(Body).shape.get(eid);
    const mass = world.storage(Body).mass.get(eid);
    const tb = physicsWorld.createBody({
        type: world.storage(Body).type.get(eid) as BodyType,
        position: {
            x: world.storage(Body).position.x.get(eid),
            y: world.storage(Body).position.y.get(eid),
            z: world.storage(Body).position.z.get(eid),
        },
        rotation: {
            v: {
                x: world.storage(Body).rotation.x.get(eid),
                y: world.storage(Body).rotation.y.get(eid),
                z: world.storage(Body).rotation.z.get(eid),
            },
            s: world.storage(Body).rotation.w.get(eid),
        },
        userData: eid,
    });
    const attached = attachShape(
        world,
        tb,
        kind,
        world.storage(Body).halfExtents.x.get(eid),
        world.storage(Body).halfExtents.y.get(eid),
        world.storage(Body).halfExtents.z.get(eid),
        world.storage(Body).halfExtents.w.get(eid),
        mass,
        world.storage(Body).friction.get(eid),
    );
    if (!attached) {
        tb.destroy();
        return null;
    }
    return tb;
}
