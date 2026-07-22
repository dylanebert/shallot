import { Body, ShapeKind } from "../physics";
import { Hulls } from "../physics/core";
import {
    BodyType,
    createHull,
    defaultShapeDef,
    defaultSurfaceMaterial,
    type HullData,
    makeBoxHull,
    type Body as TumbleBody,
    type World as TumbleWorld,
} from "./engine";

// ECS → tumble marshaling — the ONLY place a Body's authored fields become a tumble rigid body, so the
// dual-run hash gate (tumble.test.ts) and TumblePlugin's sync system read this one path. The Spring/Joint
// half of the seam is joints.ts (tumble.md "Constraint mapping"); this module is shape + mass + pose.

/** a `mass <= 0` `Body` marshals as `Kinematic` (velocity set via `PhysicsBackend.setKinematic`), never
 *  `Static` (which the engine never moves) — the substrate's mass<=0 contract covers both "never moves" and
 *  "scene-driven" (a platform, a grab anchor, the character sweep) uniformly, so every mass<=0 Body needs the
 *  type that accepts a velocity write. */
function bodyType(mass: number): BodyType {
    return mass > 0 ? BodyType.Dynamic : BodyType.Kinematic;
}

// null (not throw) on a missing/unbuildable hull: an unregistered hull id must not take down the whole
// SyncSystem frame loop — the caller warns + skips that one body, mirroring joints.ts's skip-a-bad-
// constraint convention. Point the warning at the diagnostic (the missing id), never the brand.
function hullFromRegistry(hullId: number): HullData | null {
    const name = Hulls.name(hullId);
    const entry = name ? Hulls.get(name) : undefined;
    if (!entry) {
        console.warn(`[tumble] no hull registered for id ${hullId} — skipping body`);
        return null;
    }
    const points = entry.verts.map(([x, y, z]) => ({ x, y, z }));
    const built = createHull(points, Math.max(points.length, 4));
    if (!built) {
        console.warn(
            `[tumble] createHull failed for registered hull "${entry.name}" (id ${hullId}) — skipping body`,
        );
        return null;
    }
    return built;
}

/** attach `Body`'s collider to a freshly-created tumble body, deriving the shape density from the authored
 *  `mass` (tumble computes body mass FROM shape density × volume; a static/kinematic body's density is
 *  irrelevant — tumble never derives mass for a non-dynamic body). */
function attachShape(
    tb: TumbleBody,
    kind: number,
    hx: number,
    hy: number,
    hz: number,
    w: number,
    mass: number,
    friction: number,
): boolean {
    const baseMaterial = { ...defaultSurfaceMaterial(), friction };
    const density = (volume: number) =>
        mass > 0 && volume > 0 ? mass / volume : defaultShapeDef().density;

    if (kind === ShapeKind.Sphere) {
        const volume = (4 / 3) * Math.PI * w ** 3;
        tb.createSphere(
            { baseMaterial, density: density(volume) },
            { center: { x: 0, y: 0, z: 0 }, radius: w },
        );
        return true;
    }
    if (kind === ShapeKind.Capsule) {
        // the capsule core is local-Y (physics.md): a segment of length 2·hy capped by radius w.
        const volume = Math.PI * w * w * (2 * hy) + (4 / 3) * Math.PI * w ** 3;
        tb.createCapsule(
            { baseMaterial, density: density(volume) },
            { center1: { x: 0, y: -hy, z: 0 }, center2: { x: 0, y: hy, z: 0 }, radius: w },
        );
        return true;
    }
    const hull = kind === ShapeKind.Hull ? hullFromRegistry(w) : makeBoxHull(hx, hy, hz);
    if (!hull) return false; // unregistered/unbuildable hull — the caller skips this body
    tb.createHull({ baseMaterial, density: density(hull.volume) }, hull);
    return true;
}

/** marshal a scene's `eid` (a live `Body`) into a fresh tumble body in `world`: read the authored shape/
 *  pose/mass/friction off the `Body` slab and create the matching tumble body + collider. `userData` carries
 *  `eid` so a `BodyMoveEvent` round-trips back to the entity without a reverse map. Deterministic given `eid`
 *  and the current `Body` field values — the dual-run marshaling gate (tumble.test.ts) exercises this
 *  directly, both through a live `State` and by hand-authoring the same field values. Returns `null` when the
 *  body references an unregistered/unbuildable hull (the collider can't attach): it warns, destroys the empty
 *  body, and the caller skips this eid rather than letting the throw take down the frame loop. */
export function marshalBody(world: TumbleWorld, eid: number): TumbleBody | null {
    const kind = Body.shape.get(eid);
    const mass = Body.mass.get(eid);
    const tb = world.createBody({
        type: bodyType(mass),
        position: { x: Body.pos.x.get(eid), y: Body.pos.y.get(eid), z: Body.pos.z.get(eid) },
        rotation: {
            v: { x: Body.quat.x.get(eid), y: Body.quat.y.get(eid), z: Body.quat.z.get(eid) },
            s: Body.quat.w.get(eid),
        },
        userData: eid,
    });
    const attached = attachShape(
        tb,
        kind,
        Body.halfExtents.x.get(eid),
        Body.halfExtents.y.get(eid),
        Body.halfExtents.z.get(eid),
        Body.halfExtents.w.get(eid),
        mass,
        Body.friction.get(eid),
    );
    if (!attached) {
        tb.destroy();
        return null;
    }
    return tb;
}
