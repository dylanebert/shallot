import type { JointDef, SpringDef } from "../physics";
import {
    BodyType,
    type Quat,
    type Transform,
    type Body as TumbleBody,
    type Joint as TumbleJoint,
    type World as TumbleWorld,
} from "./engine";

// Spring/Joint def → tumble joint marshaling — the constraint half of the ECS→tumble path
// (marshal.ts is the body half). The substrate's ConstraintSystem uploads the full authored set on
// change; this module diffs it against the live set by def CONTENT, so an unchanged constraint keeps
// its live tumble joint and its warm-started impulses survive a re-author (the AVBD setJoints
// kept-slot contract, physics.md "Re-upload only on change"). The mapping (tumble.md
// "Constraint mapping"): Spring → DistanceJoint-with-spring (stiffness N/m → hertz via the pair's
// reduced mass), Joint → Spherical (stiffnessAng 0) / Weld (rigid past the ∞ sentinel; intermediate
// is the documented hertz-based approximation). Both backends reject a constraint no dynamic body
// can satisfy — AVBD deactivates + counts (its GPU can't throw), this path warns + skips.

/** AVBD's ∞-stiffness sentinel (physics.md): past this, `stiffnessAng` reads rigid. */
const RIGID_THRESHOLD = 1e29;

/**
 * convert a stiffness (N/m) to the box3d soft-constraint frequency (Hz) for the pair's reduced mass:
 * `k = m_eff·ω²` with `m_eff = mA·mB/(mA+mB)`, a non-dynamic endpoint contributing ∞ mass (so
 * `m_eff` = the dynamic side's mass). Reproduces the N/m force law — the static extension under
 * gravity is `mg/k` on both backends. Pass 0 for a non-dynamic endpoint; returns 0 when neither
 * endpoint is dynamic (nothing the constraint could move — the caller skips it).
 */
export function stiffnessHertz(stiffness: number, massA: number, massB: number): number {
    const meff =
        massA > 0 && massB > 0 ? (massA * massB) / (massA + massB) : Math.max(massA, massB);
    if (meff <= 0 || stiffness <= 0) return 0;
    return Math.sqrt(stiffness / meff) / (2 * Math.PI);
}

const IDENTITY: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };

function frame(p: readonly [number, number, number], q: Quat = IDENTITY): Transform {
    return { p: { x: p[0], y: p[1], z: p[2] }, q };
}

// qB⁻¹ ⊗ qA: the frame-B rotation that makes both weld frames coincide in world at the spawn pose,
// so the weld holds the AUTHORED relative orientation (frame q identity would snap the pair to
// aligned axes instead). qA·qfA = qB·qfB with qfA = identity ⇒ qfB = qB⁻¹·qA.
function relRotation(a: TumbleBody, b: TumbleBody): Quat {
    const qa = a.getRotation();
    const qb = b.getRotation();
    const bx = -qb.v.x;
    const by = -qb.v.y;
    const bz = -qb.v.z;
    const bs = qb.s;
    return {
        v: {
            x: bs * qa.v.x + qa.s * bx + by * qa.v.z - bz * qa.v.y,
            y: bs * qa.v.y + qa.s * by + bz * qa.v.x - bx * qa.v.z,
            z: bs * qa.v.z + qa.s * bz + bx * qa.v.y - by * qa.v.x,
        },
        s: bs * qa.s - bx * qa.v.x - by * qa.v.y - bz * qa.v.z,
    };
}

const dynMass = (tb: TumbleBody): number =>
    tb.getType() === BodyType.Dynamic ? tb.getMassData().mass : 0;

const springKey = (d: SpringDef): string =>
    `${d.a}|${d.b}|${d.rA}|${d.rB}|${d.stiffness}|${d.rest}`;
const jointKey = (d: JointDef): string => `${d.a}|${d.b}|${d.rA}|${d.rB}|${d.stiffnessAng}`;

// the live tumble joints per def key — arrays because identical defs are legal (two equal springs
// both pull). A destroyed Body took its joints with it (tumble cascades), so a kept handle is
// re-checked via isValid() before reuse.
const liveSprings = new Map<string, TumbleJoint[]>();
const liveJoints = new Map<string, TumbleJoint[]>();
// the last authored def sets — retained so `SyncSystem` can re-invoke `syncJoints`/`syncSprings` over them
// when a deferred body marshals (the pump half of the late-marshal fix). AVBD's `setJoints` already retains
// the authored set (avbd/step.ts); this mirrors that contract. A ledger without a pump is inert: nothing
// calls back into `joints.ts` between signature changes, so the retained set + the re-invoke are both needed.
let retainedSprings: readonly SpringDef[] = [];
let retainedJoints: readonly JointDef[] = [];
// warned-key dedupe sets — keyed on `${springKey}|${cause}` / `${jointKey}|${cause}` so each diagnostic site
// (endpoint-unavailable, hertz-zero, both-static, non-positive-stiffness) is deduped independently. The
// endpoint-unavailable cause folds in the per-endpoint classification (`parts`), so a narrowed composition
// (a deferred half marshals, leaving only the genuinely-non-`Body` half) is a DISTINCT key that re-warns
// once — a mixed pair does not inherit the authored upload's stale key. Cleared on an authored upload
// (`syncSprings`/`syncJoints`) and in `resetConstraints`. This mirrors `index.ts`'s `failed` ledger: it gates
// the *attempt* and re-warns when the key moves — it does not mute. A boolean `quiet` that silenced the retry
// path (S2's stopgap) hid every cause that only becomes visible after a deferred marshal resolves (the
// both-static guard sits after `endpoints()` returns a pair, so the retry is the only path that can evaluate
// it). Dedupe on the composite key lets the both-static warning fire once on the retry even though the
// endpoint warning already banked the def's base key on the authored upload.
const warnedSprings = new Set<string>();
const warnedJoints = new Set<string>();

const warnOnce = (warned: Set<string>, key: string, message: string): void => {
    if (warned.has(key)) return;
    console.warn(message);
    warned.add(key);
};

// exported as a test seam only (joints.test.ts pins the diff semantics with stub joints); not on any barrel
export function syncSet<D>(
    live: Map<string, TumbleJoint[]>,
    defs: readonly D[],
    keyOf: (d: D) => string,
    create: (d: D) => TumbleJoint | null,
): void {
    const next = new Map<string, TumbleJoint[]>();
    for (const def of defs) {
        const key = keyOf(def);
        const pool = live.get(key);
        let joint: TumbleJoint | null = null;
        for (let j = pool?.pop(); j; j = pool?.pop()) {
            if (j.isValid()) {
                joint = j;
                break;
            }
        }
        joint ??= create(def);
        if (!joint) continue;
        const bucket = next.get(key);
        if (bucket) {
            bucket.push(joint);
        } else {
            next.set(key, [joint]);
        }
    }
    for (const pool of live.values()) {
        for (const j of pool) {
            if (j.isValid()) j.destroy();
        }
    }
    live.clear();
    for (const [k, v] of next) live.set(k, v);
}

// the discriminator for the deferred-marshal case: `endpoints()` is handed only the `bodies` map, so without
// this predicate it cannot separate "not a Body" from "a Body whose marshal is pending" — which is why the
// old warning named the wrong cause. The `failed` map (index.ts) sits beside `bodies` in the same module;
// pass a predicate over it so the deferred case reads as deferred-and-will-retry, not a permanent skip.
// EACH missing endpoint is classified on its own cause: a mixed pair (one truly non-`Body`, one deferred)
// names both, so neither half inherits the other's cause — the misdirection this discriminator exists to end.
// The diagnostic is deduped (not silenced): `warned` is the per-cause key set cleared on authored upload, so
// the authored upload's warning fires once and the retry's re-warn is suppressed only for the SAME cause —
// a deferred endpoint that resolves into a both-static pair warns the both-static cause on the retry because
// that composite key was never banked. The endpoint key itself folds in `parts`, so a narrowed composition
// (a deferred half marshals, leaving fewer missing endpoints) is a distinct key that re-warns once — the
// authored upload's broader warning does not swallow the retry's corrected, narrower diagnostic.
function endpoints(
    bodies: ReadonlyMap<number, TumbleBody>,
    a: number,
    b: number,
    kind: string,
    isDeferred: (eid: number) => boolean,
    warned: Set<string>,
    key: string,
): [TumbleBody, TumbleBody] | null {
    const ta = bodies.get(a);
    const tb = bodies.get(b);
    if (!ta || !tb) {
        const cause = (label: string, eid: number): string =>
            isDeferred(eid)
                ? `${label}: ${eid} is a deferred body (marshal pending, will retry)`
                : `${label}: ${eid} is not a Body (skipped)`;
        const parts: string[] = [];
        if (!ta) parts.push(cause("a", a));
        if (!tb) parts.push(cause("b", b));
        warnOnce(
            warned,
            `${key}|endpoint|${parts.join(";")}`,
            `[tumble] ${kind} endpoint unavailable — ${parts.join("; ")}`,
        );
        return null;
    }
    return [ta, tb];
}

function createSpring(
    world: TumbleWorld,
    bodies: ReadonlyMap<number, TumbleBody>,
    def: SpringDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<string>,
): TumbleJoint | null {
    const key = springKey(def);
    const pair = endpoints(bodies, def.a, def.b, "spring", isDeferred, warned, key);
    if (!pair) return null;
    const hertz = stiffnessHertz(def.stiffness, dynMass(pair[0]), dynMass(pair[1]));
    if (hertz === 0) {
        warnOnce(
            warned,
            `${key}|hertz`,
            `[tumble] spring (a: ${def.a}, b: ${def.b}) has no dynamic endpoint or non-positive stiffness — skipped`,
        );
        return null;
    }
    return world.createDistanceJoint(pair[0], pair[1], {
        localFrameA: frame(def.rA),
        localFrameB: frame(def.rB),
        length: def.rest,
        enableSpring: true,
        hertz,
        // critically damped, NOT the literal undamped elastic law: AVBD's BDF1 integration heavily
        // damps its f = k·C spring, so both backends settling to the same mg/k equilibrium (which is
        // damping-independent) is the parity behavior the swap contract asserts; an undamped tumble
        // spring would ring forever where the AVBD one settles.
        dampingRatio: 1,
    });
}

function createJoint(
    world: TumbleWorld,
    bodies: ReadonlyMap<number, TumbleBody>,
    def: JointDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<string>,
): TumbleJoint | null {
    const key = jointKey(def);
    const pair = endpoints(bodies, def.a, def.b, "joint", isDeferred, warned, key);
    if (!pair) return null;
    const [ta, tb] = pair;
    const mA = dynMass(ta);
    const mB = dynMass(tb);
    if (mA <= 0 && mB <= 0) {
        warnOnce(
            warned,
            `${key}|both-static`,
            `[tumble] joint (a: ${def.a}, b: ${def.b}) has no dynamic endpoint — unsatisfiable, skipped (the both-static guard)`,
        );
        return null;
    }
    if (def.stiffnessAng === 0) {
        return world.createSphericalJoint(ta, tb, {
            localFrameA: frame(def.rA),
            localFrameB: frame(def.rB),
        });
    }
    if (def.stiffnessAng < 0) {
        warnOnce(
            warned,
            `${key}|stiffness`,
            `[tumble] joint (a: ${def.a}, b: ${def.b}) has non-positive angular stiffness — skipped`,
        );
        return null;
    }
    // angularHertz 0 is box3d's RIGID angular constraint; an intermediate stiffnessAng maps to a soft
    // angular spring via the same reduced-mass conversion (a unit-arm approximation, I_eff ≈ m_eff —
    // the documented backend-approximate seam, tumble.md "Constraint mapping")
    const angularHertz =
        def.stiffnessAng > RIGID_THRESHOLD ? 0 : stiffnessHertz(def.stiffnessAng, mA, mB);
    return world.createWeldJoint(ta, tb, {
        localFrameA: frame(def.rA),
        localFrameB: { p: { x: def.rB[0], y: def.rB[1], z: def.rB[2] }, q: relRotation(ta, tb) },
        linearHertz: 0, // rigid pin, both mappings
        angularHertz,
        // critically damped, matching the spring path (joints.ts createSpring) and the swap-parity rule
        // this file states above (dampingRatio: 1 because settle-to-equilibrium is the behavior that
        // matches across the swap); an undamped angular spring would ring where the AVBD one settles.
        angularDampingRatio: 1,
    });
}

/** reconcile the authored spring set against the live tumble joints: unchanged defs keep their joint (warm-started impulses survive), changed/new defs create, leftovers destroy. Retains the def set for `resyncConstraints` and clears the warned-key set so the authored upload's diagnostics fire fresh. */
export function syncSprings(
    world: TumbleWorld,
    bodies: ReadonlyMap<number, TumbleBody>,
    defs: readonly SpringDef[],
    isDeferred: (eid: number) => boolean,
): void {
    retainedSprings = defs;
    warnedSprings.clear();
    syncSet(liveSprings, defs, springKey, (d) =>
        createSpring(world, bodies, d, isDeferred, warnedSprings),
    );
}

/** reconcile the authored joint set against the live tumble joints — the `syncSprings` twin over the Spherical/Weld mapping. Retains the def set for `resyncConstraints` and clears the warned-key set so the authored upload's diagnostics fire fresh. */
export function syncJoints(
    world: TumbleWorld,
    bodies: ReadonlyMap<number, TumbleBody>,
    defs: readonly JointDef[],
    isDeferred: (eid: number) => boolean,
): void {
    retainedJoints = defs;
    warnedJoints.clear();
    syncSet(liveJoints, defs, jointKey, (d) =>
        createJoint(world, bodies, d, isDeferred, warnedJoints),
    );
}

/** re-invoke `syncJoints`/`syncSprings` over the retained def sets — the pump half of the late-marshal fix.
 *  Called by `SyncSystem` on ANY body-set change: normally that is the deferred-marshal transition or a body
 *  going stale, but a continuously spawning scene changes its body set every tick and so re-syncs every tick —
 *  a no-op walk over the live joints (each `isValid()` handle is reused, nothing is created or destroyed).
 *  `syncSet` needs no new diff semantics: an unchanged def whose live joint `isValid()` is reused (warm-started
 *  impulses survive), a dropped def's `create` retries, leftovers destroy — the semantics `joints.test.ts`
 *  already pins. The retry's diagnostics are deduped (not silenced): the warned-key set is NOT cleared here, so
 *  a warning that already fired for a given cause on the authored upload is suppressed on the retry — but a
 *  cause that only becomes visible after the marshal resolves (the both-static guard) fires once, because its
 *  composite key was never banked (index.ts's never-thrash-the-frame-loop invariant). */
export function resyncConstraints(
    world: TumbleWorld,
    bodies: ReadonlyMap<number, TumbleBody>,
    isDeferred: (eid: number) => boolean,
): void {
    if (retainedSprings.length > 0)
        syncSet(liveSprings, retainedSprings, springKey, (d) =>
            createSpring(world, bodies, d, isDeferred, warnedSprings),
        );
    if (retainedJoints.length > 0)
        syncSet(liveJoints, retainedJoints, jointKey, (d) =>
            createJoint(world, bodies, d, isDeferred, warnedJoints),
        );
}

/** drop every tracked joint handle without destroying (the world they lived in is gone). Call beside the world teardown in `warm()`/`dispose()`. */
export function resetConstraints(): void {
    liveSprings.clear();
    liveJoints.clear();
    retainedSprings = [];
    retainedJoints = [];
    warnedSprings.clear();
    warnedJoints.clear();
}
