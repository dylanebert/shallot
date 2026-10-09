import {
    type Quat,
    type Body as SolverBody,
    Joint as SolverJoint,
    type PhysicsWorld as SolverWorld,
    type Transform,
} from "./api";
import type { JointDef } from "./authoring";

// Definitions are keyed by content to preserve warm-start impulses on unchanged joints.
const keyValue = (_key: string, value: unknown): unknown =>
    typeof value === "number" && !Number.isFinite(value) ? String(value) : value;
const jointKey = (def: JointDef): string => JSON.stringify(def, keyValue);
export interface ConstraintCache {
    liveJoints: Map<string, SolverJoint[]>;
    retainedJoints: readonly JointDef[];
    warnedJoints: Set<string>;
}
export function createConstraintCache(): ConstraintCache {
    return { liveJoints: new Map(), retainedJoints: [], warnedJoints: new Set() };
}
function warnOnce(warned: Set<string>, key: string, message: string): void {
    if (warned.has(key)) return;
    console.warn(message);
    warned.add(key);
}

export function syncSet<D>(
    live: Map<string, SolverJoint[]>,
    defs: readonly D[],
    keyOf: (def: D) => string,
    create: (def: D) => SolverJoint | null,
): void {
    const next = new Map<string, SolverJoint[]>();
    for (const def of defs) {
        const key = keyOf(def);
        const pool = live.get(key);
        let joint: SolverJoint | null = null;
        for (let j = pool?.pop(); j; j = pool?.pop()) {
            if (j.isValid()) {
                joint = j;
                break;
            }
        }
        joint ??= create(def);
        if (!joint) continue;
        const bucket = next.get(key);
        if (bucket) bucket.push(joint);
        else next.set(key, [joint]);
    }
    for (const pool of live.values()) for (const j of pool) if (j.isValid()) j.destroy();
    live.clear();
    for (const [key, pool] of next) live.set(key, pool);
}

function endpoints(
    bodies: ReadonlyMap<number, SolverBody>,
    def: JointDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<string>,
    key: string,
): [SolverBody, SolverBody] | null {
    const a = bodies.get(def.a);
    const b = bodies.get(def.b);
    if (a && b) return [a, b];
    const cause = (label: string, eid: number) =>
        isDeferred(eid)
            ? `${label}: ${eid} is a deferred body (marshal pending, will retry)`
            : `${label}: ${eid} is not a Body (skipped)`;
    const parts: string[] = [];
    if (!a) parts.push(cause("a", def.a));
    if (!b) parts.push(cause("b", def.b));
    warnOnce(
        warned,
        `${key}|endpoint|${parts.join(";")}`,
        `[physics] ${def.kind}Joint ${def.eid} endpoint unavailable — ${parts.join("; ")}`,
    );
    return null;
}
const validQuat = (q: Quat): boolean => {
    if (![q.v.x, q.v.y, q.v.z, q.s].every(Number.isFinite)) return false;
    const f = Math.fround;
    const lengthSquared = f(
        f(f(f(q.v.x * q.v.x) + f(q.v.y * q.v.y)) + f(q.v.z * q.v.z)) + f(q.s * q.s),
    );
    const tolerance = 20 * 2 ** -23;
    return 1 - tolerance < lengthSquared && lengthSquared < 1 + tolerance;
};
const validFrame = (frame: Transform): boolean =>
    [frame.p.x, frame.p.y, frame.p.z].every(Number.isFinite) && validQuat(frame.q);

/** Only the assertions in Box3D's create functions refuse authored values; solver clamping stays in the solver. */
export function invalidJointField(def: JointDef): string | null {
    const c = def.config;
    if (!validFrame(c.localFrameA)) return "localFrameA";
    if (!validFrame(c.localFrameB)) return "localFrameB";
    switch (def.kind) {
        case "Distance":
            if (!Number.isFinite(c.length) || !(c.length! > 0)) return "length";
            if (!(c.lowerSpringForce! <= c.upperSpringForce!))
                return "lowerSpringForce/upperSpringForce";
            break;
        case "Parallel":
            for (const name of ["hertz", "dampingRatio", "maxTorque"] as const)
                if (!Number.isFinite(c[name]) || !(c[name]! >= 0)) return name;
            break;
        case "Prismatic":
            if (!(c.lowerTranslation! <= c.upperTranslation!))
                return "lowerTranslation/upperTranslation";
            break;
        case "Spherical":
            if (
                !(
                    c.coneAngle! >= 0 &&
                    c.coneAngle! <= Math.fround(Math.fround(0.99) * Math.fround(Math.PI))
                )
            )
                return "coneAngle";
            if (!validQuat(c.targetRotation!)) return "targetRotation";
            break;
        case "Weld":
            for (const name of [
                "angularHertz",
                "angularDampingRatio",
                "linearHertz",
                "linearDampingRatio",
            ] as const)
                if (!(c[name]! >= 0)) return name;
            break;
        case "Wheel":
            if (!(c.lowerSuspensionLimit! <= c.upperSuspensionLimit!))
                return "lowerSuspensionLimit/upperSuspensionLimit";
            break;
    }
    return null;
}
function createJoint(
    world: SolverWorld,
    bodies: ReadonlyMap<number, SolverBody>,
    def: JointDef,
    isDeferred: (eid: number) => boolean,
    warned: Set<string>,
): SolverJoint | null {
    const key = jointKey(def);
    const pair = endpoints(bodies, def, isDeferred, warned, key);
    if (!pair) return null;
    const invalid = invalidJointField(def);
    if (invalid) {
        warnOnce(
            warned,
            `${key}|${invalid}`,
            `[physics] ${def.kind}Joint ${def.eid} has invalid ${invalid} — skipped`,
        );
        return null;
    }
    const [a, b] = pair;
    const config = def.config;
    switch (def.kind) {
        case "Distance":
            return world.createDistanceJoint(a, b, config);
        case "Filter":
            return world.createFilterJoint(a, b, config);
        case "Motor":
            return world.createMotorJoint(a, b, config);
        case "Parallel":
            return world.createParallelJoint(a, b, config);
        case "Prismatic":
            return world.createPrismaticJoint(a, b, config);
        case "Revolute":
            return world.createRevoluteJoint(a, b, config);
        case "Spherical":
            return world.createSphericalJoint(a, b, config);
        case "Weld":
            return world.createWeldJoint(a, b, config);
        case "Wheel":
            return world.createWheelJoint(a, b, config);
    }
}
export function syncJoints(
    cache: ConstraintCache,
    world: SolverWorld,
    bodies: ReadonlyMap<number, SolverBody>,
    defs: readonly JointDef[],
    isDeferred: (eid: number) => boolean,
): void {
    cache.retainedJoints = defs;
    cache.warnedJoints.clear();
    syncSet(cache.liveJoints, defs, jointKey, (def) =>
        createJoint(world, bodies, def, isDeferred, cache.warnedJoints),
    );
}
/** Retry deferred endpoints when the body set changes, preserving valid handles and deduplicating diagnostics by cause. */
export function resyncConstraints(
    cache: ConstraintCache,
    world: SolverWorld,
    bodies: ReadonlyMap<number, SolverBody>,
    isDeferred: (eid: number) => boolean,
): void {
    if (cache.retainedJoints.length)
        syncSet(cache.liveJoints, cache.retainedJoints, jointKey, (def) =>
            createJoint(world, bodies, def, isDeferred, cache.warnedJoints),
        );
}
export interface ConstraintIds {
    joints: [string, number[]][];
    retainedJoints: readonly JointDef[];
}
export function captureConstraints(cache: ConstraintCache): ConstraintIds {
    return {
        joints: Array.from(cache.liveJoints, ([key, pool]) => [
            key,
            pool.flatMap((j) => [j.id.index1, j.id.generation]),
        ]),
        retainedJoints: cache.retainedJoints,
    };
}
export function restoreConstraints(
    cache: ConstraintCache,
    ids: ConstraintIds,
    world: SolverWorld,
): void {
    cache.liveJoints = new Map(
        ids.joints.map(([key, flat]) => {
            const pool: SolverJoint[] = [];
            for (let i = 0; i < flat.length; i += 2)
                pool.push(
                    new SolverJoint(world.state, {
                        index1: flat[i],
                        world0: world.state.worldId,
                        generation: flat[i + 1],
                    }),
                );
            return [key, pool];
        }),
    );
    cache.retainedJoints = ids.retainedJoints;
}
export function resetConstraints(cache: ConstraintCache): void {
    cache.liveJoints.clear();
    cache.retainedJoints = [];
    cache.warnedJoints.clear();
}
