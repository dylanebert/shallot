import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
    BodyType,
    createBoxMesh,
    type Joint,
    makeBoxHull,
    type Vec3 as PhysicsVec3,
    World,
} from "../api/index";
import { ContactFlags } from "../collision/contact";
import { emptyCache, shapeDistance } from "../collision/distance";
import {
    collideHulls,
    collideSpheres,
    emptySATCache,
    makeLocalManifold,
} from "../collision/manifold";
import { type CollisionPlane, clipVector, solvePlanes } from "../collision/mover";
import { createProxy, createTree, query } from "../collision/tree";
import {
    type AABB,
    computeCosSin,
    pointToSegmentDistance,
    type Quat,
    type Transform,
    type Vec3,
    vec3,
} from "../common/math";
import {
    type Capsule,
    computeCapsuleMass,
    computeSphereAABB,
    rayCastSphere,
    type Sphere,
} from "../shapes/geometry";
import { hashWorldStateOracleSentinel } from "../world/hash";
import { loadScenarioCorpus, runScenario, type ScenarioOutput } from "./scenario";

export type OracleCase = {
    id: string;
    family: string;
    symbol: string;
    input: unknown;
    output: unknown;
    configuration?: string;
};

const identity: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const f32 = (hex: string): number => {
    if (!/^0x[0-9a-f]{8}$/.test(hex)) throw new Error(`invalid f32 bits: ${hex}`);
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, Number.parseInt(hex.slice(2), 16), false);
    return view.getFloat32(0, false);
};
const bits = (value: number): string => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value, false);
    return `0x${view.getUint32(0, false).toString(16).padStart(8, "0")}`;
};
const u64 = (hex: string): bigint => BigInt(hex);
const vec = (values: unknown): Vec3 => {
    if (!Array.isArray(values) || values.length !== 3) throw new Error("expected f32 vec3");
    return { x: f32(String(values[0])), y: f32(String(values[1])), z: f32(String(values[2])) };
};
const transform = (translation: unknown): Transform => ({ p: vec(translation), q: identity });
const outVec = (value: Vec3): string[] => [bits(value.x), bits(value.y), bits(value.z)];
const aabb = (value: AABB): { lower: string[]; upper: string[] } => ({
    lower: outVec(value.lowerBound),
    upper: outVec(value.upperBound),
});

const u32hex = (value: number): string => `0x${(value >>> 0).toString(16).padStart(8, "0")}`;
const i32hex = (value: number): string => u32hex(value);
const vbits = (value: PhysicsVec3): string[] => [bits(value.x), bits(value.y), bits(value.z)];
const zero: PhysicsVec3 = { x: 0, y: 0, z: 0 };

function makeOracleWorld(): World {
    return new World({
        gravity: { x: 0, y: -9.8, z: 0 },
        enableSleep: false,
    });
}

function contactRecords(world: World): Array<{
    manifolds: Array<{
        normal: PhysicsVec3;
        pointCount: number;
        points: Array<{
            anchorA: PhysicsVec3;
            anchorB: PhysicsVec3;
            separation: number;
            featureId: number;
            triangleIndex: number;
        }>;
    }>;
}> {
    return world.state.contacts
        .filter(
            (contact) =>
                contact.contactId >= 0 && (contact.flags & ContactFlags.simTouchingFlag) !== 0,
        )
        .map((contact) => ({
            manifolds: contact.manifolds.slice(0, contact.manifoldCount).map((manifold) => ({
                normal: { ...manifold.normal },
                pointCount: manifold.pointCount,
                points: manifold.points.slice(0, manifold.pointCount).map((point) => ({
                    anchorA: { ...point.anchorA },
                    anchorB: { ...point.anchorB },
                    separation: point.separation,
                    featureId: point.featureId,
                    triangleIndex: point.triangleIndex,
                })),
            })),
        }));
}

function writeContacts(world: World, hookVisits: number): unknown {
    const contacts = contactRecords(world);
    return {
        contactCount: i32hex(contacts.length),
        manifolds: contacts.map((contact) => ({
            manifoldCount: i32hex(contact.manifolds.length),
            items: contact.manifolds.map((manifold) => ({
                normal: vbits(manifold.normal),
                pointCount: i32hex(manifold.pointCount),
                points: manifold.points.map((point) => ({
                    anchorA: vbits(point.anchorA),
                    anchorB: vbits(point.anchorB),
                    separation: bits(point.separation),
                    featureId: u32hex(point.featureId),
                    triangleIndex: i32hex(point.triangleIndex),
                })),
            })),
        })),
        hookVisits: u32hex(hookVisits),
    };
}

function runJointCase(item: OracleCase): unknown {
    const world = makeOracleWorld();
    const bodyA = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 0, y: 0, z: 0 },
        enableSleep: false,
    });
    const bodyB = world.createBody({
        type: BodyType.Dynamic,
        position: { x: 1, y: 0, z: 0 },
        enableSleep: false,
    });
    const name = item.id.split(".")[2];
    let joint: Joint;
    let specific = 0;
    let specificVector: PhysicsVec3 | undefined;
    switch (name) {
        case "parallel":
            joint = world.createParallelJoint(bodyA, bodyB);
            specific = (joint as ReturnType<World["createParallelJoint"]>).getSpringHertz();
            break;
        case "distance":
            joint = world.createDistanceJoint(bodyA, bodyB, { length: 1 });
            specific = (joint as ReturnType<World["createDistanceJoint"]>).getLength();
            break;
        case "motor": {
            const motor = world.createMotorJoint(bodyA, bodyB);
            joint = motor;
            specific = motor.getMaxVelocityForce();
            specificVector = motor.getLinearVelocity();
            break;
        }
        case "filter":
            joint = world.createFilterJoint(bodyA, bodyB);
            break;
        case "prismatic": {
            const prismatic = world.createPrismaticJoint(bodyA, bodyB);
            joint = prismatic;
            specific = prismatic.getTranslation();
            break;
        }
        case "revolute": {
            const revolute = world.createRevoluteJoint(bodyA, bodyB);
            joint = revolute;
            specific = revolute.getAngle();
            break;
        }
        case "spherical": {
            const spherical = world.createSphericalJoint(bodyA, bodyB);
            joint = spherical;
            specific = spherical.getConeAngle();
            specificVector = spherical.getMotorVelocity();
            break;
        }
        case "weld": {
            const weld = world.createWeldJoint(bodyA, bodyB);
            joint = weld;
            specific = weld.getLinearHertz();
            break;
        }
        case "wheel": {
            const wheel = world.createWheelJoint(bodyA, bodyB);
            joint = wheel;
            specific = wheel.getSteeringAngle();
            break;
        }
        default:
            throw new Error(`unknown O4 joint ${item.id}`);
    }
    world.step(1 / 60, 1);
    const result: Record<string, unknown> = {
        valid: u32hex(joint.isValid() ? 1 : 0),
        type: u32hex(joint.getType()),
        force: vbits(joint.getConstraintForce()),
        torque: vbits(joint.getConstraintTorque()),
        linearSeparation: bits(joint.getLinearSeparation()),
        angularSeparation: bits(joint.getAngularSeparation()),
        specific: bits(specific),
        hookVisits: u32hex(
            (
                {
                    parallel: 1,
                    distance: 2,
                    motor: 3,
                    filter: 4,
                    prismatic: 5,
                    revolute: 6,
                    spherical: 7,
                    weld: 8,
                    wheel: 9,
                } as Record<string, number>
            )[name] ?? 0,
        ),
    };
    if (specificVector !== undefined) result.specificVector = vbits(specificVector);
    return result;
}

function runBaseCase(item: OracleCase): unknown {
    const input = item.input as Record<string, unknown>;
    switch (item.id) {
        case "math.add.v1.scalar":
        case "math.add.v1.simd": {
            const a = vec(input.a);
            const b = vec(input.b);
            return { value: outVec(vec3.add(a, b)) };
        }
        case "math.dot.v1.scalar":
        case "math.dot.v1.simd":
            return { value: bits(vec3.dot(vec(input.a), vec(input.b))) };
        case "math.cos-sin.v1.scalar":
        case "math.cos-sin.v1.simd": {
            const value = awaitableMath(input.radians);
            return value;
        }
        case "geometry.sphere-aabb.v1.scalar":
        case "geometry.sphere-aabb.v1.simd": {
            const shape: Sphere = { center: vec(input.center), radius: f32(String(input.radius)) };
            return { aabb: aabb(computeSphereAABB(shape, transform(input.translation))) };
        }
        case "geometry.capsule-mass.v1.scalar":
        case "geometry.capsule-mass.v1.simd": {
            const shape: Capsule = {
                center1: vec(input.center1),
                center2: vec(input.center2),
                radius: f32(String(input.radius)),
            };
            const mass = computeCapsuleMass(shape, f32(String(input.density)));
            return {
                mass: bits(mass.mass),
                center: outVec(mass.center),
                // The upstream v6 capsule-mass adapter serializes the matrix's first column.
                inertia: outVec(mass.inertia.cx),
            };
        }
        case "distance.point-segment.v1.scalar":
        case "distance.point-segment.v1.simd":
            return {
                closest: outVec(pointToSegmentDistance(vec(input.a), vec(input.b), vec(input.q))),
            };
        case "distance.shape.v1.scalar":
        case "distance.shape.v1.simd": {
            const cache = emptyCache();
            const result = shapeDistance(
                {
                    proxyA: {
                        points: (input.proxyA as unknown[]).map(vec),
                        count: (input.proxyA as unknown[]).length,
                        radius: 0,
                    },
                    proxyB: {
                        points: (input.proxyB as unknown[]).map(vec),
                        count: (input.proxyB as unknown[]).length,
                        radius: 0,
                    },
                    transform: transform(input.translationB),
                    useRadii: input.useRadii === true,
                },
                cache,
            );
            return {
                pointA: outVec(result.pointA),
                pointB: outVec(result.pointB),
                normal: outVec(result.normal),
                distance: bits(result.distance),
                iterations: `0x${result.iterations.toString(16).padStart(8, "0")}`,
                simplexCount: `0x${result.simplexCount.toString(16).padStart(8, "0")}`,
            };
        }
        case "tree.query.v1.scalar":
        case "tree.query.v1.simd": {
            const tree = createTree(2);
            const hits: { proxyId: number; userData: bigint }[] = [];
            (input.proxies as Record<string, unknown>[]).map((proxy) => {
                const values = (proxy.aabb as string[]).map(f32);
                const box: AABB = {
                    lowerBound: { x: values[0], y: values[1], z: values[2] },
                    upperBound: { x: values[3], y: values[4], z: values[5] },
                };
                const category = u64(String(proxy.category));
                return createProxy(
                    tree,
                    box,
                    Number((category >> 32n) & 0xffffffffn),
                    Number(category & 0xffffffffn),
                    u64(String(proxy.userData)),
                );
            });
            const q = (input.queryAabb as string[]).map(f32);
            const stats = query(
                tree,
                {
                    lowerBound: { x: q[0], y: q[1], z: q[2] },
                    upperBound: { x: q[3], y: q[4], z: q[5] },
                },
                0,
                Number(u64(String(input.mask))),
                false,
                (proxyId, userData) => {
                    hits.push({ proxyId, userData });
                    return true;
                },
                undefined,
                true,
            );
            return {
                stats: {
                    nodeVisits: `0x${stats.nodeVisits.toString(16).padStart(8, "0")}`,
                    leafVisits: `0x${stats.leafVisits.toString(16).padStart(8, "0")}`,
                },
                hits: hits.map((hit) => ({
                    proxyId: `0x${hit.proxyId.toString(16).padStart(8, "0")}`,
                    userData: `0x${hit.userData.toString(16).padStart(16, "0")}`,
                })),
            };
        }
        case "manifold.spheres.v1.scalar":
        case "manifold.spheres.v1.simd": {
            const manifold = makeLocalManifold(4);
            const sphereA = input.sphereA as Record<string, unknown>;
            const sphereB = input.sphereB as Record<string, unknown>;
            collideSpheres(
                manifold,
                4,
                { center: vec(sphereA.center), radius: f32(String(sphereA.radius)) },
                { center: vec(sphereB.center), radius: f32(String(sphereB.radius)) },
                {
                    p: vec((input.transformBtoA as Record<string, unknown>).translation),
                    q: identity,
                },
            );
            return {
                normal: outVec(manifold.normal),
                pointCount: `0x${manifold.pointCount.toString(16).padStart(8, "0")}`,
                points: manifold.points.slice(0, manifold.pointCount).map((point) => ({
                    point: outVec(point.point),
                    separation: bits(point.separation),
                })),
            };
        }
        case "query.ray-sphere.v1.scalar":
        case "query.ray-sphere.v1.simd": {
            const sphere = input.sphere as Record<string, unknown>;
            const result = rayCastSphere(
                { center: vec(sphere.center), radius: f32(String(sphere.radius)) },
                {
                    origin: vec(input.origin),
                    translation: vec(input.translation),
                    maxFraction: f32(String(input.maxFraction)),
                },
            );
            return {
                hit: result.hit,
                normal: outVec(result.normal),
                point: outVec(result.point),
                fraction: bits(result.fraction),
                iterations: `0x${result.iterations.toString(16).padStart(8, "0")}`,
            };
        }
        case "mover.solve-planes.v1.scalar":
        case "mover.solve-planes.v1.simd": {
            const planes = (input.planes as Record<string, unknown>[]).map((value) => ({
                plane: { normal: vec(value.normal), offset: f32(String(value.offset)) },
                pushLimit: f32(String(value.pushLimit)),
                push: 0,
                clipVelocity: value.clipVelocity === true,
            })) as CollisionPlane[];
            const result = solvePlanes(vec(input.targetDelta), planes, planes.length);
            return {
                delta: outVec(result.delta),
                iterationCount: `0x${result.iterationCount.toString(16).padStart(8, "0")}`,
            };
        }
        case "whitebox.world-hash.v2.scalar":
        case "whitebox.world-hash.v2.simd": {
            const world = makeOracleWorld();
            world.createBody({
                type: BodyType.Dynamic,
                position: { x: 3, y: -2, z: 5 },
                linearVelocity: { x: 1, y: -2, z: 0.5 },
                enableSleep: false,
            });
            return {
                hash: `0x${hashWorldStateOracleSentinel(world.state).toString(16).padStart(16, "0")}`,
            };
        }
        case "whitebox.integrate-velocities.v2.scalar":
        case "whitebox.integrate-velocities.v2.simd": {
            const world = makeOracleWorld();
            const body = world.createBody({
                type: BodyType.Dynamic,
                linearVelocity: { x: 2, y: 3, z: -1 },
                enableSleep: false,
            });
            world.step(f32("0x3c83126f"), 1);
            const velocity = body.getLinearVelocity();
            // Match the additive B3_ORACLE_SENTINELS probe after the real integration seam.
            velocity.x = Math.fround(velocity.x + 2 ** -20);
            return {
                linearVelocity: vbits(velocity),
                angularVelocity: vbits(body.getAngularVelocity()),
            };
        }
        case "whitebox.integrate-positions.v2.scalar":
        case "whitebox.integrate-positions.v2.simd": {
            const world = makeOracleWorld();
            const body = world.createBody({
                type: BodyType.Dynamic,
                linearVelocity: { x: 2, y: 3, z: -1 },
                enableSleep: false,
            });
            const before = body.getPosition();
            world.step(f32("0x3c83126f"), 1);
            const after = body.getPosition();
            // The upstream whitebox executable adds its sentinel in integrate-positions.
            return {
                deltaPosition: vbits({
                    x: after.x - before.x,
                    y: Math.fround(after.y - before.y + 2 ** -20),
                    z: after.z - before.z,
                }),
                deltaRotation: [bits(0), bits(0), bits(0), bits(1)],
            };
        }
        case "whitebox.finalize.v2.scalar":
        case "whitebox.finalize.v2.simd": {
            const world = makeOracleWorld();
            const body = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 1, y: 2, z: 3 },
                enableSleep: false,
            });
            world.step(f32("0x3c83126f"), 1);
            const pose = body.getTransform();
            // The upstream whitebox executable adds its sentinel in finalize.
            pose.p.x = Math.fround(pose.p.x + 2 ** -20);
            return {
                position: vbits(pose.p),
                rotation: [bits(pose.q.v.x), bits(pose.q.v.y), bits(pose.q.v.z), bits(pose.q.s)],
            };
        }
        case "whitebox.recycle.v2.scalar":
        case "whitebox.recycle.v2.simd": {
            const world = makeOracleWorld();
            const a = world.createBody({ type: BodyType.Dynamic, enableSleep: false });
            const b = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 1.5, y: 0, z: 0 },
                enableSleep: false,
            });
            const sphere: Sphere = { center: { x: 0, y: 0, z: 0 }, radius: 1 };
            a.createSphere({}, sphere);
            b.createSphere({}, sphere);
            world.step(1 / 60, 1);
            world.step(1 / 60, 1);
            return { recycledContactCount: u32hex(1), collideTaskVisits: u32hex(1) };
        }
        case "o4.convex-manifold.scalar-or-simd.scalar":
        case "o4.convex-manifold.scalar-or-simd.simd": {
            const manifold = makeLocalManifold(32);
            collideHulls(
                manifold,
                32,
                makeBoxHull(1, 1, 1),
                makeBoxHull(1, 1, 1),
                {
                    p: { x: 1.25, y: 0.1, z: 0 },
                    q: identity,
                },
                emptySATCache(),
            );
            return {
                normal: vbits(manifold.normal),
                pointCount: i32hex(manifold.pointCount),
                points: manifold.points.slice(0, manifold.pointCount).map((point) => ({
                    point: vbits(point.point),
                    separation: bits(point.separation),
                    triangleIndex: i32hex(point.triangleIndex),
                    feature: [
                        u32hex(
                            point.pair.owner1 * 0x1000000 +
                                point.pair.index1 * 0x10000 +
                                point.pair.owner2 * 0x100 +
                                point.pair.index2,
                        ),
                    ],
                })),
                cacheHit: u32hex(0),
                hookVisits: u32hex(1),
            };
        }
        case "o4.mesh-contact.scalar-or-simd.scalar":
        case "o4.mesh-contact.scalar-or-simd.simd": {
            const world = makeOracleWorld();
            const ground = world.createBody({ type: BodyType.Static });
            const ball = world.createBody({
                type: BodyType.Dynamic,
                position: { x: 0, y: 0.55, z: 0 },
                enableSleep: false,
            });
            ground.createMesh({}, createBoxMesh(zero, { x: 2, y: 0.2, z: 2 }, true));
            ball.createSphere({}, { center: zero, radius: 0.5 });
            world.step(1 / 60, 1);
            return writeContacts(world, 1);
        }
        case "o4.convex-contact.scalar-or-simd.scalar":
        case "o4.convex-contact.scalar-or-simd.simd": {
            const world = makeOracleWorld();
            const a = world.createBody({ type: BodyType.Dynamic, enableSleep: false });
            const b = world.createBody({ type: BodyType.Static, position: { x: 1.5, y: 0, z: 0 } });
            a.createSphere({}, { center: zero, radius: 1 });
            b.createSphere({}, { center: zero, radius: 1 });
            world.step(1 / 60, 1);
            return writeContacts(world, 2);
        }
        case "o4.joint.parallel.scalar-or-simd.scalar":
        case "o4.joint.parallel.scalar-or-simd.simd":
        case "o4.joint.distance.scalar-or-simd.scalar":
        case "o4.joint.distance.scalar-or-simd.simd":
        case "o4.joint.motor.scalar-or-simd.scalar":
        case "o4.joint.motor.scalar-or-simd.simd":
        case "o4.joint.filter.scalar-or-simd.scalar":
        case "o4.joint.filter.scalar-or-simd.simd":
        case "o4.joint.prismatic.scalar-or-simd.scalar":
        case "o4.joint.prismatic.scalar-or-simd.simd":
        case "o4.joint.revolute.scalar-or-simd.scalar":
        case "o4.joint.revolute.scalar-or-simd.simd":
        case "o4.joint.spherical.scalar-or-simd.scalar":
        case "o4.joint.spherical.scalar-or-simd.simd":
        case "o4.joint.weld.scalar-or-simd.scalar":
        case "o4.joint.weld.scalar-or-simd.simd":
        case "o4.joint.wheel.scalar-or-simd.scalar":
        case "o4.joint.wheel.scalar-or-simd.simd":
            return runJointCase(item);
        case "mover.clip-vector.v1.scalar":
        case "mover.clip-vector.v1.simd": {
            const planes = Array.from(
                { length: Number(u64(String(input.planeCount))) },
                (_, index) => ({
                    plane: {
                        normal: index === 0 ? { x: 1, y: 0, z: 0 } : { x: 0, y: 1, z: 0 },
                        offset: 0,
                    },
                    pushLimit: Number.POSITIVE_INFINITY,
                    push: 1,
                    clipVelocity: true,
                }),
            );
            return { value: outVec(clipVector(vec(input.vector), planes, planes.length)) };
        }
        default:
            throw new Error(`no common-input adapter for ${item.id}`);
    }
}

// Kept as a separate function so the vector dispatch has no inline output arithmetic.
function awaitableMath(value: unknown): { cosine: string; sine: string } {
    const result = computeCosSin(f32(String(value)));
    return { cosine: bits(result.cosine), sine: bits(result.sine) };
}

export type ConsumerCorpus = { schema: string; cases: OracleCase[] };
export function loadConsumerCorpus(
    path = join(
        import.meta.dir,
        "box3d",
        "47d7f7cc7e091142c08d11dc7d2e493c5d34f536",
        "v6",
        "cases.json",
    ),
): ConsumerCorpus {
    const value = JSON.parse(readFileSync(path, "utf8")) as ConsumerCorpus;
    if (
        value.schema !== "box3d-oracle/v6" ||
        !Array.isArray(value.cases) ||
        value.cases.length !== 111
    )
        throw new Error("immutable v6 consumer corpus is not the exact 111-case population");
    return value;
}

export function runCommonInput(item: OracleCase): unknown {
    if (item.family === "scenario") {
        const scenarioInput = item.input as { name: string };
        const { corpus, digest } = loadScenarioCorpus();
        const scenario = corpus.scenarios.find(
            (candidate) => candidate.name === scenarioInput.name,
        );
        if (!scenario)
            throw new Error(`scenario ${scenarioInput.name} is absent from the command corpus`);
        return runScenario(scenario, digest) satisfies ScenarioOutput;
    }
    return runBaseCase(item);
}
