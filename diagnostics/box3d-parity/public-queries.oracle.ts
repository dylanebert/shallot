// Stage-4 public differential oracle. Both sides author the same six shapes through their public
// helpers/APIs and receive the same float32 bit patterns. Both kernel buffers and public mover
// results are compared with native identification fields.
// Callback results are appended as delivered: order is deliberately never sorted.
import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body as EcsBody,
    BodyType as EcsBodyType,
    Compounds,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    PhysicsPlugin as CorePhysicsPlugin,
    Shape as EcsShape,
    ShapeKind,
} from "../../src/core/physics";
import { Time, World } from "../../src/engine";
import { quaternion, vector } from "../../src/standard/physics/kernel/math";
import { BodyType, defaultSurfaceMaterial } from "../../src/standard/physics/common/types";
import { createBoxMesh } from "../../src/standard/physics/shapes/mesh";
import { createCompound } from "../../src/standard/physics/shapes/compound";
import { createGrid } from "../../src/standard/physics/shapes/heightfield";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import {
    createCapsuleShape,
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
} from "../../src/standard/physics/shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { makeBodyId } from "../../src/standard/physics/world/body";
import { EventKind, eventCount, eventId } from "../../src/standard/physics/kernel/eventbuffers";
import { init, kernel } from "../../src/standard/physics/kernel/kernel";
import { ContactField, contactField, contactIsValid } from "../../src/standard/physics/collision/contact";
import { readContactManifolds } from "../../src/standard/physics/collision/manifoldstore";
import type { WorldState } from "../../src/standard/physics/world/world";
import {
    bodyCastRay,
    bodyCastShape,
    bodyCollideMover,
    bodyGetClosestPoint,
    bodyOverlapShape,
    worldCastMover,
    worldCastRay,
    worldCastRayClosest,
    worldCastShape,
    worldCollideMover,
    worldOverlapAABB,
    worldOverlapShape,
    type WorldCastHit,
} from "../../src/standard/physics/world/query";
import {
    castRay,
    castRayClosest,
    castShape,
    castShapeClosest,
    overlapAABB,
    overlapShape,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "../../src/standard/physics";
import { BodyField, bodyField } from "../../src/standard/physics/kernel/bodyrecords";
import { setQueryCallback } from "../../src/standard/physics/kernel/kernel";
import { queryColumns } from "../../src/standard/physics/kernel/querycolumns";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();

const f32 = Math.fround;
const bits = (value: number) => new Uint32Array(new Float32Array([value]).buffer)[0];
const hex = (value: number) => (value >>> 0).toString(16).padStart(8, "0");
const zero = { x: 0, y: 0, z: 0 };
const names = ["sphere", "capsule", "hull", "mesh", "heightfield", "compound"] as const;
const operations = [
    "world overlapAABB",
    "world overlapShape",
    "world castRay",
    "world castRayClosest",
    "world castShape",
    "world collideMover",
    "world castMover",
    "body castRay",
    "body castShape",
    "body overlapShape",
    "body getClosestPoint",
    "body collideMover",
] as const;

type Input = {
    operation: number;
    kind: number;
    words: number[];
    label: string;
};

type Pose = {
    position: { x: number; y: number; z: number };
    rotation: { v: { x: number; y: number; z: number }; s: number };
};

type ContactInput = {
    compositeKind: number;
    convexKind: number;
    words: number[];
    label: string;
};

const material = (id: bigint) => ({ ...defaultSurfaceMaterial(), userMaterialId: id });

function fixture(kind: number, pose?: Pose): { world: WorldState; body: number; shape: number } {
    const world = createTestWorld({ gravity: zero, enableSleep: false });
    const position = pose?.position ?? fixturePosition(kind);
    const rotation = pose?.rotation ?? { v: zero, s: 1 };
    const body = createTestBody(world, { type: BodyType.Static, position, rotation });
    const def = {
        userData: kind + 1,
        baseMaterial: material(0x100000000n + BigInt(kind + 1)),
    };
    const shapeDef = testShapeDef(def);
    let shape: number | null;
    if (kind === 0) {
        shape = createSphereShape(world, body, shapeDef, { center: { x: 0, y: -1, z: 0 }, radius: 1 });
    } else if (kind === 1) {
        shape = createCapsuleShape(world, body, shapeDef, {
            center1: { x: 0, y: -1.65, z: 0 },
            center2: { x: 0, y: -0.35, z: 0 },
            radius: 0.35,
        });
    } else if (kind === 2) {
        shape = createHullShape(world, body, shapeDef, makeBoxHull(1, 0.75, 0.6));
    } else if (kind === 3) {
        shape = createMeshShape(
            world,
            body,
            testShapeDef({ ...def, materials: [material(0x100000004n)] }),
            createBoxMesh({ x: 0, y: -0.75, z: 0 }, { x: 1, y: 0.75, z: 0.6 }, true),
            { x: 1, y: 1, z: 1 },
        );
    } else if (kind === 4) {
        shape = createHeightFieldShape(
            world,
            body,
            testShapeDef({ ...def, materials: [material(0x100000005n)] }),
            createGrid(4, 4, { x: 1, y: 1, z: 1 }, false),
        );
    } else {
        const compound = createCompound({
            hulls: [
                {
                    hull: makeBoxHull(1, 0.75, 0.6),
                    transform: {
                        p: { x: 0, y: -0.75, z: 0 },
                        q: { v: zero, s: 1 },
                    },
                    material: material(0x100000006n),
                },
            ],
            spheres: [
                {
                    sphere: { center: { x: -0.625, y: -0.2, z: 0 }, radius: 0.2 },
                    material: material(0x200000006n),
                },
                {
                    sphere: { center: { x: 0.625, y: -0.2, z: 0 }, radius: 0.2 },
                    material: material(0x300000006n),
                },
            ],
        });
        if (compound === null) throw new Error("compound fixture creation failed");
        shape = createCompoundShape(world, body, shapeDef, compound);
    }
    if (shape === null) throw new Error(`shape fixture ${kind} could not be created`);
    return { world, body, shape };
}

function fixturePosition(kind: number): { x: number; y: number; z: number } {
    return kind === 4 ? { x: -1.5, y: 0, z: -1.5 } : kind === 2 ? { x: 0, y: -0.75, z: 0 } : zero;
}

function pushVec(out: (number | undefined)[], value: { x: number; y: number; z: number }): void {
    out.push(bits(value.x), bits(value.y), bits(value.z));
}

function pushU64(out: (number | undefined)[], value: bigint): void {
    out.push(Number(value & 0xffffffffn), Number((value >> 32n) & 0xffffffffn));
}

function pushCast(
    out: (number | undefined)[],
    hit: {
        shape: number | null;
        userData: unknown;
        point: { x: number; y: number; z: number };
        normal: { x: number; y: number; z: number };
        fraction: number;
        userMaterialId: bigint;
        triangleIndex: number;
        childIndex?: number;
        hit: boolean;
    },
    includeChild: boolean,
): void {
    out.push(Number(hit.hit));
    if (!hit.hit) return;
    out.push(hit.userData as number);
    pushVec(out, hit.point);
    pushVec(out, hit.normal);
    out.push(bits(hit.fraction));
    pushU64(out, hit.userMaterialId);
    out.push(hit.triangleIndex);
    if (includeChild) out.push(hit.childIndex ?? 0);
}

function pushPlane(
    out: (number | undefined)[],
    plane: {
        plane: { normal: { x: number; y: number; z: number }; offset: number };
        point: { x: number; y: number; z: number };
        triangleIndex?: number;
        childIndex?: number;
        materialIndex?: number;
    },
): void {
    pushVec(out, plane.plane.normal);
    out.push(bits(plane.plane.offset));
    pushVec(out, plane.point);
    out.push(plane.triangleIndex, plane.childIndex, plane.materialIndex);
}

function runPublic(input: Input, bridgeMetadata = false): string[] {
    const values = input.words.map((word) => new Float32Array(new Uint32Array([word]).buffer)[0]);
    const pose = {
        position: { x: values[11], y: values[12], z: values[13] },
        rotation: { v: { x: values[14], y: values[15], z: values[16] }, s: values[17] },
    };
    const { world, body, shape } = fixture(input.kind, pose);
    const bodyId = makeBodyId(world, body);
    const origin = { x: values[0], y: values[1], z: values[2] };
    const translation = { x: values[3], y: values[4], z: values[5] };
    const proxy = { points: [zero], count: 1, radius: values[6] };
    const mover = {
        center1: { x: 0, y: -values[7], z: 0 },
        center2: { x: 0, y: values[7], z: 0 },
        radius: values[8],
    };
    const extent = values[9];
    const filtered = values[10] < 0;
    const maxFraction = Math.abs(values[10]);
    const filter = filtered ? { categoryBits: 0n, maskBits: 0n } : undefined;
    const out: (number | undefined)[] = [input.operation, input.kind];
    try {
        if (bridgeMetadata && (input.operation === 5 || input.operation === 11)) {
            const q = queryColumns(world);
            const k = q.prepare(origin, filter);
            if (input.operation === 11) q.placement({ p: pose.position, q: pose.rotation }, origin);
            q.mover(mover.center1, mover.center2, mover.radius);
            const records: number[][] = [];
            let count = 0;
            const previous = setQueryCallback(world.ecsState, (_kind, _id, data, length) => {
                const f = new Float32Array(k.memory.buffer, data, length * 10);
                const u = new Int32Array(k.memory.buffer, data, length * 10);
                const record = [world.shapeUserData[shape] as number];
                if (input.operation === 5) record.push(length);
                for (let i = 0; i < length; ++i) {
                    const n = i * 10;
                    for (let j = 0; j < 7; ++j) record.push(bits(f[n + j]));
                    record.push(u[n + 7], u[n + 8], u[n + 9]);
                }
                records.push(record);
                count += length;
                return 1;
            });
            try {
                if (input.operation === 5) k.worldQuery(world.worldId, 5, 1);
                else
                    k.bodyQuery(
                        world.worldId,
                        4,
                        bodyField(world, body, BodyField.headShapeId),
                        8,
                    );
            } finally {
                setQueryCallback(world.ecsState, previous);
            }
            if (input.operation === 11) out.push(count);
            for (const record of records) out.push(...record);
            if (input.operation === 5) out.push(records.length);
            return out.map((v) => (v === undefined ? "unrepresented" : hex(v)));
        }
        if (input.operation === 0) {
            let count = 0;
            const stats = worldOverlapAABB(
                world,
                {
                    lowerBound: {
                        x: origin.x - extent,
                        y: origin.y - extent,
                        z: origin.z - extent,
                    },
                    upperBound: {
                        x: origin.x + extent,
                        y: origin.y + extent,
                        z: origin.z + extent,
                    },
                },
                (shape) => {
                    out.push(world.shapeUserData[shape] as number);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 1) {
            let count = 0;
            const stats = worldOverlapShape(
                world,
                origin,
                proxy,
                (shape) => {
                    out.push(world.shapeUserData[shape] as number);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 2 || input.operation === 4) {
            let count = 0;
            const callback = (hit: WorldCastHit) => {
                out.push(hit.userData as number);
                pushVec(out, hit.point);
                pushVec(out, hit.normal);
                out.push(bits(hit.fraction));
                pushU64(out, hit.userMaterialId);
                out.push(hit.triangleIndex, hit.childIndex);
                count++;
                return hit.fraction;
            };
            const stats =
                input.operation === 2
                    ? worldCastRay(world, origin, translation, callback, filter)
                    : worldCastShape(world, origin, proxy, translation, callback, filter);
            out.push(count, stats.nodeVisits, stats.leafVisits);
        } else if (input.operation === 3) {
            pushCast(out, worldCastRayClosest(world, origin, translation, filter), true);
            const q = queryColumns(world);
            out.push(q.resultU[1], q.resultU[2]);
        } else if (input.operation === 5) {
            let count = 0;
            worldCollideMover(
                world,
                origin,
                mover,
                (shape, planes) => {
                    out.push(world.shapeUserData[shape] as number, planes.length);
                    for (const plane of planes) pushPlane(out, plane);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count);
        } else if (input.operation === 6) {
            let count = 0;
            const fraction = worldCastMover(world, origin, mover, translation, filter, (shape) => {
                out.push(world.shapeUserData[shape] as number);
                count++;
                return true;
            });
            out.push(bits(fraction), count);
        } else if (input.operation === 7) {
            pushCast(
                out,
                bodyCastRay(
                    world,
                    bodyId,
                    origin,
                    translation,
                    { p: pose.position, q: pose.rotation },
                    filter,
                    maxFraction,
                ),
                false,
            );
        } else if (input.operation === 8) {
            pushCast(
                out,
                bodyCastShape(
                    world,
                    bodyId,
                    origin,
                    proxy,
                    translation,
                    { p: pose.position, q: pose.rotation },
                    filter,
                    maxFraction,
                ),
                false,
            );
        } else if (input.operation === 9) {
            out.push(
                Number(
                    bodyOverlapShape(
                        world,
                        bodyId,
                        origin,
                        proxy,
                        { p: pose.position, q: pose.rotation },
                        filter,
                    ),
                ),
            );
        } else if (input.operation === 10) {
            const result = bodyGetClosestPoint(world, bodyId, origin);
            pushVec(out, result.point);
            out.push(bits(result.distance));
        } else {
            const planes = bodyCollideMover(
                world,
                bodyId,
                origin,
                mover,
                { p: pose.position, q: pose.rotation },
                8,
                filter,
            );
            out.push(planes.length);
            for (const entry of planes) {
                out.push(world.shapeUserData[entry.shape] as number);
                pushPlane(out, entry.plane);
            }
        }
        return out.map((v) => (v === undefined ? "unrepresented" : hex(v)));
    } finally {
        destroyTestWorld(world);
    }
}

async function runEntityQuery(input: Input): Promise<string[]> {
    const values = input.words.map((word) => new Float32Array(new Uint32Array([word]).buffer)[0]);
    const world = new World();
    for (const component of CorePhysicsPlugin.components!)
        world.registry.register(component, CorePhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    const recovery = StandardPhysicsPlugin.recovery!;
    if (recovery === "stateless") throw new Error("physics must declare recovery");
    world.registerRecovery(StandardPhysicsPlugin.name, recovery(world));
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);

    try {
        world.resource(PhysicsWorldDefinition).gravity = zero;
        world.step(Time.FIXED_DT);
        const body = world.create();
        world.add(body, EcsBody, {
            type: EcsBodyType.Static,
            position: [values[11], values[12], values[13], 0],
            rotation: [values[14], values[15], values[16], values[17]],
        });
        let kind: number;
        let geometry = 0;
        let sphereCenter: [number, number, number] = [0, 0, 0];
        let capsuleCenter1: [number, number, number] = [0, -0.5, 0];
        let capsuleCenter2: [number, number, number] = [0, 0.5, 0];
        let radius = 0.5;
        let scale: [number, number, number, number] = [1, 1, 1, 0];
        if (input.kind === 0) {
            kind = ShapeKind.Sphere;
            sphereCenter = [0, -1, 0];
            radius = 1;
        } else if (input.kind === 1) {
            kind = ShapeKind.Capsule;
            capsuleCenter1 = [0, -1.65, 0];
            capsuleCenter2 = [0, -0.35, 0];
            radius = 0.35;
        } else if (input.kind === 2) {
            kind = ShapeKind.Hull;
            scale = [2, 1.5, 1.2, 0];
        } else if (input.kind === 3) {
            kind = ShapeKind.Mesh;
            // b3CreateBoxMesh({ 0, -0.75, 0 }, { 1, 0.75, 0.6 }, true)'s definition.
            geometry = world.resource(PhysicsMeshes).register({
                name: "entity-query-mesh",
                vertices: new Float32Array([
                    1, 0, 0.6, -1, 0, 0.6, -1, -1.5, 0.6, 1, -1.5, 0.6, 1, 0, -0.6, -1, 0, -0.6, -1, -1.5,
                    -0.6, 1, -1.5, -0.6,
                ]),
                indices: new Uint32Array([
                    0, 1, 3, 1, 2, 3, 0, 4, 1, 1, 4, 5, 0, 3, 7, 4, 0, 7, 4, 7, 5, 6, 5, 7, 1, 5, 2, 6, 2, 5,
                    3, 2, 7, 6, 7, 2,
                ]),
                identifyEdges: true,
            });
            scale = [1, 1, 1, 0];
        } else if (input.kind === 4) {
            kind = ShapeKind.HeightField;
            // b3CreateGrid(4, 4, { 1, 1, 1 }, false)'s definition.
            geometry = world.resource(HeightFields).register({
                name: "entity-query-heightfield",
                heights: new Float32Array(16),
                scale: [1, 1, 1],
                countX: 4,
                countZ: 4,
                globalMinimumHeight: -256,
                globalMaximumHeight: 256,
            });
        } else {
            kind = ShapeKind.Compound;
            // b3MakeBoxHull(1, 0.75, 0.6)'s points.
            const hull = world.resource(Hulls).register({
                name: "entity-query-compound-hull",
                points: new Float32Array([
                    -1, -0.75, -0.6, 1, -0.75, -0.6, 1, 0.75, -0.6, -1, 0.75, -0.6, -1, -0.75, 0.6, 1, -0.75,
                    0.6, 1, 0.75, 0.6, -1, 0.75, 0.6,
                ]),
            });
            geometry = world.resource(Compounds).register({
                name: "entity-query-compound",
                hulls: [
                    {
                        hull,
                        transform: { position: [0, -0.75, 0], rotation: [0, 0, 0, 1] },
                        material: material(0x100000006n),
                    },
                ],
                spheres: [
                    { center: [-0.625, -0.2, 0], radius: 0.2, material: material(0x200000006n) },
                    { center: [0.625, -0.2, 0], radius: 0.2, material: material(0x300000006n) },
                ],
            });
        }
        const shape = world.create();
        world.add(shape, EcsShape, {
            body,
            kind,
            geometry,
            sphereCenter,
            capsuleCenter1,
            capsuleCenter2,
            radius,
            scale,
            userMaterialIdLow: input.kind + 1,
            userMaterialIdHigh: 1,
        });
        world.step(Time.FIXED_DT);

        const origin = { x: values[0], y: values[1], z: values[2] };
        const translation = { x: values[3], y: values[4], z: values[5] };
        const proxy = { points: [zero], count: 1, radius: values[6] };
        const extent = values[9];
        const filter = values[10] < 0 ? { categoryBits: 0n, maskBits: 0n } : undefined;
        const out: (number | undefined)[] = [input.operation, input.kind];
        const appendHit = (hit: {
            point: { x: number; y: number; z: number };
            normal: { x: number; y: number; z: number };
            fraction: number;
            userMaterialId: bigint;
            triangleIndex: number;
            childIndex: number;
        }) => {
            out.push(input.kind + 1);
            pushVec(out, hit.point);
            pushVec(out, hit.normal);
            out.push(bits(hit.fraction));
            pushU64(out, hit.userMaterialId);
            out.push(hit.triangleIndex, hit.childIndex);
        };
        if (input.operation === 0) {
            let count = 0;
            overlapAABB(
                world,
                {
                    lowerBound: { x: origin.x - extent, y: origin.y - extent, z: origin.z - extent },
                    upperBound: { x: origin.x + extent, y: origin.y + extent, z: origin.z + extent },
                },
                (shapeEid, bodyEid) => {
                    expect([shapeEid, bodyEid]).toEqual([shape, body]);
                    out.push(input.kind + 1);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count);
        } else if (input.operation === 1) {
            let count = 0;
            overlapShape(
                world,
                origin,
                proxy,
                (shapeEid, bodyEid) => {
                    expect([shapeEid, bodyEid]).toEqual([shape, body]);
                    out.push(input.kind + 1);
                    count++;
                    return true;
                },
                filter,
            );
            out.push(count);
        } else if (input.operation === 2) {
            let count = 0;
            castRay(world, origin, translation, (hit) => {
                expect([hit.shape, hit.body]).toEqual([shape, body]);
                appendHit(hit);
                count++;
                return hit.fraction;
            }, filter);
            out.push(count);
        } else if (input.operation === 13) {
            const hit = {
                shape: 0,
                body: 0,
                point: { x: 0, y: 0, z: 0 },
                normal: { x: 0, y: 0, z: 0 },
                fraction: 0,
                userMaterialId: 0n,
                triangleIndex: 0,
                childIndex: 0,
            };
            const didHit = castShapeClosest(world, origin, proxy, translation, hit, filter);
            out.push(Number(didHit));
            if (didHit) {
                expect([hit.shape, hit.body]).toEqual([shape, body]);
                appendHit(hit);
            }
        } else if (input.operation === 3) {
            const hit = {
                shape: 0,
                body: 0,
                point: { x: 0, y: 0, z: 0 },
                normal: { x: 0, y: 0, z: 0 },
                fraction: 0,
                userMaterialId: 0n,
                triangleIndex: 0,
                childIndex: 0,
            };
            const didHit = castRayClosest(world, origin, translation, hit, filter);
            out.push(Number(didHit));
            if (didHit) {
                expect([hit.shape, hit.body]).toEqual([shape, body]);
                appendHit(hit);
            }
        } else if (input.operation === 4) {
            let count = 0;
            castShape(world, origin, proxy, translation, (hit) => {
                expect([hit.shape, hit.body]).toEqual([shape, body]);
                appendHit(hit);
                count++;
                return hit.fraction;
            }, filter);
            out.push(count);
        } else {
            throw new Error(`entity query does not cover operation ${input.operation}`);
        }
        return out.map((value) => (value === undefined ? "unrepresented" : hex(value)));
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}

function contactPublic(input: ContactInput): string[] {
    const values = input.words.map((word) => new Float32Array(new Uint32Array([word]).buffer)[0]);
    const { world } = fixture(input.compositeKind);
    try {
        const body = createTestBody(world, {
            type: BodyType.Dynamic,
            position: { x: values[0], y: values[1], z: values[2] },
            rotation: { v: { x: values[3], y: values[4], z: values[5] }, s: values[6] },
        });
        const def = testShapeDef({ userData: 99, enableContactEvents: true });
        if (input.convexKind === 0) {
            createSphereShape(world, body, def, { center: zero, radius: values[7] });
        } else if (input.convexKind === 1) {
            createCapsuleShape(world, body, def, {
                center1: { x: 0, y: -values[8], z: 0 },
                center2: { x: 0, y: values[8], z: 0 },
                radius: values[7],
            });
        } else {
            createHullShape(world, body, def, makeBoxHull(values[7], values[8], values[9]));
        }
        stepTestWorld(world, f32(1 / 60), 4);
        const count = eventCount(world, EventKind.ContactBegin);
        const out = [12, input.compositeKind, input.convexKind, count];
        for (let i = 0; i < count; ++i) {
            const shapeA = eventId(world, EventKind.ContactBegin, i, 0);
            const shapeB = eventId(world, EventKind.ContactBegin, i, 2);
            const contactId = eventId(world, EventKind.ContactBegin, i, 4);
            const contact = contactId.index1 - 1;
            out.push(world.shapeUserData[shapeA.index1 - 1] as number, world.shapeUserData[shapeB.index1 - 1] as number);
            out.push(Number(contactIsValid(world, contactId)));
            out.push(bits(kernel(world.ecsState).eventFloat(world.worldId, EventKind.ContactBegin, i, 7)));
            const dataShapeA = contactField(world, contact, ContactField.shapeIdA);
            const dataShapeB = contactField(world, contact, ContactField.shapeIdB);
            out.push(world.shapeUserData[dataShapeA] as number, world.shapeUserData[dataShapeB] as number);
            const manifolds = readContactManifolds(world, contact);
            out.push(manifolds.length);
            for (const manifold of manifolds) {
                out.push(manifold.pointCount);
                pushVec(out, manifold.normal);
                out.push(bits(manifold.twistImpulse));
                pushVec(out, manifold.frictionImpulse);
                pushVec(out, manifold.rollingImpulse);
                for (const point of manifold.points) {
                    pushVec(out, point.anchorA);
                    pushVec(out, point.anchorB);
                    out.push(
                        bits(point.separation),
                        bits(point.baseSeparation),
                        bits(point.normalImpulse),
                    );
                    out.push(bits(point.totalNormalImpulse), bits(point.normalVelocity));
                    out.push(point.featureId, point.triangleIndex, Number(point.persisted));
                }
            }
        }
        return out.map(hex);
    } finally {
        destroyTestWorld(world);
    }
}

let seed = 0x47d7f7cc;
function random(): number {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 0x100000000;
}

function identityPose(kind: number): Pose {
    return { position: fixturePosition(kind), rotation: { v: zero, s: 1 } };
}

function generatedPose(kind: number): Pose {
    const base = fixturePosition(kind);
    const axis = { x: random() - 0.5, y: random() - 0.5, z: random() - 0.5 };
    const rotation = quaternion.fromAxisAngle(vector.normalize(axis), (random() - 0.5) * 0.9);
    return {
        position: {
            x: f32(base.x + (random() - 0.5) * 0.24),
            y: f32(base.y + (random() - 0.5) * 0.12),
            z: f32(base.z + (random() - 0.5) * 0.24),
        },
        rotation,
    };
}

function words(origin: number[], translation: number[], pose: Pose, maxFraction = 1): number[] {
    return [
        ...origin,
        ...translation,
        0.25,
        0.25,
        0.3,
        0.4,
        maxFraction,
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.rotation.v.x,
        pose.rotation.v.y,
        pose.rotation.v.z,
        pose.rotation.s,
    ].map((value) => bits(value));
}

const inputs: Input[] = [];
for (let operation = 0; operation < operations.length; operation++) {
    for (let kind = 0; kind < names.length; kind++) {
        const cast =
            operation === 2 ||
            operation === 3 ||
            operation === 4 ||
            operation === 6 ||
            operation === 7 ||
            operation === 8;
        const closest = operation === 10;
        const controlPose = identityPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                cast || closest ? [0, 2, 0] : [0, -0.2, 0],
                cast ? [0, -4, 0] : [0, 0, 0],
                controlPose,
            ),
            label: "deterministic hit",
        });
        const jitterX = f32((random() - 0.5) * 0.35);
        const jitterZ = f32((random() - 0.5) * 0.25);
        const seededPose = generatedPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                cast || closest ? [jitterX, 2, jitterZ] : [jitterX, -0.2, jitterZ],
                cast ? [0, -4, 0] : [0, 0, 0],
                seededPose,
            ),
            label: `seeded pose ${hex(bits(jitterX))}/${hex(bits(jitterZ))}`,
        });
        const missPose = generatedPose(kind);
        inputs.push({
            operation,
            kind,
            words: words(
                [3, cast || closest ? 2 : -0.2, 3],
                cast ? [0, -4, 0] : [0, 0, 0],
                missPose,
            ),
            label: "deterministic miss",
        });
        if (operation !== 10) {
            const filteredPose = generatedPose(kind);
            inputs.push({
                operation,
                kind,
                words: words(
                    cast ? [0, 2, 0] : [0, -0.2, 0],
                    cast ? [0, -4, 0] : [0, 0, 0],
                    filteredPose,
                    -1,
                ),
                label: "zero query filter",
            });
        }
    }
}
for (const operation of [2, 3, 4, 7, 8]) {
    inputs.push({
        operation,
        kind: 5,
        words: words([1, 2, 0], [0, -4, 0], identityPose(5)),
        label: "strict compound box clip",
    });
}

function contactWords(pose: Pose, dimensions: number[]): number[] {
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.rotation.v.x,
        pose.rotation.v.y,
        pose.rotation.v.z,
        pose.rotation.s,
        ...dimensions,
    ].map(bits);
}

const contactInputs: ContactInput[] = [];
for (const compositeKind of [3, 4, 5]) {
    const centered: Array<[number, number[]]> = [
        [0, [0.5, 0, 0]],
        [1, [0.3, 0.35, 0]],
        [2, [0.38, 0.45, 0.32]],
    ];
    for (const [convexKind, dimensions] of centered) {
        contactInputs.push({
            compositeKind,
            convexKind,
            words: contactWords(
                { position: { x: 0, y: 0.35, z: 0 }, rotation: { v: zero, s: 1 } },
                dimensions,
            ),
            label: `centered ${names[convexKind]}`,
        });
    }
    for (let sample = 0; sample < 5; sample++) {
        const convexKind = sample % 3;
        const pose = generatedPose(0);
        pose.position.x = f32((random() - 0.5) * 0.3);
        pose.position.y = f32(0.05 + random() * 0.05);
        pose.position.z = f32((random() - 0.5) * 0.3);
        if (compositeKind === 4 && convexKind === 1) pose.position.y = f32(0.35);
        const radius = f32(0.35 + random() * 0.2);
        let dimensions =
            convexKind === 0
                ? [radius, 0, 0]
                : convexKind === 1
                  ? [f32(radius * 0.65), f32(0.2 + random() * 0.35), 0]
                  : [radius, f32(0.28 + random() * 0.25), f32(0.24 + random() * 0.24)];
        if (compositeKind === 4 && convexKind === 1) dimensions = [f32(0.65), f32(0.6), 0];
        contactInputs.push({
            compositeKind,
            convexKind,
            words: contactWords(pose, dimensions),
            label: `seeded ${sample} ${names[convexKind]}`,
        });
    }
}

const closestInputs = inputs
    .filter((input) => input.operation === 4 && input.label === "deterministic hit")
    .map((input) => ({ ...input, operation: 13, label: "deterministic castShapeClosest" }));
const nativeInput = [
    ...inputs.map((input) => `${input.operation} ${input.kind} ${input.words.map(hex).join(" ")}`),
    ...closestInputs.map((input) => `${input.operation} ${input.kind} ${input.words.map(hex).join(" ")}`),
    ...contactInputs.map(
        (input) =>
            `12 ${input.compositeKind} ${input.convexKind} ${input.words.map(hex).join(" ")}`,
    ),
].join("\n");
const nativeRows = nativeSseOutput("public-queries.c", nativeInput, [
    ...inputs.map((input) => `${operations[input.operation]}:${names[input.kind]}:${input.label}`),
    ...closestInputs.map((input) => `world castShapeClosest:${names[input.kind]}:${input.label}`),
    ...contactInputs.map((input) => `contact:${names[input.compositeKind]}:${input.label}`),
])
    .trim()
    .split("\n")
    .map((line) =>
        line
            .trim()
            .split(/\s+/)
            .map((v) => v.padStart(8, "0")),
    );
if (nativeRows.length !== inputs.length + closestInputs.length + contactInputs.length)
    throw new Error(
        `native returned ${nativeRows.length} rows for ${inputs.length + closestInputs.length + contactInputs.length} cases`,
    );

const actualRows = inputs.map((input) => runPublic(input, true));
for (let operation = 0; operation < operations.length; operation++) {
    test(`${operations[operation]}: all six public shape kinds equal Box3D SSE2 bits`, () => {
        for (let i = 0; i < inputs.length; i++) {
            const input = inputs[i];
            if (input.operation !== operation) continue;
            if (JSON.stringify(actualRows[i]) !== JSON.stringify(nativeRows[i])) {
                console.info(
                    JSON.stringify({
                        operation: operations[operation],
                        shape: names[input.kind],
                        case: input.label,
                        input: input.words.map(hex),
                        actual: actualRows[i],
                        native: nativeRows[i],
                    }),
                );
            }
            expect(actualRows[i]).toEqual(nativeRows[i]);
        }
    });
}

test("ECS World queries equal native Box3D for all six geometry kinds", async () => {
    for (let i = 0; i < inputs.length; i++) {
        const input = inputs[i];
        if (input.operation > 4 || input.label !== "deterministic hit") continue;
        const actual = await runEntityQuery(input);
        expect(actual).toEqual(nativeRows[i].slice(0, -2));
    }
});

test("ECS castShapeClosest equals native Box3D for all six geometry kinds", async () => {
    for (let i = 0; i < closestInputs.length; i++) {
        const input = closestInputs[i];
        const actual = await runEntityQuery(input);
        expect(actual).toEqual(nativeRows[inputs.length + i]);
    }
});

test("published mover fields equal native; identification is also compared through kernel buffers", () => {
    for (let i = 0; i < inputs.length; ++i) {
        if (inputs[i].operation !== 5 && inputs[i].operation !== 11) continue;
        const actual = runPublic(inputs[i]);
        expect(actual.length).toBe(nativeRows[i].length);
        for (let j = 0; j < actual.length; ++j) {
            // Both kernel buffers and public PlaneResult identification fields are compared
            // with native results.
            if (actual[j] !== "unrepresented") expect(actual[j]).toBe(nativeRows[i][j]);
        }
    }
});

for (const [operation, entry] of [
    [5, "worldCollideMover"],
    [11, "bodyCollideMover"],
] as const) {
    test(`${entry}: public mover planes preserve triangleIndex, childIndex and materialIndex`, () => {
        for (let i = 0; i < inputs.length; ++i) {
            if (inputs[i].operation !== operation) continue;
            expect(runPublic(inputs[i])).toEqual(nativeRows[i]);
        }
    });
}

for (const kind of [3, 4, 5]) {
    test(`${names[kind]} contacts: generated sphere/capsule/hull public manifolds equal Box3D SSE2 bits`, () => {
        const rows = contactInputs
            .map((input, i) => ({ input, native: nativeRows[inputs.length + closestInputs.length + i] }))
            .filter(
                (row) =>
                    row.input.compositeKind === kind &&
                    !(kind === 3 && row.input.label === "seeded 0 sphere"),
            );
        const mismatches: string[] = [];
        const untouched: string[] = [];
        for (const { input, native } of rows) {
            const actual = contactPublic(input);
            if (Number.parseInt(actual[3], 16) === 0) untouched.push(input.label);
            if (JSON.stringify(actual) !== JSON.stringify(native)) {
                mismatches.push(input.label);
                console.info(
                    JSON.stringify({
                        shape: names[kind],
                        convex: names[input.convexKind],
                        case: input.label,
                        input: input.words.map(hex),
                        actual,
                        native,
                    }),
                );
            }
        }
        expect(untouched).toEqual([]);
        expect(mismatches).toEqual([]);
    });
}

test("triangle_manifold.rs:578-579 vs math_functions.c:554-555: seeded mesh sphere contact preserves native barycentrics", () => {
    const i = contactInputs.findIndex(
        (input) => input.compositeKind === 3 && input.label === "seeded 0 sphere",
    );
    const actual = contactPublic(contactInputs[i]);
    expect(Number.parseInt(actual[3], 16)).toBeGreaterThan(0);
    expect(actual).toEqual(nativeRows[inputs.length + closestInputs.length + i]);
});
