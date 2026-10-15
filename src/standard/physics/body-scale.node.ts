import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { expectWarning } from "../../../scripts/test-console";
import {
    Body,
    BodyType,
    Compounds,
    HeightFields,
    PhysicsMeshes,
    Shape,
    ShapeKind,
    SphericalJoint,
} from "../../core/physics";
import { GlobalTransform, Transform } from "../../core/transform";
import type { World } from "../../engine";
import { createApp } from "../../engine/app";
import { Character, CharacterPlugin, hashPhysics, StandardPhysicsPlugin } from ".";
import { compoundBuild, heightFieldBuild } from "./geometry-builds";
import { BodyField, bodyField } from "./kernel/bodyrecords";
import { ShapeField, shapeField } from "./kernel/shaperecords";
import { physicsBodyIndex, physicsState } from "./runtime";
import { getShapeGeometryScale, readShapeCapsule, readShapeSphere } from "./shapes/shape";
import { getMassData } from "./world/body";
import { worldCounters } from "./world/world";

setDefaultTimeout(TIMEOUT.node);
const app = () => createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
type Scale = [number, number, number];
function body(world: World, scale: Scale = [1, 1, 1]) {
    const eid = world.create();
    world.add(eid, Transform, { scale });
    world.add(eid, Body, { type: BodyType.Dynamic, gravityScale: 0 });
    return eid;
}
function shapeId(world: World, eid: number) {
    return bodyField(physicsState(world)!, physicsBodyIndex(world, eid), BodyField.headShapeId);
}
function generation(world: World, eid: number) {
    return shapeField(physicsState(world)!, shapeId(world, eid), ShapeField.generation);
}

test("body scale multiplies hull Shape.scale and collides with identical geometry and world hashes", async () => {
    const hashes: bigint[][] = [];
    for (const [s, local] of [
        [
            [1, 1, 1],
            [2, 3, 4],
        ],
        [
            [2, 3, 4],
            [1, 1, 1],
        ],
        [
            [2, 1, 2],
            [1, 3, 2],
        ],
    ] as [Scale, Scale][]) {
        const a = await app();
        try {
            const floor = body(a.world);
            a.world.storage(Body).type.set(floor, BodyType.Static);
            a.world.storage(Transform).translation.set(floor, 0, -0.5, 0);
            a.world.add(floor, Shape, { scale: [10, 1, 10] });
            const eid = body(a.world, s);
            a.world.storage(Body).gravityScale.set(eid, 1);
            a.world.storage(Transform).translation.set(eid, 0, 3, 0);
            a.world.add(eid, Shape, { scale: local });
            const h: bigint[] = [];
            for (let i = 0; i < 120; i++) {
                a.world.tick();
                h.push(hashPhysics(a.world));
            }
            hashes.push(h);
            expect(a.world.storage(GlobalTransform).translation.y.get(eid)).toBeCloseTo(1.5, 1);
            expect(getMassData(physicsState(a.world)!, physicsBodyIndex(a.world, eid)).mass).toBe(
                24000,
            );
        } finally {
            a.dispose();
        }
    }
    expect(hashes[1]).toEqual(hashes[0]);
    expect(hashes[2]).toEqual(hashes[0]);
});

test("updateBodyMass false retains mass when body scale changes geometry through a setter", async () => {
    const a = await app();
    try {
        const w = a.world,
            eid = body(w);
        w.add(eid, Shape);
        w.tick();
        const state = physicsState(w)!,
            index = physicsBodyIndex(w, eid),
            mass = getMassData(state, index).mass;
        w.storage(Shape).updateBodyMass.set(eid, 0);
        w.storage(Transform).scale.set(eid, 2, 3, 4);
        w.tick();
        expect(getMassData(state, index).mass).toBe(mass);
    } finally {
        a.dispose();
    }
});

test("a separate Shape uses its body's scale, updates hulls in place and publishes scale without physics overwriting it", async () => {
    const a = await app();
    try {
        const w = a.world,
            eid = body(w, [2, 3, 4]),
            child = w.create();
        w.add(child, Transform, { scale: [9, 9, 9] });
        w.add(child, Shape, { body: eid });
        w.tick();
        const id = shapeId(w, eid),
            gen = generation(w, eid);
        expect(getMassData(physicsState(w)!, physicsBodyIndex(w, eid)).mass).toBe(24000);
        w.storage(Transform).scale.set(eid, 3, 4, 5);
        w.tick();
        expect(shapeId(w, eid)).toBe(id);
        expect(generation(w, eid)).toBe(gen);
        expect(getMassData(physicsState(w)!, physicsBodyIndex(w, eid)).mass).toBe(60000);
        expect(
            Array.from(w.storage(GlobalTransform).scale.column.slice(eid * 4, eid * 4 + 3)),
        ).toEqual([3, 4, 5]);
        expect(Array.from(w.storage(Transform).scale.column.slice(eid * 4, eid * 4 + 3))).toEqual([
            3, 4, 5,
        ]);
    } finally {
        a.dispose();
    }
});

for (const kind of [ShapeKind.Sphere, ShapeKind.Capsule])
    test(`${kind === ShapeKind.Sphere ? "sphere" : "capsule"} scales radius and centers, refuses non-uniform scale once, retries and uses its setter`, async () => {
        const a = await app();
        try {
            const w = a.world,
                eid = body(w, [-2, 2, -2]);
            w.add(eid, Shape, {
                kind,
                radius: 0.5,
                sphereCenter: [1, 2, 3],
                capsuleCenter1: [1, 2, 3],
                capsuleCenter2: [1, 4, 3],
            });
            w.tick();
            const gen = generation(w, eid),
                id = shapeId(w, eid),
                state = physicsState(w)!;
            if (kind === ShapeKind.Sphere)
                expect(readShapeSphere(state, id)).toEqual({
                    center: { x: -2, y: 4, z: -6 },
                    radius: 1,
                });
            else
                expect(readShapeCapsule(state, id)).toEqual({
                    center1: { x: -2, y: 4, z: -6 },
                    center2: { x: -2, y: 8, z: -6 },
                    radius: 1,
                });
            w.storage(Transform).scale.set(eid, 3, 3, 3);
            w.tick();
            expect(generation(w, eid)).toBe(gen);
            expect(
                kind === ShapeKind.Sphere
                    ? readShapeSphere(state, id).radius
                    : readShapeCapsule(state, id).radius,
            ).toBe(1.5);
            w.storage(Transform).scale.set(eid, 2, 3, 4);
            expectWarning(/Shape .*requires uniform body Transform.scale magnitudes/);
            w.tick();
            w.tick();
            expect(worldCounters(state).shapeCount).toBe(0);
            w.storage(Transform).scale.set(eid, 4, 4, 4);
            w.tick();
            expect(worldCounters(state).shapeCount).toBe(1);
        } finally {
            a.dispose();
        }
    });

test("height fields build per definition and positive body scale, recreate on change, and retry refused signs", async () => {
    const a = await app();
    try {
        const w = a.world;
        const geometry = w.resource(HeightFields).register({
            name: "field",
            heights: new Float32Array([0, 1, 1, 0]),
            countX: 2,
            countZ: 2,
            scale: [2, 3, 4],
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
        });
        const eid = body(w, [2, 3, 4]);
        w.storage(Body).type.set(eid, BodyType.Static);
        w.add(eid, Shape, { kind: ShapeKind.HeightField, geometry });
        w.tick();
        const unit = heightFieldBuild(w, geometry)!,
            scaled = heightFieldBuild(w, geometry, 2, 3, 4)!;
        expect(scaled.scale).toEqual({ x: 4, y: 9, z: 16 });
        expect(heightFieldBuild(w, geometry, 2, 3, 4)).toBe(scaled);
        expect(heightFieldBuild(w, geometry, 1, 1, 1)).toBe(unit);
        const gen = generation(w, eid);
        w.storage(Transform).scale.set(eid, 3, 4, 5);
        w.tick();
        expect(generation(w, eid)).not.toBe(gen);
        w.storage(Transform).scale.set(eid, -1, 1, 1);
        expectWarning(/HeightField Shape .*requires positive body Transform.scale/);
        w.tick();
        w.tick();
        expect(worldCounters(physicsState(w)!).shapeCount).toBe(0);
        w.storage(Transform).scale.set(eid, 1, 1, 1);
        w.tick();
        expect(worldCounters(physicsState(w)!).shapeCount).toBe(1);
    } finally {
        a.dispose();
    }
});

test("compound scale multiplies child positions and geometry in child frames, retains rotations and recreates", async () => {
    const a = await app();
    try {
        const w = a.world,
            rotation: [number, number, number, number] = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
        const geometry = w.resource(Compounds).register({
            name: "compound",
            hulls: [{ hull: 0, transform: { position: [1, 2, 3], rotation } }],
            spheres: [{ center: [2, 3, 4], radius: 0.5 }],
            capsules: [{ center1: [1, 2, 3], center2: [1, 3, 3], radius: 0.25 }],
        });
        const eid = body(w, [2, 2, 2]);
        w.storage(Body).type.set(eid, BodyType.Static);
        w.add(eid, Shape, { kind: ShapeKind.Compound, geometry });
        w.tick();
        const built = compoundBuild(w, geometry, 2, 2, 2)!;
        expect(compoundBuild(w, geometry, 2, 2, 2)).toBe(built);
        expect(built.hulls[0].transform.p).toEqual({ x: 2, y: 4, z: 6 });
        expect(built.hulls[0].transform.q).toEqual({
            v: { x: 0, y: 0, z: Math.SQRT1_2 },
            s: Math.SQRT1_2,
        });
        expect(built.spheres[0].sphere).toEqual({ center: { x: 4, y: 6, z: 8 }, radius: 1 });
        expect(built.capsules[0].capsule.radius).toBe(0.5);
        const gen = generation(w, eid);
        w.storage(Transform).scale.set(eid, 3, 3, 3);
        w.tick();
        expect(generation(w, eid)).not.toBe(gen);
        w.storage(Transform).scale.set(eid, 2, 3, 4);
        expectWarning(
            /compound "compound" sphere or capsule child requires uniform body Transform.scale/,
        );
        w.tick();
        w.tick();
        expect(worldCounters(physicsState(w)!).shapeCount).toBe(0);
        w.storage(Transform).scale.set(eid, 1, 1, 1);
        w.tick();
        expect(worldCounters(physicsState(w)!).shapeCount).toBe(1);
    } finally {
        a.dispose();
    }
});

test("mesh body scale preserves signed per-axis geometry and changes through the setter", async () => {
    const a = await app();
    try {
        const w = a.world,
            geometry = w.resource(PhysicsMeshes).register({
                name: "mesh",
                vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
                indices: new Uint32Array([0, 1, 2]),
            });
        const eid = body(w, [-2, 3, 4]);
        w.add(eid, Shape, { kind: ShapeKind.Mesh, geometry, scale: [2, 1, 0.5] });
        w.tick();
        const state = physicsState(w)!,
            id = shapeId(w, eid),
            gen = generation(w, eid);
        expect(getShapeGeometryScale(state, id)).toEqual({ x: -4, y: 3, z: 2 });
        w.storage(Transform).scale.set(eid, 4, 5, 6);
        w.tick();
        expect(generation(w, eid)).toBe(gen);
        expect(getShapeGeometryScale(state, id)).toEqual({ x: 8, y: 5, z: 3 });
    } finally {
        a.dispose();
    }
});

for (const scale of [
    [0, 1, 1],
    [Number.NaN, 1, 1],
    [1, Number.POSITIVE_INFINITY, 1],
] as Scale[])
    test(`invalid body scale ${scale} refuses attachment and retries`, async () => {
        const a = await app();
        try {
            const w = a.world,
                eid = body(w, scale);
            w.add(eid, Shape);
            expectWarning(/refuses zero or non-finite body Transform.scale/);
            w.tick();
            w.tick();
            expect(worldCounters(physicsState(w)!).shapeCount).toBe(0);
            w.storage(Transform).scale.set(eid, 1, 1, 1);
            w.tick();
            expect(worldCounters(physicsState(w)!).shapeCount).toBe(1);
        } finally {
            a.dispose();
        }
    });

test("non-uniform mirrored compounds scale hull and mesh children in their own rotated frames", async () => {
    const a = await app();
    try {
        const w = a.world;
        const mesh = w.resource(PhysicsMeshes).register({
            name: "child",
            vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]),
            indices: new Uint32Array([0, 1, 2]),
        });
        const transform = {
            position: [1, 2, 3] as Scale,
            rotation: [0, 0, 1, 0] as [number, number, number, number],
        };
        const geometry = w.resource(Compounds).register({
            name: "signed",
            hulls: [{ hull: 0, transform }],
            meshes: [{ mesh, transform, scale: [2, 1, 0.5] }],
        });
        const built = compoundBuild(w, geometry, -2, 3, 4)!;
        expect(built.hulls[0].transform.p).toEqual({ x: -2, y: 6, z: 12 });
        expect(built.hulls[0].hull.aabb).toEqual({
            lowerBound: { x: -1, y: -1.5, z: -2 },
            upperBound: { x: 1, y: 1.5, z: 2 },
        });
        expect(built.meshes[0].scale).toEqual({ x: -4, y: 3, z: 2 });
        expect(built.meshes[0].transform).toEqual(built.hulls[0].transform);
        const eid = body(w, [-2, 3, 4]);
        w.storage(Body).type.set(eid, BodyType.Static);
        w.add(eid, Shape, { kind: ShapeKind.Compound, geometry });
        w.tick();
        expect(worldCounters(physicsState(w)!).shapeCount).toBe(1);
        const saved = w.snapshot(),
            hash = hashPhysics(w);
        w.storage(Transform).scale.set(eid, 2, 3, 4);
        w.tick();
        w.restore(saved);
        expect(compoundBuild(w, geometry, -2, 3, 4)).toBe(built);
        expect(hashPhysics(w)).toBe(hash);
    } finally {
        a.dispose();
    }
});

for (const kind of [ShapeKind.HeightField, ShapeKind.Compound])
    test(`scaled ${kind === ShapeKind.HeightField ? "height-field" : "compound"} cache releases deleted and replaced definitions and their builds`, async () => {
        const a = await app();
        try {
            const w = a.world;
            const refs: WeakRef<object>[] = [];
            function cycle(i: number) {
                const def =
                    kind === ShapeKind.HeightField
                        ? {
                              name: "collect",
                              heights: new Float32Array([0, 1, 1, 0]),
                              countX: 2,
                              countZ: 2,
                              scale: [1, 1, 1] as Scale,
                              globalMinimumHeight: 0,
                              globalMaximumHeight: 1,
                          }
                        : { name: "collect", spheres: [{ center: [0, 0, 0] as Scale, radius: 1 }] };
                if (kind === ShapeKind.HeightField) {
                    const registry = w.resource(HeightFields);
                    const id = registry.register(def as import("../../core/physics").HeightField);
                    const built = heightFieldBuild(w, id, 2, 2, 2)!;
                    refs.push(new WeakRef(def), new WeakRef(built));
                    if (i % 2) registry.delete("collect");
                } else {
                    const registry = w.resource(Compounds);
                    const id = registry.register(def as import("../../core/physics").Compound);
                    const built = compoundBuild(w, id, 2, 2, 2)!;
                    refs.push(new WeakRef(def), new WeakRef(built));
                    if (i % 2) registry.delete("collect");
                }
            }
            for (let i = 0; i < 10; i++) cycle(i);
            // The job that made the refs, or a stack slot the conservative scan reads, may keep a target
            // alive through one collection; yielding around each collection leaves only real retention.
            const alive = () => refs.filter((ref) => ref.deref() !== undefined).length;
            const deadline = Date.now() + 1_000;
            while (Date.now() < deadline) {
                await new Promise((resolve) => setTimeout(resolve, 0));
                Bun.gc(true);
                await new Promise((resolve) => setTimeout(resolve, 0));
                if (alive() === 0) break;
            }
            expect(alive()).toBe(0);
        } finally {
            a.dispose();
        }
    });

test("a scaled Character moves the same capsule that its body's shape attaches", async () => {
    const positions: number[] = [];
    for (const scale of [1, 2]) {
        const a = await createApp({ defaults: false, plugins: [CharacterPlugin] });
        try {
            const w = a.world,
                wall = body(w);
            w.storage(Body).type.set(wall, BodyType.Static);
            w.storage(Transform).translation.set(wall, 3, 0, 0);
            w.add(wall, Shape, { scale: [0.5, 10, 10] });
            const eid = body(w, [scale, scale, scale]);
            w.storage(Body).type.set(eid, BodyType.Kinematic);
            w.add(eid, Shape, {
                kind: ShapeKind.Capsule,
                radius: 0.6 / scale,
                capsuleCenter1: [0, -1 / scale, 0],
                capsuleCenter2: [0, 1 / scale, 0],
            });
            w.add(eid, Character);
            for (let i = 0; i < 10; i++) {
                w.storage(Character).velocity.set(eid, 30, 0, 0);
                w.tick();
            }
            positions.push(w.storage(GlobalTransform).translation.x.get(eid));
        } finally {
            a.dispose();
        }
    }
    expect(positions[1]).toBe(positions[0]);
    expect(positions[0]).toBeGreaterThan(2);
    expect(positions[0]).toBeLessThan(2.2);
});

test("joint anchors remain in unscaled body-frame meters after body scale changes", async () => {
    const a = await app();
    try {
        const w = a.world,
            anchor = body(w, [2, 3, 4]),
            bob = body(w);
        w.storage(Body).type.set(anchor, BodyType.Static);
        w.storage(Transform).translation.set(bob, 1, 0, 0);
        const joint = w.create();
        w.add(joint, SphericalJoint, { a: anchor, b: bob, localAnchorA: [1, 0, 0] });
        w.tick();
        w.storage(Transform).scale.set(anchor, 4, 5, 6);
        for (let i = 0; i < 10; i++) w.tick();
        expect(w.storage(GlobalTransform).translation.x.get(bob)).toBeCloseTo(1, 5);
    } finally {
        a.dispose();
    }
});
