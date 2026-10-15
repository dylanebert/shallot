import { expect, test } from "bun:test";
import {
    Body,
    BodyType,
    Compounds,
    DistanceJoint,
    HeightFields,
    Hulls,
    ParallelJoint,
    PhysicsMeshes,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    ShapeKind,
    SphericalJoint,
    UNIT_CUBE_ID,
    WeldJoint,
    WheelJoint,
} from "../../../core/physics";
import { createApp, Time } from "../../../engine";
import { DEFAULT_MASK_BITS } from "../common/constants";
import type { AABB, WorldTransform } from "../common/records";
import { drawPhysics, PhysicsWorldDefinition, StandardPhysicsPlugin } from "../index";
import { kernel } from "../kernel/kernel";
import { quaternion } from "../kernel/math";
import { shapeHullGeometry } from "../marshal-shape";
import { createHull } from "../shapes/build-hull";
import type { Sphere } from "../shapes/geometry";
import { createHeightField } from "../shapes/heightfield";
import { createGridMesh } from "../shapes/mesh";
import { destroyTestWorld } from "../testing";
import { defaultDebugDraw, worldDraw } from "./draw";
import { drawScene, reactionScene } from "./draw-scene";

test("drawPhysics preserves ECS-authored sphere geometry and placement", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        const body = world.create();
        world.add(body, Body, { type: BodyType.Dynamic, position: [1, 2, 3, 0] });
        const shape = world.create();
        world.add(shape, Shape, {
            body,
            kind: ShapeKind.Sphere,
            sphereCenter: [0.2, 0, 0],
            radius: 0.5,
            density: 2,
        });
        world.step(Time.FIXED_DT);

        const draw = defaultDebugDraw();
        draw.drawShapes = true;
        const spheres: unknown[] = [];
        draw.drawSolidSphere = (transform, sphere) =>
            spheres.push([structuredClone(transform), structuredClone(sphere)]);
        drawPhysics(world, draw);
        expect(spheres).toEqual([
            [
                { p: { x: 1, y: 2, z: 3 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
                { center: { x: Math.fround(0.2), y: 0, z: 0 }, radius: 0.5 },
            ],
        ]);
    } finally {
        app.dispose();
    }
});

test("drawPhysics preserves ECS-authored primitive, frame, bounds and mass payloads", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
        // b3CreateGridMesh(1, 1, 2, 0, true)'s definition.
        const meshExpected = createGridMesh(1, 1, 2, 0, true);
        const meshId = world.resource(PhysicsMeshes).register({
            name: "draw-payload-mesh",
            vertices: new Float32Array([-1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0, 1]),
            indices: new Uint32Array([0, 1, 3, 3, 2, 0]),
            useMedianSplit: true,
            identifyEdges: true,
        });
        const heightExpected = createHeightField({
            heights: [0, 0.25, 0.5, 1],
            materialIndices: null,
            scale: { x: 1, y: 2, z: 3 },
            countX: 2,
            countZ: 2,
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
            clockwiseWinding: false,
        });
        const heightId = world.resource(HeightFields).register({
            name: "draw-payload-height-field",
            heights: new Float32Array([0, 0.25, 0.5, 1]),
            scale: [1, 2, 3],
            countX: 2,
            countZ: 2,
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
        });
        const corners = [-1, 1].flatMap((x) =>
            [-1, 1].flatMap((y) => [-1, 1].map((z) => ({ x: 0.25 * x, y: 0.5 * y, z: 0.75 * z }))),
        );
        const childHull = world.resource(Hulls).register({
            name: "draw-payload-child",
            points: new Float32Array(corners.flatMap(({ x, y, z }) => [x, y, z])),
        });
        const compoundId = world.resource(Compounds).register({
            name: "draw-payload-compound",
            hulls: [
                { hull: childHull, transform: { position: [0.5, 1, 1.5], rotation: [0, 0, 0, 1] } },
            ],
        });
        const childExpected = createHull(corners, 128)!;
        type BodyMotionType = (typeof BodyType)[keyof typeof BodyType];
        const body = (
            x: number,
            type: BodyMotionType = BodyType.Static,
            rotation: [number, number, number, number] = [0, 0, 0, 1],
        ) => {
            const eid = world.create();
            world.add(eid, Body, { type, position: [x, 0, 0, 0], rotation });
            return eid;
        };
        const shape = (bodyEid: number, definition: Record<string, unknown>) => {
            const eid = world.create();
            world.add(eid, Shape, { body: bodyEid, ...definition } as never);
            return eid;
        };
        const capsuleBody = body(-8);
        shape(capsuleBody, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [-1, 0, 0],
            capsuleCenter2: [1, 0, 0],
            radius: 0.3,
        });
        const hullBody = body(-4);
        const hullShape = shape(hullBody, {
            kind: ShapeKind.Hull,
            geometry: UNIT_CUBE_ID,
            scale: [0.5, 0.7, 0.9, 0],
        });
        const meshBody = body(0);
        shape(meshBody, { kind: ShapeKind.Mesh, geometry: meshId, scale: [1, 1, 1, 0] });
        const heightBody = body(4);
        shape(heightBody, { kind: ShapeKind.HeightField, geometry: heightId });
        const compoundBody = body(8);
        shape(compoundBody, { kind: ShapeKind.Compound, geometry: compoundId });
        const massBody = body(12, BodyType.Dynamic);
        shape(massBody, {
            kind: ShapeKind.Sphere,
            sphereCenter: [0.2, 0, 0],
            radius: 0.5,
            density: 2,
        });
        const frameA = body(16);
        const frameB = body(18, BodyType.Kinematic, [0, 0, Math.SQRT1_2, Math.SQRT1_2]);
        const joint = world.create();
        world.add(joint, DistanceJoint, {
            a: frameA,
            b: frameB,
            localAnchorA: [1, 2, 3, 0],
        });
        world.step(Time.FIXED_DT);

        const draw = defaultDebugDraw();
        draw.drawShapes = draw.drawBounds = draw.drawMass = draw.drawJoints = true;
        const capsules: unknown[] = [];
        const hulls: unknown[][] = [];
        const meshes: unknown[] = [];
        const heightFields: unknown[] = [];
        const spheres: unknown[] = [];
        const segments: unknown[] = [];
        const points: unknown[] = [];
        const transforms: unknown[] = [];
        const bounds: unknown[] = [];
        const strings: unknown[] = [];
        draw.drawSolidCapsule = (...args) => capsules.push(structuredClone(args));
        draw.drawSolidHull = (...args) => hulls.push(structuredClone(args));
        draw.drawSolidMesh = (...args) => meshes.push(structuredClone(args));
        draw.drawSolidHeightField = (...args) => heightFields.push(structuredClone(args));
        draw.drawSolidSphere = (...args) => spheres.push(structuredClone(args));
        draw.drawSegment = (...args) => segments.push(structuredClone(args));
        draw.drawPoint = (...args) => points.push(structuredClone(args));
        draw.drawTransform = (...args) => transforms.push(structuredClone(args));
        draw.drawAabb = (...args) => bounds.push(structuredClone(args));
        draw.drawString = (...args) => strings.push(structuredClone(args));
        drawPhysics(world, draw);

        const pose = (x: number, y = 0, z = 0, q = { v: { x: 0, y: 0, z: 0 }, s: 1 }) => ({
            p: { x, y, z },
            q,
        });
        expect(capsules).toEqual([
            [
                pose(-8),
                {
                    center1: { x: -1, y: 0, z: 0 },
                    center2: { x: 1, y: 0, z: 0 },
                    radius: Math.fround(0.3),
                },
                0xa9a9a9,
            ],
        ]);
        expect(hulls).toHaveLength(2);
        const standaloneHull = hulls.find(
            ([transform]) => (transform as { p: { x: number } }).p.x === -4,
        );
        expect(standaloneHull?.[0]).toEqual(pose(-4));
        expect(standaloneHull?.[1]).toEqual(shapeHullGeometry(world, hullShape));
        expect((standaloneHull?.[1] as { points: unknown[] } | undefined)?.points).toEqual([
            { x: -0.5, y: -0.699999988079071, z: 0.8999999761581421 },
            { x: 0.5, y: 0.699999988079071, z: -0.8999999761581421 },
            { x: 0.5, y: -0.699999988079071, z: -0.8999999761581421 },
            { x: 0.5, y: -0.699999988079071, z: 0.8999999761581421 },
            { x: 0.5, y: 0.699999988079071, z: 0.8999999761581421 },
            { x: -0.5, y: 0.699999988079071, z: 0.8999999761581421 },
            { x: -0.5, y: 0.699999988079071, z: -0.8999999761581421 },
            { x: -0.5, y: -0.699999988079071, z: -0.8999999761581421 },
        ]);
        const compoundChild = hulls.find(
            ([transform]) => (transform as { p: { x: number } }).p.x === 8.5,
        );
        expect(compoundChild?.[0]).toEqual(pose(8.5, 1, 1.5));
        expect((compoundChild?.[1] as { points: unknown[] } | undefined)?.points).toEqual(
            childExpected.points,
        );
        expect(compoundChild?.[1]).toEqual(childExpected);
        expect(meshes).toEqual([
            [pose(0), { scale: { x: 1, y: 1, z: 1 }, data: meshExpected }, 0xa9a9a9],
        ]);
        expect(heightFields).toEqual([[pose(4), heightExpected, 0xa9a9a9]]);
        expect(spheres).toEqual([
            [pose(12), { center: { x: Math.fround(0.2), y: 0, z: 0 }, radius: 0.5 }, 0xd2b48c],
        ]);
        expect(segments).toEqual([
            [{ x: 16, y: 0, z: 0 }, { x: 17, y: 2, z: 3 }, 0x8fbc8f],
            [{ x: 17, y: 2, z: 3 }, { x: 18, y: 0, z: 0 }, 0x8fbc8f],
            [{ x: 18, y: 0, z: 0 }, { x: 18, y: 0, z: 0 }, 0x8fbc8f],
        ]);
        expect(points).toEqual([
            [{ x: 17, y: 2, z: 3 }, 6, 0x8fbc8f],
            [{ x: 18, y: 0, z: 0 }, 6, 0x8fbc8f],
        ]);
        expect(transforms).toEqual([
            [pose(12.199999809265137)],
            [pose(17, 2, 3)],
            [
                pose(18, 0, 0, {
                    v: { x: 0, y: 0, z: 0.7071068286895752 },
                    s: 0.7071068286895752,
                }),
            ],
        ]);
        expect(bounds).toContainEqual([
            {
                lowerBound: {
                    x: -9.340001106262207,
                    y: -0.3400000333786011,
                    z: -0.3400000333786011,
                },
                upperBound: { x: -6.659999847412109, y: 0.3400000333786011, z: 0.3400000333786011 },
            },
            0xffd700,
        ]);
        expect(strings).toEqual([
            [
                { x: 12.300000190734863, y: 0.10000000149011612, z: 0.10000000149011612 },
                "  1.05",
                0xffffff,
            ],
        ]);
    } finally {
        app.dispose();
    }
});

test("drawPhysics reports the deleted golden's exact stepped ECS joint reactions", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const { world } = app;
        const joints = [
            DistanceJoint,
            ParallelJoint,
            PrismaticJoint,
            RevoluteJoint,
            SphericalJoint,
            WeldJoint,
            WheelJoint,
        ];
        const rotation = quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, 0.4);
        for (let i = 0; i < joints.length; i++) {
            const anchor = world.create();
            world.add(anchor, Body, { position: [i * 5 - 15, 0, 0, 0] });
            const bob = world.create();
            world.add(bob, Body, {
                type: BodyType.Dynamic,
                position: [i * 5 - 14, 2, 0, 0],
                rotation: [rotation.v.x, rotation.v.y, rotation.v.z, rotation.s],
                angularVelocity: [1, 2, 3, 0],
            });
            world.add(bob, Shape, {
                kind: ShapeKind.Sphere,
                sphereCenter: [0.2, 0, 0],
                radius: 0.5,
                density: 2,
            });
            const joint = world.create();
            world.add(joint, joints[i], { a: anchor, b: bob } as never);
        }
        for (let step = 0; step < 4; step++) world.step(Time.FIXED_DT);

        const draw = defaultDebugDraw();
        draw.drawJoints = draw.drawJointExtras = true;
        const reactions: unknown[] = [];
        draw.drawString = (position, text) => {
            if (text.startsWith("f = ")) reactions.push([structuredClone(position), text]);
        };
        drawPhysics(world, draw);
        expect(reactions).toEqual([
            [
                { x: -14.745168685913086, y: 0.4374263882637024, z: -0.02148989960551262 },
                "f = 23.39, t = 0.000",
            ],
            [
                { x: -9.500715255737305, y: 0.9885704517364502, z: -0.007636573165655136 },
                "f = 0.000, t = 1.992",
            ],
            [
                { x: -4.512058258056641, y: 0.011173244565725327, z: -0.00030215305741876364 },
                "f = 62.79, t = 40.14",
            ],
            [
                { x: 0.004820622503757477, y: 0.0017359405755996704, z: -0.00006760167889297009 },
                "f = 488.1, t = 2.057",
            ],
            [
                { x: 5.005710601806641, y: 0.003071241080760956, z: -0.002261389046907425 },
                "f = 495.5, t = 0.000",
            ],
            [
                { x: 10.004926681518555, y: 0.010855603963136673, z: 0.000043295847717672586 },
                "f = 57.15, t = 35.96",
            ],
            [
                { x: 15.568255424499512, y: 0.0012533068656921387, z: -0.0003257670905441046 },
                "f = 409.1, t = 0.000",
            ],
        ]);
    } finally {
        app.dispose();
    }
});

// The direct-handle drawing checks below remain internal solver/parity tests; ECS claims above use drawPhysics.
test("debug draw visits each authored shape kind and drawing primitive", () => {
    const { world, draw } = drawScene();
    const calls: unknown[] = [];
    for (const key of [
        "drawSolidSphere",
        "drawSolidCapsule",
        "drawSolidHull",
        "drawSolidMesh",
        "drawSolidHeightField",
        "drawSegment",
        "drawPoint",
        "drawTransform",
        "drawAabb",
        "drawString",
    ] as const) {
        draw[key] = (...args: unknown[]) => {
            calls.push(structuredClone([key, ...args]));
        };
    }
    worldDraw(world, draw, DEFAULT_MASK_BITS);
    const kinds = calls.map((call) => (call as unknown[])[0]);
    expect(kinds).toEqual(
        expect.arrayContaining([
            "drawSolidSphere",
            "drawSolidCapsule",
            "drawSolidHull",
            "drawSolidMesh",
            "drawSolidHeightField",
            "drawSegment",
            "drawPoint",
            "drawTransform",
            "drawAabb",
            "drawString",
        ]),
    );
    destroyTestWorld(world);
});

test("debug draw reports nonzero stepped joint force and torque", () => {
    const { world, draw } = reactionScene();
    const calls: unknown[][] = [];
    for (const key of [
        "drawSolidSphere",
        "drawSolidCapsule",
        "drawSolidHull",
        "drawSolidMesh",
        "drawSolidHeightField",
        "drawSegment",
        "drawPoint",
        "drawTransform",
        "drawAabb",
        "drawString",
    ] as const) {
        draw[key] = (...args: unknown[]) => {
            calls.push(structuredClone([key, ...args]));
        };
    }
    try {
        worldDraw(world, draw, DEFAULT_MASK_BITS);
        const labels = calls.filter(
            (call) => call[0] === "drawString" && String(call[2]).startsWith("f = "),
        );
        expect(labels.length).toBe(7);
        expect(labels.some((call) => Number(/f = ([^,]+)/.exec(String(call[2]))![1]) > 0)).toBe(
            true,
        );
        expect(labels.some((call) => Number(/t = (.+)/.exec(String(call[2]))![1]) > 0)).toBe(true);
    } finally {
        destroyTestWorld(world);
    }
});

test("a shapes-only nested draw from the first callback preserves the full outer stream", () => {
    const { world, draw } = drawScene();
    const nested = defaultDebugDraw();
    nested.drawShapes = true;
    const baseline: unknown[] = [];
    const calls: unknown[] = [];
    for (const key of [
        "drawSolidSphere",
        "drawSolidCapsule",
        "drawSolidHull",
        "drawSolidMesh",
        "drawSolidHeightField",
        "drawSegment",
        "drawPoint",
        "drawTransform",
        "drawAabb",
        "drawString",
    ] as const) {
        draw[key] = (...args: unknown[]) => baseline.push(structuredClone([key, ...args]));
    }
    worldDraw(world, draw, DEFAULT_MASK_BITS);
    let entered = false;
    for (const key of [
        "drawSolidSphere",
        "drawSolidCapsule",
        "drawSolidHull",
        "drawSolidMesh",
        "drawSolidHeightField",
        "drawSegment",
        "drawPoint",
        "drawTransform",
        "drawAabb",
        "drawString",
    ] as const) {
        draw[key] = (...args: unknown[]) => {
            calls.push(structuredClone([key, ...args]));
            if (!entered) {
                entered = true;
                worldDraw(world, nested, DEFAULT_MASK_BITS);
            }
        };
    }
    try {
        worldDraw(world, draw, DEFAULT_MASK_BITS);
        expect(calls).toEqual(baseline);
    } finally {
        destroyTestWorld(world);
    }
});

test("a throwing nested callback releases both pending streams", () => {
    const { world, draw } = drawScene();
    const nested = defaultDebugDraw();
    nested.drawShapes = true;
    const error = new Error("nested draw callback");
    nested.drawSolidHull = () => {
        throw error;
    };
    draw.drawSolidHull = () => {
        worldDraw(world, nested, DEFAULT_MASK_BITS);
    };
    const k = kernel(world.ecsState);
    try {
        expect(() => worldDraw(world, draw, DEFAULT_MASK_BITS)).toThrow(error);
        expect(k.worldDrawLen()).toBe(0);
        expect(() => worldDraw(world, defaultDebugDraw(), DEFAULT_MASK_BITS)).not.toThrow();
        expect(k.worldDrawLen()).toBe(0);
    } finally {
        destroyTestWorld(world);
    }
});

test("draw snapshots flags and bounds and hands out fresh callback values", () => {
    const { world, draw } = drawScene();
    const bounds: AABB[] = [];
    const frames: WorldTransform[] = [];
    const spheres: Sphere[] = [];
    draw.drawSolidSphere = (t, s) => {
        frames.push(t);
        spheres.push(s);
        draw.drawBounds = false;
        draw.drawingBounds.lowerBound.x = 1000;
    };
    draw.drawAabb = (aabb) => {
        bounds.push(aabb);
    };
    worldDraw(world, draw, DEFAULT_MASK_BITS);
    expect(bounds.length).toBe(6);
    expect(new Set(bounds).size).toBe(bounds.length);
    const center = spheres[0].center.x;
    spheres[0].center.x = 500;
    frames[0].p.x = 500;
    draw.drawBounds = true;
    draw.drawingBounds.lowerBound.x = -100;
    worldDraw(world, draw, DEFAULT_MASK_BITS);
    expect(spheres[1].center.x).toBe(center);
    expect(frames[1].p.x).toBe(1);
    destroyTestWorld(world);
});
