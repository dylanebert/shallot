import { readSimCenter, readSimTransform, simFlags } from "../kernel/bodycolumns";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import { bodyType, shapeBodyId, shapeSensorIndex } from "../kernel/filtercolumns";
import { JointField, jointCapacity, jointField } from "../kernel/jointrecords";
// Debug visualization walk: resolve every shape and joint in the world to a flat set of typed draw
// callbacks the caller renders. Ported from Box3D's b3World_Draw (physics_world.c) + b3DrawJoint
// (joint.c). This is a read-only view of the simulation — it never feeds the world-state hash, so it
// sits outside the bit-exact contract (fround discipline is unnecessary here; the geometry math is
// display-only). Shape geometry stays internal; the walk hands the renderer resolved primitives.
//
// Scope: the minimal debug draw — solid shapes (hue by body state), joints (anchors + connection +
// local frames), fat-AABB bounds, and dynamic-body mass markers. The ornate per-joint-type gizmos
// (limit arcs, motor/steering indicators) and the contact/island/graph-color diagnostics are the
// heavier diagnostic tier, out of the minimal renderer's fidelity floor.

import type { BroadPhase } from "../collision/broadphase";
import { NULL_INDEX } from "../common/array";
import { hi32, lo32 } from "../common/bits";
import { SetType } from "../common/constants";
import {
    type AABB,
    type Pos,
    transformWorldPoint,
    vec3,
    type WorldTransform,
    xf,
} from "../common/math";
import { BodyType, ShapeType, type SurfaceMaterial } from "../common/types";
import { J_LOCAL_FRAME_A, J_LOCAL_FRAME_B } from "../kernel/columns";
import { readJointVec3 } from "../kernel/jointcolumns";
import { readFatAabb } from "../kernel/shapecolumns";
import * as tree from "../kernel/treecolumns";

const drawBounds = { lowerBound: vec3.zero(), upperBound: vec3.zero() };

import { getCompoundChild } from "../shapes/compound";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { Mesh } from "../shapes/mesh";
import type { Shape } from "../shapes/shape";
import { getJointConstraintForce, getJointConstraintTorque, JointType } from "../solver/joint";
import { BodyFlags, getBodySim, readBodyTransform } from "./body";
import type { WorldState } from "./world";

const BODY_TYPE_COUNT = 3;

/**
 * Named debug colors, low 24 bits packed RGB (b3HexColor). The subset the draw walk emits; matches
 * the X11/CSS values Box3D uses so a ported sample looks the same.
 */
export const DebugColor = {
    red: 0xff0000,
    orange: 0xffa500,
    yellow: 0xffff00,
    lime: 0x00ff00,
    turquoise: 0x40e0d0,
    slateGray: 0x708090,
    lightSlateGray: 0x778899,
    darkGray: 0xa9a9a9,
    tan: 0xd2b48c,
    steelBlue: 0x4682b4,
    lightSteelBlue: 0xb0c4de,
    wheat: 0xf5deb3,
    gold: 0xffd700,
    darkSeaGreen: 0x8fbc8f,
    plum: 0xdda0dd,
    yellowGreen: 0x9acd32,
    azure: 0xf0ffff,
    white: 0xffffff,
} as const;

/**
 * The callbacks and flags a debug draw supplies to {@link worldDraw} (b3DebugDraw). Every callback
 * defaults to a no-op ({@link defaultDebugDraw}); a renderer overrides the ones it needs. Colors are
 * low-24-bit packed RGB. Enable a `draw*` flag to receive that category.
 */
export type DebugDraw = {
    /** A sphere at `transform`'s position; its rotation drives the surface pattern so spin reads. */
    drawSolidSphere(transform: WorldTransform, sphere: Sphere, color: number): void;
    /** A capsule whose local +x is the long axis, centered and oriented by `transform`. */
    drawSolidCapsule(transform: WorldTransform, capsule: Capsule, color: number): void;
    /** A convex hull, oriented by `transform`. */
    drawSolidHull(transform: WorldTransform, hull: HullData, color: number): void;
    /** A static triangle mesh, oriented by `transform`. */
    drawSolidMesh(transform: WorldTransform, mesh: Mesh, color: number): void;
    /** A static height field, oriented by `transform`. */
    drawSolidHeightField(
        transform: WorldTransform,
        heightField: HeightFieldData,
        color: number,
    ): void;
    /** A line segment. */
    drawSegment(p1: Pos, p2: Pos, color: number): void;
    /** A point, size in pixels. */
    drawPoint(p: Pos, size: number, color: number): void;
    /** A coordinate frame (RGB axes). */
    drawTransform(transform: WorldTransform): void;
    /** A wireframe bounding box. */
    drawAabb(aabb: AABB, color: number): void;
    /** A positioned text label. */
    drawString(p: Pos, s: string, color: number): void;

    /** World bounds culling the walk. Shapes outside are skipped. */
    drawingBounds: AABB;
    /** Global scale for joint gizmos. */
    jointScale: number;
    /** Scale for force vectors (joint extras). */
    forceScale: number;

    /** Draw shapes. */
    drawShapes: boolean;
    /** Draw joints. */
    drawJoints: boolean;
    /** Draw joint force/torque readouts. */
    drawJointExtras: boolean;
    /** Draw shape fat AABBs. */
    drawBounds: boolean;
    /** Draw dynamic-body mass frames + values. */
    drawMass: boolean;

    /** Opaque context handed back to every callback. */
    context: unknown;
};

const noop = (): void => {};

/** @returns a debug draw with no-op callbacks, ±100 m bounds, and every category off (b3DefaultDebugDraw). */
export function defaultDebugDraw(): DebugDraw {
    const h = 100;
    return {
        drawSolidSphere: noop,
        drawSolidCapsule: noop,
        drawSolidHull: noop,
        drawSolidMesh: noop,
        drawSolidHeightField: noop,
        drawSegment: noop,
        drawPoint: noop,
        drawTransform: noop,
        drawAabb: noop,
        drawString: noop,
        drawingBounds: { lowerBound: { x: -h, y: -h, z: -h }, upperBound: { x: h, y: h, z: h } },
        jointScale: 1,
        forceScale: 1,
        drawShapes: false,
        drawJoints: false,
        drawJointExtras: false,
        drawBounds: false,
        drawMass: false,
        context: null,
    };
}

/** The debug hue for a body, by simulation state (the b3World_Draw shape-color ladder). */
function bodyColor(world: WorldState, body: number, shape: Shape): number {
    const material: SurfaceMaterial =
        shape.materials !== null ? shape.materials[0] : shape.material;
    if (material.customColor !== 0) {
        return material.customColor;
    }
    const sim = getBodySim(world, body);
    if (
        bodyType(world, bodyField(world, body, BodyField.id)) === BodyType.Dynamic &&
        bodyField(world, body, BodyField.mass) === 0
    )
        return DebugColor.red;
    if (bodyField(world, body, BodyField.setIndex) === SetType.Disabled)
        return DebugColor.slateGray;
    if (shapeSensorIndex(world, shape.id) !== NULL_INDEX) return DebugColor.wheat;
    if (bodyField(world, body, BodyField.flags) & BodyFlags.hadTimeOfImpact) return DebugColor.lime;
    if (
        simFlags(world, sim) & BodyFlags.isBullet &&
        bodyField(world, body, BodyField.setIndex) === SetType.Awake
    )
        return DebugColor.turquoise;
    if (bodyField(world, body, BodyField.flags) & BodyFlags.isSpeedCapped) return DebugColor.yellow;
    if (simFlags(world, sim) & BodyFlags.isFast) return DebugColor.orange;
    if (bodyType(world, bodyField(world, body, BodyField.id)) === BodyType.Static)
        return DebugColor.darkGray;
    if (bodyType(world, bodyField(world, body, BodyField.id)) === BodyType.Kinematic) {
        return bodyField(world, body, BodyField.setIndex) === SetType.Awake
            ? DebugColor.steelBlue
            : DebugColor.lightSteelBlue;
    }
    if (bodyField(world, body, BodyField.setIndex) === SetType.Awake) return DebugColor.tan;
    return DebugColor.lightSlateGray;
}

/** Dispatch one shape's resolved geometry to the matching solid callback, under `transform`. */
function drawSolidShape(
    draw: DebugDraw,
    shape: Shape,
    transform: WorldTransform,
    color: number,
): void {
    switch (shape.type) {
        case ShapeType.Sphere:
            draw.drawSolidSphere(transform, shape.sphere as Sphere, color);
            break;
        case ShapeType.Capsule:
            draw.drawSolidCapsule(transform, shape.capsule as Capsule, color);
            break;
        case ShapeType.Hull:
            draw.drawSolidHull(transform, shape.hull as HullData, color);
            break;
        case ShapeType.Mesh:
            draw.drawSolidMesh(transform, shape.mesh as Mesh, color);
            break;
        case ShapeType.HeightField:
            draw.drawSolidHeightField(transform, shape.heightField as HeightFieldData, color);
            break;
        case ShapeType.Compound: {
            const compound = shape.compound;
            if (compound === undefined) break;
            const childCount =
                compound.capsules.length +
                compound.hulls.length +
                compound.meshes.length +
                compound.spheres.length;
            for (let i = 0; i < childCount; ++i) {
                const child = getCompoundChild(compound, i);
                const childXf = xf.mul(transform, child.transform);
                switch (child.type) {
                    case ShapeType.Sphere:
                        draw.drawSolidSphere(childXf, child.sphere as Sphere, color);
                        break;
                    case ShapeType.Capsule:
                        draw.drawSolidCapsule(childXf, child.capsule as Capsule, color);
                        break;
                    case ShapeType.Hull:
                        draw.drawSolidHull(childXf, child.hull as HullData, color);
                        break;
                    case ShapeType.Mesh:
                        draw.drawSolidMesh(childXf, child.mesh as Mesh, color);
                        break;
                }
            }
            break;
        }
    }
}

/** Draw one joint: anchors, the A→pA→pB→B connection, and both local frames (b3DrawJoint, minimal). */
function drawJoint(draw: DebugDraw, world: WorldState, jointId: number): void {
    const bodyPoseScratch1 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
    const bodyPoseScratch2 = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };

    const joint = jointId;
    const bodyA = jointField(world, joint, JointField.bodyIdA + 3 * 0);
    const bodyB = jointField(world, joint, JointField.bodyIdA + 3 * 1);
    if (
        bodyField(world, bodyA, BodyField.setIndex) === SetType.Disabled ||
        bodyField(world, bodyB, BodyField.setIndex) === SetType.Disabled
    )
        return;

    const anchorA = readJointVec3(world, joint, J_LOCAL_FRAME_A);
    const anchorB = readJointVec3(world, joint, J_LOCAL_FRAME_B);
    const transformA = readBodyTransform(world, bodyA, bodyPoseScratch1);
    const transformB = readBodyTransform(world, bodyB, bodyPoseScratch2);
    const pA = transformWorldPoint(transformA, anchorA);
    const pB = transformWorldPoint(transformB, anchorB);

    if (jointField(world, joint, JointField.type) === JointType.Filter) {
        draw.drawSegment(pA, pB, DebugColor.gold);
        return;
    }
    if (jointField(world, joint, JointField.type) === JointType.Motor) {
        draw.drawSegment(pA, pB, DebugColor.plum);
        draw.drawPoint(pA, 8, DebugColor.yellowGreen);
        draw.drawPoint(pB, 8, DebugColor.plum);
        return;
    }

    draw.drawSegment(transformA.p, pA, DebugColor.darkSeaGreen);
    draw.drawSegment(pA, pB, DebugColor.darkSeaGreen);
    draw.drawSegment(transformB.p, pB, DebugColor.darkSeaGreen);
    draw.drawPoint(pA, 6, DebugColor.darkSeaGreen);
    draw.drawPoint(pB, 6, DebugColor.darkSeaGreen);
    draw.drawTransform({ p: pA, q: transformA.q });
    draw.drawTransform({ p: pB, q: transformB.q });

    if (draw.drawJointExtras) {
        const force = getJointConstraintForce(world, joint);
        const torque = getJointConstraintTorque(world, joint);
        const p = vec3.lerp(pA, pB, 0.5);
        draw.drawSegment(p, vec3.mulAdd(p, 0.001, force), DebugColor.azure);
        const fLen = vec3.length(force);
        const tLen = vec3.length(torque);
        draw.drawString(
            p,
            `f = ${fLen.toPrecision(4)}, t = ${tLen.toPrecision(4)}`,
            DebugColor.azure,
        );
    }
}

/**
 * Walk every shape and joint whose fat AABB overlaps `draw.drawingBounds`, resolving each to the
 * typed callbacks on `draw` (b3World_Draw). `maskBits` filters by shape category. Read-only.
 */
export function worldDraw(world: WorldState, draw: DebugDraw, maskBits: bigint): void {
    const centerScratch1 = { x: 0, y: 0, z: 0 };
    const transformScratch2 = {
        p: { x: 0, y: 0, z: 0 },
        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
    };

    (world.broadPhase as BroadPhase).store.refreshIfStale();
    const trees = (world.broadPhase as BroadPhase).trees;
    const visitedBodies = new Set<number>();
    const maskHi = hi32(maskBits);
    const maskLo = lo32(maskBits);

    for (let i = 0; i < BODY_TYPE_COUNT; ++i) {
        tree.query(trees[i], draw.drawingBounds, maskHi, maskLo, false, (_proxyId, shapeId) => {
            const centerScratch1 = {
                p: { x: 0, y: 0, z: 0 },
                q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
            };

            const shape = world.shapes[shapeId];
            visitedBodies.add(shapeBodyId(world, shape.id));

            if (draw.drawShapes) {
                const body = shapeBodyId(world, shape.id);
                const sim = getBodySim(world, body);
                const color = bodyColor(world, body, shape);
                drawSolidShape(draw, shape, readSimTransform(world, sim, centerScratch1), color);
            }
            if (draw.drawBounds) {
                draw.drawAabb(readFatAabb(world, shape.id, drawBounds), DebugColor.gold);
            }
            return true;
        });
    }

    if (draw.drawMass) {
        for (const bodyId of visitedBodies) {
            const body = bodyId;
            if (bodyType(world, bodyField(world, body, BodyField.id)) !== BodyType.Dynamic)
                continue;
            const sim = getBodySim(world, body);
            const transform: WorldTransform = {
                p: readSimCenter(world, sim, centerScratch1),
                q: readSimTransform(world, sim, transformScratch2).q,
            };
            draw.drawTransform(transform);
            const p = transformWorldPoint(transform, { x: 0.1, y: 0.1, z: 0.1 });
            draw.drawString(
                p,
                `  ${bodyField(world, body, BodyField.mass).toFixed(2)}`,
                DebugColor.white,
            );
        }
    }

    if (draw.drawJoints) {
        for (let jointId = 0; jointId < jointCapacity(world); ++jointId) {
            const joint = jointId;
            if (jointField(world, joint, JointField.setIndex) === NULL_INDEX) continue;
            drawJoint(draw, world, jointId);
        }
    }
}
