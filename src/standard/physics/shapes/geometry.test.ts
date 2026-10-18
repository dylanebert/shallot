import { expect, test } from "bun:test";
import type { Transform, Vec3 } from "../common/records";
import { transform } from "../kernel/math";

const f32 = Math.fround;

import { BodyType } from "../common/types";
import { ShapeField, shapeField } from "../kernel/shaperecords";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "../testing";
import { destroyBody } from "../world/body";
import {
    type Capsule,
    computeCapsuleAABB,
    computeSphereAABB,
    type MassData,
    roundCapsule,
    roundSphere,
    type Sphere,
} from "./geometry";
import gold from "./geometry.gold.json";
import {
    computeShapeAABBOut,
    computeShapeMass,
    createCapsuleShape,
    createSphereShape,
    destroyShape,
    shapeIsValid,
} from "./shape";

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

function shapeEntityId(world: ReturnType<typeof createTestWorld>, shape: number) {
    return {
        index1: shape + 1,
        world0: world.worldId,
        generation: shapeField(world, shape, ShapeField.generation),
    };
}

type MassGold = { name: string; mass: string; center: string[]; inertia: string[] };

function assertMass(m: MassData, g: MassGold) {
    bitEqual(m.mass, g.mass, `${g.name} mass`);
    bitEqual(m.center.x, g.center[0], `${g.name} center.x`);
    bitEqual(m.center.y, g.center[1], `${g.name} center.y`);
    bitEqual(m.center.z, g.center[2], `${g.name} center.z`);
    const inertia = [
        m.inertia.cx.x,
        m.inertia.cx.y,
        m.inertia.cx.z,
        m.inertia.cy.x,
        m.inertia.cy.y,
        m.inertia.cy.z,
        m.inertia.cz.x,
        m.inertia.cz.y,
        m.inertia.cz.z,
    ];
    for (let i = 0; i < 9; ++i) bitEqual(inertia[i], g.inertia[i], `${g.name} inertia[${i}]`);
}

const sphereGold = (name: string) => gold.spheres.find((s) => s.name === name) as MassGold;
const capsuleGold = (name: string) => gold.capsules.find((c) => c.name === name) as MassGold;

/** Keep the authoring gold on its existing builders while driving shape lifetime through production. */
function exercisePublicShapeLifecycle(): void {
    const world = createTestWorld();
    const body = createTestBody(world, { type: BodyType.Dynamic });
    const stale = createSphereShape(world, body, testShapeDef(), {
        center: v(0, 0, 0),
        radius: 0.5,
    })!;
    const capsule = createCapsuleShape(world, body, testShapeDef(), {
        center1: v(0, -1, 0),
        center2: v(0, 1, 0),
        radius: 0.25,
    })!;
    expect(shapeIsValid(world, shapeEntityId(world, stale)), "shape slot is kernel-live").toBe(
        true,
    );
    expect(shapeIsValid(world, shapeEntityId(world, capsule)), "shape slot is kernel-live").toBe(
        true,
    );
    const staleId = shapeEntityId(world, stale);
    destroyShape(world, stale, false);
    expect(shapeIsValid(world, staleId), "destroyed shape id is stale").toBe(false);
    const reused = createSphereShape(world, body, testShapeDef(), {
        center: v(0, 0, 0),
        radius: 0.5,
    })!;
    expect(reused).toBe(stale);
    expect(shapeIsValid(world, staleId), "reused shape slot rejects the old generation").toBe(
        false,
    );
    expect(
        shapeIsValid(world, shapeEntityId(world, reused)),
        "reused shape slot is kernel-live",
    ).toBe(true);
    expect(shapeEntityId(world, reused).generation).not.toBe(staleId.generation);
    destroyShape(world, capsule, false);
    destroyShape(world, reused, false);
    destroyBody(world, body);
    destroyTestWorld(world);
}

test("sphere and capsule shape mass equals the Box3D C reference's f32 bits, including the ragdoll bone capsule whose sphere inertia needs an f32 0.4 literal", () => {
    exercisePublicShapeLifecycle();
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        // f32-round non-exact literals (0.35, 0.3) to match the C float inputs bit-for-bit.
        const unit: Sphere = { center: v(0, 0, 0), radius: 1 };
        assertMass(
            computeShapeMass(
                world,
                createSphereShape(world, body, testShapeDef({ density: 1 }), unit)!,
            ),
            sphereGold("unit"),
        );

        const offset: Sphere = { center: v(0.5, -1, 2), radius: f32(0.35) };
        assertMass(
            computeShapeMass(
                world,
                createSphereShape(world, body, testShapeDef({ density: 2.5 }), offset)!,
            ),
            sphereGold("offset"),
        );

        const vertical: Capsule = { center1: v(0, -1, 0), center2: v(0, 1, 0), radius: 0.5 };
        assertMass(
            computeShapeMass(
                world,
                createCapsuleShape(world, body, testShapeDef({ density: 1 }), vertical)!,
            ),
            capsuleGold("vertical"),
        );

        const skew: Capsule = {
            center1: v(-1, 0.5, 0.25),
            center2: v(1.5, -0.5, 0.75),
            radius: f32(0.3),
        };
        assertMass(
            computeShapeMass(
                world,
                createCapsuleShape(world, body, testShapeDef({ density: 3 }), skew)!,
            ),
            capsuleGold("skew"),
        );

        // Regression: an x-axis ragdoll bone (r = 0.12, water density). The sphere-inertia 0.4f
        // literal must be f32-rounded — f64 0.4 vs C 0.4f round differently for this mass, so the
        // buggy version was ~1 ULP off on the axis inertia (the divergence that broke the rain
        // fixture).
        const bone: Capsule = {
            center1: v(0.06, 0, 0),
            center2: v(-0.06, 0, 0),
            radius: f32(0.12),
        };
        assertMass(
            computeShapeMass(
                world,
                createCapsuleShape(world, body, testShapeDef({ density: 1000 }), bone)!,
            ),
            capsuleGold("bone"),
        );
    } finally {
        destroyTestWorld(world);
    }
});

test("sphere and capsule AABBs bound center +/- radius and follow the transform's translation, and a rotated capsule shape's AABB fills caller storage", () => {
    exercisePublicShapeLifecycle();
    // AABBs compose transform.point (bit-exact) with min/max/sub/add; concrete extremes pin the
    // wrapper without reimplementing the transform.
    const id = transform.identity();

    // sphere AABB is center +/- radius under identity
    const idSphere = computeSphereAABB({ center: v(0.5, -1, 2), radius: 0.25 }, id);
    expect(idSphere.lowerBound, "sphere AABB under identity: lowerBound").toEqual(
        v(0.25, -1.25, 1.75),
    );
    expect(idSphere.upperBound, "sphere AABB under identity: upperBound").toEqual(
        v(0.75, -0.75, 2.25),
    );

    // sphere AABB translates with the transform
    const t: Transform = { p: v(1, 2, 3), q: id.q };
    const movedSphere = computeSphereAABB({ center: v(0, 0, 0), radius: 1 }, t);
    expect(movedSphere.lowerBound, "sphere AABB translated by (1,2,3): lowerBound").toEqual(
        v(0, 1, 2),
    );
    expect(movedSphere.upperBound, "sphere AABB translated by (1,2,3): upperBound").toEqual(
        v(2, 3, 4),
    );

    // capsule AABB encloses both hemispheres under identity
    const capsule = computeCapsuleAABB(
        { center1: v(0, -1, 0), center2: v(0, 1, 0), radius: 0.5 },
        id,
    );
    expect(capsule.lowerBound, "capsule AABB under identity: lowerBound").toEqual(
        v(-0.5, -1.5, -0.5),
    );
    expect(capsule.upperBound, "capsule AABB under identity: upperBound").toEqual(v(0.5, 1.5, 0.5));

    const output = { lowerBound: v(0, 0, 0), upperBound: v(0, 0, 0) };
    const rotated: Transform = {
        p: v(1, 2, 3),
        q: { v: v(0, 0, Math.SQRT1_2), s: Math.SQRT1_2 },
    };
    const world = createTestWorld();
    try {
        const body = createTestBody(world, { type: BodyType.Dynamic });
        const shape = createCapsuleShape(world, body, testShapeDef(), {
            center1: v(0, -1, 0),
            center2: v(0, 1, 0),
            radius: 0.5,
        })!;
        expect(computeShapeAABBOut(world, shape, rotated, output)).toBe(output);
        expect(output).toEqual({
            lowerBound: v(fromBits("0xbefffffe"), 1.5, 2.5),
            upperBound: v(2.5, 2.5, 3.5),
        });
    } finally {
        destroyTestWorld(world);
    }
});

test("roundSphere and roundCapsule round every sphere and capsule field to f32", () => {
    exercisePublicShapeLifecycle();
    // The C holds geometry as f32 struct fields; callers pass f64 JS numbers. A field that is
    // not f32-exact (e.g. 0.3) must be rounded on storage or the solver arithmetic diverges
    // from the C.

    // roundSphere rounds every field to f32
    const s = roundSphere({ center: { x: 0.1, y: 0.2, z: 0.3 }, radius: 0.3 });
    expect(s.radius, "roundSphere(0.3 radius): radius").toBe(f32(0.3));
    expect(s.radius, "roundSphere(0.3 radius): radius must not stay f64").not.toBe(0.3);
    expect(s.center, "roundSphere(0.1, 0.2, 0.3 center): center").toEqual({
        x: f32(0.1),
        y: f32(0.2),
        z: f32(0.3),
    });

    // roundCapsule rounds every field to f32
    const c = roundCapsule({
        center1: { x: -0.3, y: 0, z: 0.7 },
        center2: { x: 0.3, y: 0, z: 0.7 },
        radius: 0.3,
    });
    expect(c.radius, "roundCapsule(0.3 radius): radius").toBe(f32(0.3));
    expect(c.radius, "roundCapsule(0.3 radius): radius must not stay f64").not.toBe(0.3);
    expect(c.center1.x, "roundCapsule(-0.3 center1.x): center1.x").toBe(f32(-0.3));
    expect(c.center1.z, "roundCapsule(0.7 center1.z): center1.z").toBe(f32(0.7));
    expect(c.center2, "roundCapsule(0.3, 0, 0.7 center2): center2").toEqual({
        x: f32(0.3),
        y: 0,
        z: f32(0.7),
    });
});
