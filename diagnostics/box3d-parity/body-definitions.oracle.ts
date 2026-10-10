import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "../../src/engine";
import { Body, BodyMotionLock, BodyType } from "../../src/core/physics";
import { hashPhysics, StandardPhysicsPlugin } from "../../src/standard/physics";
import {
    defaultSurfaceMaterial,
    hash,
    init,
    makeBoxHull,
    PhysicsWorld,
} from "../../src/standard/physics/api";
import { nativeSseOutput } from "./native-evidence";
import { assertPublicOracleKernel } from "./oracle-kernel";

setDefaultTimeout(180_000);
await init(undefined, { threads: 0 });
await assertPublicOracleKernel();
const snapshotFields = [
    "hash",
    "type",
    "linearDamping",
    "angularDamping",
    "gravityScale",
    "sleepThreshold",
    "sleepEnabled",
    "enabled",
    "bullet",
    "motionLocks",
    "fastRotation",
    "contactRecycling",
    "awake",
    "massData.mass",
    "massData.center.x",
    "massData.center.y",
    "massData.center.z",
    "massData.inertia.cx.x",
    "massData.inertia.cx.y",
    "massData.inertia.cx.z",
    "massData.inertia.cy.x",
    "massData.inertia.cy.y",
    "massData.inertia.cy.z",
    "massData.inertia.cz.x",
    "massData.inertia.cz.y",
    "massData.inertia.cz.z",
];
const floatFields = new Set([
    2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
]);
const native = nativeSseOutput("body-definitions.c", "")
    .trim()
    .split("\n")
    .map((line, row) => {
        const values = line.trim().split(/\s+/);
        if (values.length !== snapshotFields.length)
            throw new Error(`native Body definitions row ${row} has ${values.length} fields`);
        return values.map((value, field) => {
            if (field === 0) return BigInt(`0x${value}`);
            return Number.parseInt(value, floatFields.has(field) ? 16 : 10);
        });
    });
if (native.length !== 25)
    throw new Error(`native Body definitions oracle returned ${native.length} snapshots`);

const floatStorage = new DataView(new ArrayBuffer(4));
const floatBits = (value: number) => {
    floatStorage.setFloat32(0, value, true);
    return floatStorage.getUint32(0, true);
};

function expectNativeSnapshot(
    world: PhysicsWorld,
    body: ReturnType<PhysicsWorld["createBody"]>,
    index: number,
    label: string,
): void {
    const expected = native[index];
    if (expected === undefined) throw new Error(`${label}: missing native snapshot at ${index}`);
    const mass = body.getMassData();
    const actual = [
        hash(world),
        body.getType(),
        floatBits(body.getLinearDamping()),
        floatBits(body.getAngularDamping()),
        floatBits(body.getGravityScale()),
        floatBits(body.getSleepThreshold()),
        Number(body.isSleepEnabled()),
        Number(body.isEnabled()),
        Number(body.isBullet()),
        body.getMotionLocks(),
        Number(body.isFastRotationAllowed()),
        Number(body.isContactRecyclingEnabled()),
        Number(body.isAwake()),
        floatBits(mass.mass),
        floatBits(mass.center.x),
        floatBits(mass.center.y),
        floatBits(mass.center.z),
        floatBits(mass.inertia.cx.x),
        floatBits(mass.inertia.cx.y),
        floatBits(mass.inertia.cx.z),
        floatBits(mass.inertia.cy.x),
        floatBits(mass.inertia.cy.y),
        floatBits(mass.inertia.cy.z),
        floatBits(mass.inertia.cz.x),
        floatBits(mass.inertia.cz.y),
        floatBits(mass.inertia.cz.z),
    ];
    const comparisonOrder = [
        1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9,
    ];
    for (const field of comparisonOrder) {
        if (actual[field] !== expected[field]) {
            throw new Error(
                `${label}: ${snapshotFields[field]} ECS ${actual[field]} != native Box3D ${expected[field]}`,
            );
        }
    }
    if (actual[0] !== expected[0]) {
        throw new Error(`${label}: hash ECS ${actual[0]} != native Box3D ${expected[0]}`);
    }
}

test("a default ECS Body hashes equal to native Box3D", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const eid = app.world.create();
        app.world.add(eid, Body);
        app.world.tick();
        expect(hashPhysics(app.world)).toBe(native[0]?.[0]);
    } finally {
        app.dispose();
    }
});

test("each live Body-definition setter matches native at its mutation boundary", () => {
    const world = new PhysicsWorld();
    try {
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 1, y: 4, z: -2 },
            linearVelocity: { x: 0.3, y: 0.1, z: -0.2 },
            angularVelocity: { x: 0.1, y: 0.2, z: -0.3 },
        });
        const material = defaultSurfaceMaterial();
        material.friction = 0.5;
        body.createSphere(
            { density: 1, baseMaterial: material },
            { center: { x: 0.25, y: 0.1, z: -0.15 }, radius: 0.5 },
        );
        let index = 1;
        const compare = (label: string) => expectNativeSnapshot(world, body, index++, label);

        compare("spawn body definition and shape");
        body.setLinearDamping(0.23);
        compare("setLinearDamping");
        body.setAngularDamping(0.31);
        compare("setAngularDamping");
        body.setGravityScale(0.4);
        compare("setGravityScale");
        body.setSleepThreshold(0.12);
        compare("setSleepThreshold");

        const linearXAndAngularZ = BodyMotionLock.linearX | BodyMotionLock.angularZ;
        body.setMotionLocks(linearXAndAngularZ);
        compare("setMotionLocks partial rotation locks");
        const fullyLockedRotation =
            BodyMotionLock.linearX |
            BodyMotionLock.angularX |
            BodyMotionLock.angularY |
            BodyMotionLock.angularZ;
        const inertiaBeforeFixedRotation = body.getMassData().inertia.cx.x;
        body.setMotionLocks(fullyLockedRotation);
        compare("setMotionLocks into fixed rotation and mass update");
        expect(body.getMassData().inertia.cx.x).toBe(0);
        body.setMotionLocks(BodyMotionLock.linearX);
        compare("setMotionLocks out of fixed rotation and mass update");
        expect(body.getMassData().inertia.cx.x).toBe(inertiaBeforeFixedRotation);

        body.enableSleep(false);
        compare("enableSleep(false)");
        body.enableSleep(true);
        compare("enableSleep(true)");
        body.setBullet(true);
        compare("setBullet(true)");
        body.allowFastRotation(true);
        compare("allowFastRotation(true)");
        body.setLinearVelocity({ x: 0, y: 0, z: 0 });
        compare("setLinearVelocity(zero) before sleep");
        body.setAngularVelocity({ x: 0, y: 0, z: 0 });
        compare("setAngularVelocity(zero) before sleep");
        body.setAwake(false);
        compare("setAwake(false)");
        body.setAwake(true);
        compare("setAwake(true)");
        body.setType(BodyType.Kinematic);
        compare("setType(kinematic)");
        body.setType(BodyType.Dynamic);
        compare("setType(dynamic)");
        body.disable();
        compare("disable");
        body.enable();
        compare("enable");
        expect(index).toBe(21);
    } finally {
        world.destroy();
    }
});

test("enableContactRecycling matches native through real contact creation", () => {
    const world = new PhysicsWorld();
    const identity = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    try {
        const floor = world.createBody({ position: { x: 0, y: -0.5, z: 0 } });
        floor.createHull({ baseMaterial: defaultSurfaceMaterial() }, makeBoxHull(5, 0.5, 5));
        const body = world.createBody({
            type: BodyType.Dynamic,
            position: { x: 0, y: 3, z: 0 },
        });
        body.createHull(
            { density: 1, baseMaterial: defaultSurfaceMaterial() },
            makeBoxHull(0.5, 0.5, 0.5),
        );
        let index = 21;
        const compare = (label: string) => expectNativeSnapshot(world, body, index++, label);

        body.enableContactRecycling(false);
        compare("enableContactRecycling(false)");
        body.setTransform({ x: 0, y: 0.4, z: 0 }, identity);
        world.step(1 / 60, 4);
        expect(world.getCounters().contactCount).toBeGreaterThan(0);
        compare("contact created with recycling disabled");

        body.enableContactRecycling(true);
        compare("enableContactRecycling(true)");
        body.setTransform({ x: 0, y: 3, z: 0 }, identity);
        world.step(1 / 60, 4);
        body.setTransform({ x: 0, y: 0.4, z: 0 }, identity);
        world.step(1 / 60, 4);
        expect(world.getCounters().contactCount).toBeGreaterThan(0);
        compare("contact recreated with recycling enabled");
        expect(index).toBe(25);
    } finally {
        world.destroy();
    }
});
