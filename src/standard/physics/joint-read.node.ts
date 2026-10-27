import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    Transform,
    WeldJoint,
    WheelJoint,
} from "@dylanebert/shallot";
import {
    getJointLinearSeparation,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import gold from "./joint-read.gold.json";

setDefaultTimeout(TIMEOUT.node);

async function publicJointSeparations(): Promise<number[]> {
    const result: number[] = [];
    for (const kind of [
        FilterJoint,
        DistanceJoint,
        MotorJoint,
        ParallelJoint,
        PrismaticJoint,
        RevoluteJoint,
        SphericalJoint,
        WeldJoint,
        WheelJoint,
    ]) {
        let joint = 0;
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin],
            setup(world) {
                world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
                const a = world.create(),
                    b = world.create();
                world.add(a, Transform);
                world.add(b, Transform, { translation: [1, 2, 3] });
                world.add(a, Body, { type: BodyType.Dynamic });
                world.add(b, Body, { type: BodyType.Dynamic });
                joint = world.create();
                world.add(joint, kind, { a, b });
            },
        });
        try {
            expect(getJointLinearSeparation(app.world, joint)).toBeNull();
            app.world.tick();
            const separation = getJointLinearSeparation(app.world, joint);
            expect(separation).not.toBeNull();
            result.push(new Uint32Array(new Float32Array([separation!]).buffer)[0]!);
            app.world.destroy(joint);
            expect(getJointLinearSeparation(app.world, joint)).toBeNull();
            app.world.tick();
            expect(getJointLinearSeparation(app.world, joint)).toBeNull();
        } finally {
            app.dispose();
        }
    }
    return result;
}

test("public joint linear separations equal native Box3D and absent joints return null", async () => {
    expect(await publicJointSeparations()).toEqual(gold);
});
