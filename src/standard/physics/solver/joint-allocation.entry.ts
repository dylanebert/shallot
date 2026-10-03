import { createApp } from "@dylanebert/shallot";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init, shutdown, threads } from "../kernel/kernel";
import { makeBoxHull } from "../shapes/hull";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create(input: string) {
    const count = Number(input);
    await init(undefined, { threads: count });
    if (threads(undefined) !== (count || 1))
        throw new Error("joint allocation requires the requested schedule");
    const solver = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    const hull = makeBoxHull(0.25, 0.25, 0.25);
    const kinds = [
        "Parallel",
        "Distance",
        "Filter",
        "Motor",
        "Prismatic",
        "Revolute",
        "Spherical",
        "Weld",
        "Wheel",
    ] as const;
    for (let i = 0; i < kinds.length; i++) {
        const anchor = solver.createBody({ position: { x: i * 4, y: 0, z: 0 } });
        const body = solver.createBody({
            type: BodyType.Dynamic,
            position: { x: i * 4, y: 1, z: 0 },
        });
        body.createHull({}, hull);
        switch (kinds[i]) {
            case "Parallel":
                solver.createParallelJoint(anchor, body, {});
                break;
            case "Distance":
                solver.createDistanceJoint(anchor, body, { length: 1 });
                break;
            case "Filter":
                solver.createFilterJoint(anchor, body);
                break;
            case "Motor":
                solver.createMotorJoint(anchor, body, {});
                break;
            case "Prismatic":
                solver.createPrismaticJoint(anchor, body, {});
                break;
            case "Revolute":
                solver.createRevoluteJoint(anchor, body, {});
                break;
            case "Spherical":
                solver.createSphericalJoint(anchor, body, {});
                break;
            case "Weld":
                solver.createWeldJoint(anchor, body, {});
                break;
            case "Wheel":
                solver.createWheelJoint(anchor, body, {});
                break;
        }
    }
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "joint-allocation",
                systems: [
                    { name: "joint-step", group: "fixed", update: () => solver.step(1 / 60, 4) },
                ],
            },
        ],
    });
    return {
        step: () => app.world.step(1 / 60),
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: async () => {
            app.dispose();
            solver.destroy();
            await shutdown(undefined);
        },
    };
}
