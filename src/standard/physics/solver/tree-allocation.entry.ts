import { createApp } from "@dylanebert/shallot";
import { PhysicsWorld } from "../api/world";
import { BodyType } from "../common/types";
import { init, shutdown } from "../kernel/kernel";
import * as tree from "../kernel/treecolumns";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

export default async function create() {
    await init(undefined, { threads: 0 });
    const solver = new PhysicsWorld({
        gravity: { x: 0, y: 0, z: 0 },
        enableSleep: false,
        enableContinuous: false,
    });
    const bodies: ReturnType<PhysicsWorld["createBody"]>[] = [];
    for (let i = 0; i < 64; i++) {
        const body = solver.createBody({
            type: BodyType.Kinematic,
            position: { x: (i % 32) * 3, y: 1, z: Math.floor(i / 32) * 3 },
        });
        body.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        bodies.push(body);
    }
    const position = { x: 0, y: 1, z: 0 };
    const rotation = { v: { x: 0, y: 0, z: 0 }, s: 1 };
    const box = { lowerBound: { x: -10, y: -10, z: -10 }, upperBound: { x: -9, y: -9, z: -9 } };
    let frame = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "tree-allocation",
                systems: [
                    {
                        name: "tree-step",
                        group: "fixed",
                        update: () => {
                            frame++;
                            for (let i = 0; i < bodies.length; i++) {
                                position.x = (i % 32) * 3 + (frame % 2);
                                position.z = Math.floor(i / 32) * 3;
                                bodies[i].setTransform(position, rotation);
                            }
                            const t = solver.state.broadPhase.trees[BodyType.Kinematic];
                            const proxy = tree.createProxy(t, box, 0, 1, 0);
                            tree.enlargeProxy(t, proxy, box);
                            tree.destroyProxy(t, proxy);
                            tree.rebuild(t, false);
                            solver.step(1 / 60, 4);
                        },
                    },
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
