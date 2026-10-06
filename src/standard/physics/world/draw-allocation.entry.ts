import { createApp } from "@dylanebert/shallot";
import { queryColumns } from "../kernel/querycolumns";
import { drawScene } from "./draw-scene";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export let leaked: Set<number> | undefined;

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [] });
    const { world, draw } = drawScene();
    const q = queryColumns(world.state);
    const k = q.prepare(draw.drawingBounds.lowerBound);
    draw.drawShapes =
        draw.drawBounds =
        draw.drawMass =
        draw.drawJoints =
        draw.drawJointExtras =
            false;
    return {
        step: () => {
            if (input === "buffer") {
                q.prepare(draw.drawingBounds.lowerBound);
                q.bounds(draw.drawingBounds);
                k.worldDraw(world.state.worldId, 31, 0xffffffff, 0xffffffff, world.state.invH);
                if (k.worldDrawLen() === 0) throw new Error("draw subject emitted no primitives");
            } else {
                world.draw(draw);
                if (input === "leak") leaked = new Set([1, 2, 3]);
            }
        },
        wait: () => app.world.gpu.device.queue.onSubmittedWorkDone(),
        dispose: () => {
            world.destroy();
            app.dispose();
            leaked = undefined;
        },
    };
}
