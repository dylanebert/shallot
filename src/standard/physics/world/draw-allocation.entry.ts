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
            if (input === "buffer" || input === "nested-buffer") {
                q.prepare(draw.drawingBounds.lowerBound);
                q.bounds(draw.drawingBounds);
                const start = k.worldDraw(
                    world.state.worldId,
                    31,
                    0xffffffff,
                    0xffffffff,
                    world.state.invH,
                );
                if (k.worldDrawLen() === start)
                    throw new Error("draw subject emitted no primitives");
                if (input === "nested-buffer") {
                    const nested = k.worldDraw(
                        world.state.worldId,
                        1,
                        0xffffffff,
                        0xffffffff,
                        world.state.invH,
                    );
                    if (nested <= start || k.worldDrawLen() <= nested)
                        throw new Error("nested draw lost its pending stream");
                    k.worldDrawRelease(nested);
                    if (k.worldDrawLen() !== nested)
                        throw new Error("nested draw failed to release its stream");
                }
                k.worldDrawRelease(start);
            } else {
                world.draw(draw);
                if (input === "leak") leaked = new Set([1, 2, 3]);
            }
        },
        dispose: () => {
            world.destroy();
            app.dispose();
            leaked = undefined;
        },
    };
}
