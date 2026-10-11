import { createApp } from "@dylanebert/shallot";
import { DEFAULT_MASK_BITS } from "../common/constants";
import { queryColumns } from "../kernel/querycolumns";
import { destroyTestWorld } from "../testing";
import { worldDraw } from "./draw";
import { drawScene } from "./draw-scene";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};
export let leaked: Set<number> | undefined;

export default async function create(input: string) {
    const app = await createApp({ defaults: false, plugins: [] });
    const { world, draw } = drawScene();
    const q = queryColumns(world);
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
                const start = k.worldDraw(world.worldId, 31, 0xffffffff, 0xffffffff, world.invH);
                if (k.worldDrawLen() === start)
                    throw new Error("draw subject emitted no primitives");
                if (input === "nested-buffer") {
                    const nested = k.worldDraw(
                        world.worldId,
                        1,
                        0xffffffff,
                        0xffffffff,
                        world.invH,
                    );
                    if (nested <= start || k.worldDrawLen() <= nested)
                        throw new Error("nested draw lost its pending stream");
                    k.worldDrawRelease(nested);
                    if (k.worldDrawLen() !== nested)
                        throw new Error("nested draw failed to release its stream");
                }
                k.worldDrawRelease(start);
            } else {
                worldDraw(world, draw, DEFAULT_MASK_BITS);
                if (input === "leak") leaked = new Set([1, 2, 3]);
            }
        },
        dispose: () => {
            destroyTestWorld(world);
            app.dispose();
            leaked = undefined;
        },
    };
}
