import { expect, test } from "bun:test";
import { createApp, Time } from "../../engine";
import { GlobalTransform, Transform } from "../transform";
import {
    RenderTransformHistory,
    RenderTransformHistoryPlugin,
    readRenderGlobalTransform,
} from "./global-transform";

test("a fixed-clock jump discards drawn history instead of blending across the gap", async () => {
    const app = await createApp({ defaults: false, plugins: [RenderTransformHistoryPlugin] });
    const w = app.world;
    const eid = w.create();
    w.add(eid, Transform);
    w.add(eid, RenderTransformHistory);
    w.addSystem({
        group: "fixed",
        update(w) {
            w.storage(Transform).translation.x.set(eid, w.time.fixedTick * 10);
        },
    });
    const pose = new Float32Array(12);
    try {
        w.step(Time.FIXED_DT * 2.5);
        readRenderGlobalTransform(w, eid, pose);
        expect(pose[0]).toBeCloseTo(15);
        w.jumpToTick(401);
        readRenderGlobalTransform(w, eid, pose);
        expect(pose[0]).toBe(w.storage(GlobalTransform).translation.x.get(eid));
        expect(pose[0]).toBe(20);
        w.step(0);
        readRenderGlobalTransform(w, eid, pose);
        expect(pose[0]).toBe(20);
    } finally {
        app.dispose();
    }
});
