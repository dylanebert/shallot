import { expect, test } from "bun:test";
import {
    Camera,
    createApp,
    NotShadowCaster,
    pressKey,
    releaseKey,
    Transform,
} from "@dylanebert/shallot";
import { RenderLayers } from "@dylanebert/shallot/rendering";
import { Scene, ScenePlugin } from "./scene";

test("E drops the multipart tool into the world and picks it back up; W stops at the wall", async () => {
    const app = await createApp({ defaults: false, plugins: [ScenePlugin] });
    try {
        const w = app.world,
            s = w.resource(Scene);
        expect(w.storage(Camera).order.get(s.modelCamera)).toBe(1);
        expect(w.storage(Camera).fov.get(s.modelCamera)).toBe(70);
        const toggle = () => {
            pressKey(w, "KeyE");
            w.step(1 / 60);
            releaseKey(w, "KeyE");
            w.step(1 / 60);
        };
        const z = w.storage(Transform).translation.z.get(s.parts[0]!);
        toggle();
        expect(s.held).toBe(false);
        for (const eid of s.parts) {
            expect(w.storage(RenderLayers).mask.get(eid)).toBe(1);
            expect(w.has(eid, NotShadowCaster)).toBe(false);
        }
        pressKey(w, "KeyW");
        for (let i = 0; i < 180; i++) w.step(1 / 60);
        releaseKey(w, "KeyW");
        w.step(1 / 60);
        expect(s.z).toBe(-0.65);
        expect(w.storage(Transform).translation.z.get(s.parts[0]!)).toBe(z);
        toggle();
        expect(s.held).toBe(true);
        for (const eid of s.parts) {
            expect(w.storage(RenderLayers).mask.get(eid)).toBe(2);
            expect(w.has(eid, NotShadowCaster)).toBe(true);
        }
        expect(w.storage(Transform).translation.z.get(s.parts[0]!)).toBeLessThan(z);
    } finally {
        app.dispose();
    }
});
