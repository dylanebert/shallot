import { expect, test } from "bun:test";
import {
    Camera,
    CameraMode,
    createApp,
    pressKey,
    resizeViewport,
    Transform,
} from "@dylanebert/shallot";
import { CameraViewport, RenderLayers } from "@dylanebert/shallot/rendering";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { Scene, ScenePlugin } from "./scene";

test("the arena's minimap follows the player and fits every viewport", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [ScenePlugin],
        setup(w) {
            w.resource(PhysicsWorldDefinition).workerCount = 1;
        },
    });
    try {
        const w = app.world,
            s = w.resource(Scene);
        expect(w.storage(Camera).mode.get(s.minimap)).toBe(CameraMode.Orthographic);
        expect(w.storage(Camera).order.get(s.minimap)).toBeGreaterThan(
            w.storage(Camera).order.get(s.camera),
        );
        expect(w.storage(RenderLayers).mask.get(s.marker)).toBe(2);
        expect(w.storage(RenderLayers).mask.get(s.minimap)).toBe(3);
        for (const [width, height] of [
            [640, 360],
            [390, 844],
            [80, 40],
            [1, 1],
        ]) {
            resizeViewport(w, 0, width!, height!, 1);
            w.step(1 / 60);
            const v = w.storage(CameraViewport);
            expect(v.width.get(s.minimap)).toBe(v.height.get(s.minimap));
            expect(v.x.get(s.minimap) + v.width.get(s.minimap)).toBeLessThanOrEqual(width!);
            expect(v.y.get(s.minimap) + v.height.get(s.minimap)).toBeLessThanOrEqual(height!);
        }
        const t = w.storage(Transform),
            before = t.translation.z.get(s.marker);
        pressKey(w, "KeyW");
        for (let i = 0; i < 30; i++) w.step(1 / 60);
        expect(t.translation.z.get(s.marker)).toBeLessThan(before - 0.5);
        expect(t.translation.x.get(s.marker)).toBe(t.translation.x.get(s.camera));
        expect(t.translation.z.get(s.marker)).toBe(t.translation.z.get(s.camera));
    } finally {
        app.dispose();
    }
});
