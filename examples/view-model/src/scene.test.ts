import { expect, test } from "bun:test";
import {
    Camera,
    Character,
    createApp,
    NotShadowCaster,
    pointerLockChanged,
    pointerMove,
    pressKey,
    releaseKey,
    Transform,
} from "@dylanebert/shallot";
import { RenderLayers } from "@dylanebert/shallot/rendering";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { Scene, ScenePlugin } from "./scene";

test("the standard character stops at the wall; its cameras and multipart model follow the same frame pose", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            ScenePlugin,
            {
                name: "CheckPhysics",
                initialize(world) {
                    world.resource(PhysicsWorldDefinition).workerCount = 1;
                },
            },
        ],
    });
    try {
        const w = app.world,
            s = w.resource(Scene);
        expect(w.has(s.player, Character)).toBe(true);
        expect(w.storage(Camera).order.get(s.modelCamera)).toBe(1);
        expect(w.storage(Camera).fov.get(s.modelCamera)).toBe(70);
        pressKey(w, "KeyW");
        for (let frame = 0; frame < 180; frame++) w.step(1 / 60);
        releaseKey(w, "KeyW");
        w.step(1 / 60);
        const t = w.storage(Transform);
        expect(t.translation.z.get(s.player)).toBeGreaterThan(-0.61);
        expect(t.translation.z.get(s.player)).toBeLessThan(-0.5);
        pointerLockChanged(w, true);
        pointerMove(w, { x: 0, y: 0, deltaX: 50, deltaY: 10 });
        w.step(1 / 144);
        expect(t.rotation.y.get(s.camera)).not.toBe(0);
        for (const eid of [s.modelCamera, ...s.parts]) {
            expect(t.rotation.x.get(eid)).toBe(t.rotation.x.get(s.camera));
            expect(t.rotation.y.get(eid)).toBe(t.rotation.y.get(s.camera));
            expect(t.rotation.z.get(eid)).toBe(t.rotation.z.get(s.camera));
            expect(t.rotation.w.get(eid)).toBe(t.rotation.w.get(s.camera));
        }
        for (const eid of s.parts) {
            expect(w.storage(RenderLayers).mask.get(eid)).toBe(2);
            expect(w.has(eid, NotShadowCaster)).toBe(true);
        }
    } finally {
        app.dispose();
    }
});
