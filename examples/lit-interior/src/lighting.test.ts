import { afterEach, expect, test } from "bun:test";
import {
    Body,
    BodyType,
    Content,
    createApp,
    InputPlugin,
    Outline,
    PointLight,
    pressKey,
    releaseKey,
    Shape,
    SpotLight,
    StandardPhysicsPlugin,
    Transform,
    TransformPlugin,
} from "@dylanebert/shallot";
import { Flashlight, FlashlightPlugin } from "./flashlight";
import { InteractionInput, InteractPlugin, Usable } from "./interact";

const apps: Awaited<ReturnType<typeof createApp>>[] = [];
afterEach(() => {
    for (const app of apps) app.dispose();
    apps.length = 0;
});
async function scene() {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "GameCheck",
                recovery: "stateless",
                dependencies: [StandardPhysicsPlugin, InputPlugin, TransformPlugin],
                components: [...InteractPlugin.components!, Outline, PointLight, SpotLight],
                systems: [...InteractPlugin.systems!, ...FlashlightPlugin.systems!],
            },
        ],
    });
    apps.push(app);
    const w = app.world;
    w.resource(Content);
    const camera = w.create();
    w.add(camera, Transform, { translation: [0, 1.4, 0] });
    w.add(camera, InteractionInput);
    const lamp = w.create();
    w.add(lamp, PointLight, { intensity: 80000 });
    const target = w.create();
    w.add(target, Transform, { translation: [0, 1.4, -2] });
    w.add(target, Body, { type: BodyType.Static });
    w.add(target, Shape);
    w.add(target, Usable, { active: 1, lamp });
    const light = w.create();
    w.add(light, Transform);
    w.add(light, SpotLight, { shadowMapsEnabled: 1 });
    Object.assign(w.resource(Flashlight), { camera, light });
    w.tick();
    w.step(1 / 60);
    return { w, camera, light, lamp };
}

test("E on the wall switch sets its ceiling lamp's intensity to zero and back", async () => {
    const { w, lamp } = await scene();
    expect(w.storage(PointLight).intensity.get(lamp)).toBe(80000);
    pressKey(w, "KeyE");
    w.step(1 / 60);
    expect(w.storage(PointLight).intensity.get(lamp)).toBe(0);
    releaseKey(w, "KeyE");
    w.step(1 / 60);
    pressKey(w, "KeyE");
    w.step(1 / 60);
    expect(w.storage(PointLight).intensity.get(lamp)).toBe(80000);
});

test("the casting flashlight's placement equals the view's each frame, including fractional and zero steps", async () => {
    const { w, camera, light } = await scene();
    const t = w.storage(Transform);
    for (const [i, dt] of [1 / 120, 0, 1 / 60].entries()) {
        t.translation.set(camera, i + 1, 1.7, -i);
        t.rotation.set(camera, 0, Math.sin(i / 4), 0, Math.cos(i / 4));
        w.step(dt);
        expect([
            t.translation.x.get(light),
            t.translation.y.get(light),
            t.translation.z.get(light),
        ]).toEqual([
            t.translation.x.get(camera),
            t.translation.y.get(camera),
            t.translation.z.get(camera),
        ]);
        expect([
            t.rotation.x.get(light),
            t.rotation.y.get(light),
            t.rotation.z.get(light),
            t.rotation.w.get(light),
        ]).toEqual([
            t.rotation.x.get(camera),
            t.rotation.y.get(camera),
            t.rotation.z.get(camera),
            t.rotation.w.get(camera),
        ]);
    }
    expect(w.storage(SpotLight).shadowMapsEnabled.get(light)).toBe(1);
});
