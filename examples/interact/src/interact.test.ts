import { afterEach, expect, test } from "bun:test";
import {
    Body,
    BodyType,
    Content,
    createApp,
    InputPlugin,
    NotShadowCaster,
    Outline,
    PointLight,
    pressKey,
    releaseKey,
    Shape,
    Sprite,
    StandardPhysicsPlugin,
    Text,
    Transform,
    TransformPlugin,
    type World,
} from "@dylanebert/shallot";
import { Action, InteractionInput, InteractPlugin, makePrompt, Prompt, Usable } from "./interact";

const apps: Awaited<ReturnType<typeof createApp>>[] = [];
afterEach(() => {
    for (const app of apps) app.dispose();
    apps.length = 0;
});
async function scene(action: number = Action.Lamp) {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "GameCheck",
                recovery: "stateless",
                dependencies: [StandardPhysicsPlugin, InputPlugin, TransformPlugin],
                components: [
                    ...InteractPlugin.components!,
                    Outline,
                    PointLight,
                    Text,
                    Sprite,
                    NotShadowCaster,
                ],
                systems: InteractPlugin.systems,
            },
        ],
    });
    apps.push(app);
    const w = app.world;
    const camera = w.create();
    w.add(camera, Transform, { translation: [0, 1.4, -1.8] });
    w.add(camera, InteractionInput);
    const target = box(w, [0, 1.4, -3.8]);
    const light = w.create();
    w.add(light, PointLight, { intensity: 0 });
    w.add(target, Usable, { action, light });
    const prompt = makePrompt(w, 0, 0);
    w.resource(Content);
    w.tick();
    const step = (n = 1) => {
        for (let i = 0; i < n; i++) w.step(1 / 60);
    };
    step();
    return { w, camera, target, light, prompt, step };
}
function box(w: World, translation: readonly [number, number, number]) {
    const eid = w.create();
    w.add(eid, Transform, { translation });
    w.add(eid, Body, { type: BodyType.Static });
    w.add(eid, Shape);
    return eid;
}

test("in reach and in view focuses, outlines and shows the prompt", async () => {
    const { w, camera, target, prompt } = await scene();
    expect(w.storage(InteractionInput).target.get(camera)).toBe(target);
    expect(w.has(target, Outline)).toBe(true);
    expect(w.storage(Sprite).visible.get(prompt)).toBe(1);
    expect(w.has(prompt, NotShadowCaster)).toBe(true);
    expect(w.has(w.storage(Prompt).ring.get(prompt), NotShadowCaster)).toBe(true);
});

test("the billboard cap, key and ring have distinct view depths and both texts share a baseline, even side-on", async () => {
    const { w, camera, prompt, step } = await scene();
    const t = w.storage(Transform);
    const p = w.storage(Prompt);
    const key = p.key.get(prompt),
        label = p.label.get(prompt),
        ring = p.ring.get(prompt);
    for (const side of [false, true]) {
        if (side) {
            t.translation.set(camera, 2, 1.4, -3.8);
            t.rotation.set(camera, 0, Math.SQRT1_2, 0, Math.SQRT1_2);
            step();
        }
        const depth = side ? t.translation.x : t.translation.z;
        expect(depth.get(key) - depth.get(prompt)).toBeCloseTo(0.01);
        expect(depth.get(ring) - depth.get(prompt)).toBeCloseTo(0.02);
        expect(t.translation.y.get(key)).toBeCloseTo(t.translation.y.get(label));
        expect(w.storage(Text).anchor.y.get(key)).toBe(0);
        expect(w.storage(Text).anchor.y.get(label)).toBe(0);
    }
});

test("out of reach or behind a wall neither focuses nor outlines nor shows the prompt", async () => {
    const { w, camera, target, prompt, step } = await scene();
    w.storage(Transform).translation.z.set(camera, 2);
    step();
    expect(w.storage(InteractionInput).target.get(camera)).toBe(0);
    expect(w.has(target, Outline)).toBe(false);
    expect(w.storage(Sprite).visible.get(prompt)).toBe(0);
    w.storage(Transform).translation.z.set(camera, -1.8);
    w.storage(Transform).rotation.set(camera, 0, 1, 0, 0);
    step();
    expect(w.storage(InteractionInput).target.get(camera)).toBe(0);
    expect(w.has(target, Outline)).toBe(false);
    expect(w.storage(Sprite).visible.get(prompt)).toBe(0);
    w.storage(Transform).rotation.set(camera, 0, 0, 0, 1);
    box(w, [0, 1.4, -2.7]);
    w.tick();
    step();
    expect(w.storage(InteractionInput).target.get(camera)).toBe(0);
    expect(w.has(target, Outline)).toBe(false);
    expect(w.storage(Sprite).visible.get(prompt)).toBe(0);
});

test("one press stays latched until a fixed tick and fires once", async () => {
    const { w, target, step } = await scene();
    pressKey(w, "KeyE");
    w.step(0);
    expect(w.storage(Usable).active.get(target)).toBe(0);
    for (let tick = 0; tick < 4; tick++) {
        step();
        expect(w.storage(Usable).active.get(target)).toBe(1);
    }
});

test("a hold fires on its completing tick, not before, and not again while held", async () => {
    const { w, target, step } = await scene(Action.Hold);
    pressKey(w, "KeyE");
    for (let tick = 0; tick < 59; tick++) {
        step();
        expect(w.storage(Usable).active.get(target)).toBe(0);
    }
    step();
    expect(w.storage(Usable).active.get(target)).toBe(1);
    for (let tick = 0; tick < 65; tick++) {
        step();
        expect(w.storage(Usable).active.get(target)).toBe(1);
    }
});

test("a hold resets on release or lost focus and requires a fresh press", async () => {
    const { w, target, camera, step } = await scene(Action.Hold);
    pressKey(w, "KeyE");
    step(30);
    releaseKey(w, "KeyE");
    step();
    expect(w.storage(InteractionInput).ticks.get(camera)).toBe(0);
    pressKey(w, "KeyE");
    step(30);
    w.storage(Transform).translation.x.set(camera, 5);
    step();
    expect(w.storage(InteractionInput).ticks.get(camera)).toBe(0);
    w.storage(Transform).translation.x.set(camera, 0);
    step(65);
    expect(w.storage(Usable).active.get(target)).toBe(0);
    releaseKey(w, "KeyE");
    step();
    pressKey(w, "KeyE");
    step(60);
    expect(w.storage(Usable).active.get(target)).toBe(1);
});

test("lost focus between fixed ticks remains latched until the hold consumes it", async () => {
    const { w, camera, target, step } = await scene(Action.Hold);
    pressKey(w, "KeyE");
    step(30);
    w.storage(Transform).translation.x.set(camera, 5);
    w.step(0);
    w.storage(Transform).translation.x.set(camera, 0);
    w.step(0);
    step(60);
    expect(w.storage(Usable).active.get(target)).toBe(0);
    expect(w.storage(InteractionInput).ticks.get(camera)).toBe(0);
});

for (const [name, action] of Object.entries(Action))
    test(`${name} changes its world state on a fixed tick`, async () => {
        const { w, target, light, step } = await scene(action);
        pressKey(w, "KeyE");
        step(action === Action.Hold ? 60 : 1);
        expect(w.storage(Usable).active.get(target)).toBe(1);
        if (action === Action.Lamp) expect(w.storage(PointLight).intensity.get(light)).toBe(150000);
        if (action === Action.Button)
            expect(w.storage(Transform).translation.z.get(target)).toBeCloseTo(-3.92);
        if (action === Action.Hold)
            expect(w.storage(Transform).translation.y.get(target)).toBeCloseTo(2.2);
    });
