import { afterEach, expect, test } from "bun:test";
import {
    Body,
    BodyType,
    createApp,
    play,
    pressKey,
    registerClip,
    releaseKey,
    Shape,
    Sound,
    StandardPhysicsPlugin,
    setInputEnabled,
    Transform,
} from "@dylanebert/shallot";
import {
    InteractionInput,
    InteractionPress,
    Sliding,
    SoundGamePlugin,
    setVolume,
    Usable,
} from "./sound";

const apps: Awaited<ReturnType<typeof createApp>>[] = [];
afterEach(() => {
    for (const app of apps) app.dispose();
    apps.length = 0;
});
async function scene() {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, SoundGamePlugin],
    });
    apps.push(app);
    const w = app.world;
    const clip = registerClip(w, new Float32Array([0.2, 0.1, 0]));
    const speaker = play(w, clip, { loop: true, position: [0, 1.4, -2] });
    w.add(speaker, Usable);
    w.add(speaker, Body, { type: BodyType.Static });
    w.add(speaker, Shape);
    const camera = w.create();
    w.add(camera, InteractionInput);
    w.add(camera, Transform, { translation: [0, 1.4, 0.4] });
    w.tick();
    const use = (target: number) => {
        const press = w.create();
        w.add(press, InteractionPress, { camera, target });
        w.tick();
    };
    return { w, speaker, clip, use };
}
test("E on the speaker sets and clears paused on its Sound", async () => {
    const { w, speaker } = await scene();
    setInputEnabled(w, true);
    pressKey(w, "KeyE");
    w.step(1 / 60);
    expect(w.storage(Sound).paused.get(speaker)).toBe(1);
    releaseKey(w, "KeyE");
    w.step(1 / 60);
    pressKey(w, "KeyE");
    w.step(1 / 60);
    expect(w.storage(Sound).paused.get(speaker)).toBe(0);
});
test("the slider's value is the music's volume", async () => {
    const { w, speaker } = await scene();
    setVolume(w, speaker, 0.23);
    expect(w.storage(Sound).volume.get(speaker)).toBeCloseTo(0.23);
});
test("the bell use creates one positioned once Sound", async () => {
    const { w, speaker, clip, use } = await scene();
    const bell = w.create();
    w.add(bell, Transform, { translation: [5, 1.4, -2] });
    w.add(bell, Usable, { bell: 1, clip });
    use(bell);
    const sounds = [...w.query([Sound])].filter((eid) => eid !== speaker);
    expect(sounds).toHaveLength(1);
    const once = sounds[0]!;
    expect(w.storage(Sound).loop.get(once)).toBe(0);
    expect(w.storage(Transform).translation.x.get(once)).toBe(5);
    expect(w.storage(Transform).translation.y.get(once)).toBeCloseTo(1.4);
    expect(w.storage(Transform).translation.z.get(once)).toBe(-2);
});
test("the speaker's Sound rides its moving transform", async () => {
    const { w, speaker } = await scene();
    w.add(speaker, Sliding);
    w.tick();
    const first = w.storage(Transform).translation.x.get(speaker);
    w.tick();
    expect(w.storage(Transform).translation.x.get(speaker)).toBeGreaterThan(first);
    expect(w.has(speaker, Sound)).toBe(true);
});
