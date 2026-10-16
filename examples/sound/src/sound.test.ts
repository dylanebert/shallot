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
import { chime } from "./chime";
import {
    CHIME_VOLUME,
    InteractionInput,
    InteractionPress,
    Sliding,
    SoundGamePlugin,
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
    const clip = registerClip(w, chime(48000));
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
test("the cube use creates one positioned once Sound on the synthesized clip at its volume", async () => {
    const { w, speaker, clip, use } = await scene();
    const bell = w.create();
    w.add(bell, Transform, { translation: [5, 1.4, -2] });
    w.add(bell, Usable, { bell: 1, clip });
    use(bell);
    const sounds = [...w.query([Sound])].filter((eid) => eid !== speaker);
    expect(sounds).toHaveLength(1);
    const once = sounds[0]!;
    expect(w.storage(Sound).loop.get(once)).toBe(0);
    expect(w.storage(Sound).clip.get(once)).toBe(clip);
    expect(w.storage(Sound).volume.get(once)).toBeCloseTo(CHIME_VOLUME);
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

test("the synthesized PCM is finite, non-silent and decays to silence by its end", () => {
    for (const rate of [44100, 48000]) {
        const pcm = chime(rate);
        expect(pcm.length).toBeLessThan(rate);
        expect(pcm.every(Number.isFinite)).toBe(true);
        const energy = (samples: Float32Array) =>
            samples.reduce((sum, x) => sum + x * x, 0) / samples.length;
        const onset = energy(pcm.subarray(0, rate / 10));
        const tail = energy(pcm.subarray(-rate / 10));
        expect(onset).toBeGreaterThan(0.01);
        expect(tail).toBeLessThan(onset * 0.0001);
        expect(pcm[0]).toBe(0);
        expect(pcm.at(-1)).toBe(0);
    }
});
