import { expect, test } from "bun:test";
import {
    AudioPlugin,
    audioContextState,
    Devices,
    play,
    Sound,
    sample,
    sfx,
    World,
} from "@dylanebert/shallot";
import { Audio, alloc, gate, tickAudio } from "./device";

test("each World uploads each sample version once to its own worklet", () => {
    const a = new World();
    const b = new World();
    const id = sample(new Float32Array([0.25]), "recipient-sample");
    const received: object[][] = [[], []];
    for (const [i, world] of [a, b].entries()) {
        world.resource(Audio).node = {
            port: {
                postMessage: (batch: { commands: object[] }) => received[i].push(...batch.commands),
            },
        } as unknown as AudioWorkletNode;
    }
    const uploads = (i: number) =>
        received[i].filter(
            (command) =>
                (command as { type: string; id: number }).type === "set_sample" &&
                (command as { id: number }).id === id,
        );
    tickAudio(a);
    tickAudio(b);
    expect([uploads(0).length, uploads(1).length]).toEqual([1, 1]);
    const counts = received.map((commands) => commands.length);
    tickAudio(a);
    tickAudio(b);
    expect(received.map((commands) => commands.length)).toEqual(counts);
    sample(new Float32Array([0.5]), "recipient-sample");
    tickAudio(a);
    tickAudio(b);
    expect([uploads(0).length, uploads(1).length]).toEqual([2, 2]);
    a.dispose();
    b.dispose();
});

test("SFX cooldown progress belongs to the World that admitted the trigger", () => {
    const a = new World();
    const b = new World();
    sample(new Float32Array([0]), "recipient-cooldown");
    sfx("recipient-cooldown", { cooldown: 1 });
    expect(play(a, "recipient-cooldown")).toBeGreaterThanOrEqual(0);
    expect(play(b, "recipient-cooldown")).toBeGreaterThanOrEqual(0);
    expect(play(a, "recipient-cooldown")).toBe(-1);
    expect(play(b, "recipient-cooldown")).toBe(-1);
    a.dispose();
    b.dispose();
});

test("audio voice slots and worklet queues belong to their explicit World", () => {
    const a = new World();
    const b = new World();
    const first = a.resource(Audio);
    const second = b.resource(Audio);
    const voice = alloc(a);
    expect(first.free.length).toBe(63);
    expect(second.free.length).toBe(64);
    gate(a, voice, 1);
    expect(first.queue).toEqual([
        { type: "voice_active", voiceId: 0, active: true },
        { type: "gate", voiceId: 0, value: 1 },
    ]);
    expect(second.queue.length).toBe(0);
    expect(first).not.toBe(second);
    a.dispose();
    b.dispose();
});

test("a suspended World drops a one-shot Sound while leaving a loop pending for resume", () => {
    const world = new World();
    for (const system of AudioPlugin.systems ?? []) world.addSystem(system, AudioPlugin.name);
    sample(new Float32Array([0]), "s4-suspended-audio");
    audioContextState(world, "suspended");
    const oneShot = play(world, "s4-suspended-audio");
    const loop = play(world, "s4-suspended-audio", { loop: true });
    if (oneShot < 0 || loop < 0) throw new Error("test sounds did not spawn");
    world.storage(Sound).voice.set(loop, -1);
    world.step(0);
    if (world.exists(oneShot)) throw new Error("suspended one-shot was not dropped");
    if (
        !world.exists(loop) ||
        world.storage(Sound).loop.get(loop) !== 1 ||
        world.storage(Sound).voice.get(loop) !== -1
    )
        throw new Error("suspended loop was not left pending");
    if (world.resource(Devices).audio.context !== "suspended")
        throw new Error("audio state changed");
    world.dispose();
});
