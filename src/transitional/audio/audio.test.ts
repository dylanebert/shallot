import { expect, test } from "bun:test";
import {
    AudioPlugin,
    audioContextState,
    Devices,
    play,
    Sound,
    State,
    sample,
    sfx,
} from "@dylanebert/shallot";
import { Audio, alloc, gate, tickAudio } from "./device";

test("each State uploads each sample version once to its own worklet", () => {
    const a = new State();
    const b = new State();
    const id = sample(new Float32Array([0.25]), "recipient-sample");
    const received: object[][] = [[], []];
    for (const [i, state] of [a, b].entries()) {
        state.resource(Audio).node = {
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

test("SFX cooldown progress belongs to the State that admitted the trigger", () => {
    const a = new State();
    const b = new State();
    sample(new Float32Array([0]), "recipient-cooldown");
    sfx("recipient-cooldown", { cooldown: 1 });
    expect(play(a, "recipient-cooldown")).toBeGreaterThanOrEqual(0);
    expect(play(b, "recipient-cooldown")).toBeGreaterThanOrEqual(0);
    expect(play(a, "recipient-cooldown")).toBe(-1);
    expect(play(b, "recipient-cooldown")).toBe(-1);
    a.dispose();
    b.dispose();
});

test("audio voice slots and worklet queues belong to their explicit State", () => {
    const a = new State();
    const b = new State();
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

test("a suspended State drops a one-shot Sound while leaving a loop pending for resume", () => {
    const state = new State();
    for (const system of AudioPlugin.systems ?? []) state.addSystem(system, AudioPlugin.name);
    sample(new Float32Array([0]), "s4-suspended-audio");
    audioContextState(state, "suspended");
    const oneShot = play(state, "s4-suspended-audio");
    const loop = play(state, "s4-suspended-audio", { loop: true });
    if (oneShot < 0 || loop < 0) throw new Error("test sounds did not spawn");
    state.of(Sound).voice.set(loop, -1);
    state.step(0);
    if (state.exists(oneShot)) throw new Error("suspended one-shot was not dropped");
    if (
        !state.exists(loop) ||
        state.of(Sound).loop.get(loop) !== 1 ||
        state.of(Sound).voice.get(loop) !== -1
    )
        throw new Error("suspended loop was not left pending");
    if (state.resource(Devices).audio.context !== "suspended")
        throw new Error("audio state changed");
    state.dispose();
});
