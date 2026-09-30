import { expect, test } from "bun:test";
import {
    AudioPlugin,
    audioContextState,
    Devices,
    play,
    Sound,
    State,
    sample,
} from "@dylanebert/shallot";
import { Audio, alloc, gate } from "./device";

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
