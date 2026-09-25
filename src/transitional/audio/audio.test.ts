import {
    AudioPlugin,
    audioContextState,
    devices,
    play,
    Sound,
    State,
    sample,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";

check(
    "suspended audio drops one-shots and keeps loops pending",
    {
        claim: "a suspended State drops a one-shot Sound while leaving a loop pending for resume",
    },
    () => {
        const state = new State();
        for (const system of AudioPlugin.systems ?? []) state.addSystem(system, AudioPlugin.name);
        sample(new Float32Array([0]), "s4-suspended-audio");
        audioContextState(state, "suspended");
        const oneShot = play(state, "s4-suspended-audio");
        const loop = play(state, "s4-suspended-audio", { loop: true });
        if (oneShot < 0 || loop < 0) throw new Error("test sounds did not spawn");
        Sound.voice.set(loop, -1);
        state.step(0);
        if (state.exists(oneShot)) throw new Error("suspended one-shot was not dropped");
        if (!state.exists(loop) || Sound.loop.get(loop) !== 1 || Sound.voice.get(loop) !== -1)
            throw new Error("suspended loop was not left pending");
        if (devices(state).audio.context !== "suspended") throw new Error("audio state changed");
        state.dispose();
    },
);
