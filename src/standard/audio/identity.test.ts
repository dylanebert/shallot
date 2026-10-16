import { expect, test } from "bun:test";
import { AudioOutput, Clips, play, registerClip, Sound } from "../../core/audio";
import { Devices } from "../../core/input";
import { World } from "../../engine";
import { AudioRuntime, loadClip, reconcileAudio } from "./runtime";

test("clips with the same display label keep distinct handles and PCM for an existing Sound", () => {
    const world = new World();
    try {
        const first = registerClip(world, new Float32Array([0.25]), 48000, "same");
        const sound = play(world, first);
        const second = registerClip(world, new Float32Array([0.75]), 48000, "same");
        expect(second).not.toBe(first);
        expect(world.storage(Sound).clip.get(sound)).toBe(first);
        expect(world.resource(Clips)[first].channels[0][0]).toBe(0.25);
        expect(world.resource(Clips)[second].channels[0][0]).toBe(0.75);
        expect(world.resource(Clips)[world.storage(Sound).clip.get(sound)].channels[0][0]).toBe(
            0.25,
        );
        const uploads: { id: number; channels: Float32Array[] }[] = [];
        const runtime = world.resource(AudioRuntime);
        runtime.context = { sampleRate: 48000 } as AudioContext;
        runtime.node = {
            port: {
                postMessage: (message: { type: string; id: number; channels: Float32Array[] }) => {
                    if (message.type === "clip") uploads.push(message);
                },
            },
        } as unknown as AudioWorkletNode;
        world.resource(Devices).audio.context = "running";
        reconcileAudio(world);
        const voice = runtime.slots.find((entry) => entry?.eid === sound)!;
        expect(voice.clip).toBe(first);
        expect(uploads.find((entry) => entry.id === voice.clip)?.channels[0][0]).toBe(0.25);
    } finally {
        world.dispose();
    }
});

test("loading one URL concurrently and again returns one handle with one fetch", async () => {
    const world = new World();
    const originalFetch = globalThis.fetch;
    let fetches = 0;
    globalThis.fetch = (async () => {
        fetches++;
        return new Response(new ArrayBuffer(4));
    }) as unknown as typeof fetch;
    world.resource(AudioOutput).context = {
        decodeAudioData: async () => ({
            numberOfChannels: 1,
            sampleRate: 48000,
            getChannelData: () => new Float32Array([0.5]),
        }),
    } as unknown as AudioContext;
    try {
        const [first, second] = await Promise.all([
            loadClip(world, "/tone.wav"),
            loadClip(world, "/tone.wav"),
        ]);
        expect(second).toBe(first);
        expect(await loadClip(world, "/tone.wav")).toBe(first);
        expect(fetches).toBe(1);
    } finally {
        globalThis.fetch = originalFetch;
        world.dispose();
    }
});
