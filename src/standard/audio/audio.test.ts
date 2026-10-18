import { expect, test } from "bun:test";
import { AudioOutput, Clips, Listener, play, registerClip, Sound } from "../../core/audio";
import { Devices } from "../../core/input";
import { GlobalTransform, Transform } from "../../core/transform";
import { World } from "../../engine";
import {
    AudioRuntime,
    disposeAudio,
    initializeAudio,
    installAudioRestore,
    reconcileAudio,
    returnAudioBuffer,
} from "./runtime";

interface Message {
    type: string;
    id?: number;
    buffer?: Float64Array;
}

function voiceFor(world: World, eid: number) {
    return world.resource(AudioRuntime).slots.find((voice) => voice?.eid === eid);
}

function setup(world: World): Message[] {
    const messages: Message[] = [];
    const runtime = world.resource(AudioRuntime);
    installAudioRestore(world, runtime);
    runtime.context = { sampleRate: 48000 } as AudioContext;
    runtime.node = {
        port: {
            postMessage(message: Message) {
                messages.push(message);
            },
        },
    } as unknown as AudioWorkletNode;
    world.resource(AudioOutput).context = runtime.context;
    world.resource(Devices).audio.context = "running";
    return messages;
}

function returnBatch(world: World, messages: Message[], finished: readonly number[] = []): void {
    const message = [...messages].reverse().find((entry) => entry.type === "batch");
    if (!message?.buffer) throw new Error("audio frame did not send a command buffer");
    returnAudioBuffer(world, message.buffer, finished);
}

function commands(message: Message): Float64Array {
    if (!message.buffer) throw new Error("message has no command buffer");
    return message.buffer;
}

function lastBatch(messages: Message[]): Message {
    const message = [...messages].reverse().find((entry) => entry.type === "batch");
    if (!message) throw new Error("audio frame did not send a command buffer");
    return message;
}

test("pausing a live loop retains its voice without STOP and resumes without PLAY", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.3]));
    const eid = play(world, clip, { loop: true });
    reconcileAudio(world);
    const handle = voiceFor(world, eid)!.handle;
    returnBatch(world, messages);
    world.storage(Sound).paused.set(eid, 1);
    reconcileAudio(world);
    expect(commands(lastBatch(messages))[2]).not.toBe(3);
    expect(voiceFor(world, eid)?.handle).toBe(handle);
    returnBatch(world, messages);
    world.storage(Sound).paused.set(eid, 0);
    reconcileAudio(world);
    expect(commands(lastBatch(messages))[2]).not.toBe(1);
    expect(voiceFor(world, eid)?.handle).toBe(handle);
    world.dispose();
});

test("a paused once sound with a live owner survives restore and completes after resume", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.3]));
    const eid = play(world, clip);
    reconcileAudio(world);
    const handle = voiceFor(world, eid)!.handle;
    returnBatch(world, messages);
    world.storage(Sound).paused.set(eid, 1);
    reconcileAudio(world);
    returnBatch(world, messages);
    const snapshot = world.snapshot();
    world.restore(snapshot);
    expect(world.has(eid, Sound)).toBe(true);
    expect(voiceFor(world, eid)?.handle).toBe(handle);
    world.storage(Sound).paused.set(eid, 0);
    reconcileAudio(world);
    returnBatch(world, messages, [handle]);
    expect(world.has(eid, Sound)).toBe(false);
    world.dispose();
});

test("a paused sound without a voice does not start", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.3]));
    const eid = play(world, clip, { loop: true });
    world.storage(Sound).paused.set(eid, 1);
    reconcileAudio(world);
    expect(commands(lastBatch(messages))[0]).toBe(0);
    expect(voiceFor(world, eid)).toBeUndefined();
    world.dispose();
});

test("each World owns its clip registry and uploads each clip once to its own worklet", () => {
    const first = new World();
    const second = new World();
    const firstMessages = setup(first);
    const secondMessages = setup(second);
    const firstId = registerClip(first, new Float32Array([0.25]), 48000, "shared-name");
    const secondId = registerClip(second, new Float32Array([0.75]), 48000, "shared-name");
    expect(first.resource(Clips)).not.toBe(second.resource(Clips));
    expect(first.resource(Clips)[firstId]?.channels[0][0]).toBe(0.25);
    expect(second.resource(Clips)[secondId]?.channels[0][0]).toBe(0.75);

    reconcileAudio(first);
    reconcileAudio(second);
    expect([firstId, secondId]).toEqual([0, 0]);
    expect(
        firstMessages.filter((message) => message.type === "clip").map((message) => message.id),
    ).toEqual([0]);
    expect(
        secondMessages.filter((message) => message.type === "clip").map((message) => message.id),
    ).toEqual([0]);
    returnBatch(first, firstMessages);
    returnBatch(second, secondMessages);
    reconcileAudio(first);
    reconcileAudio(second);
    expect(firstMessages.filter((message) => message.type === "clip")).toHaveLength(1);
    expect(secondMessages.filter((message) => message.type === "clip")).toHaveLength(1);

    registerClip(first, new Float32Array([0.5]), 48000, "shared-name");
    returnBatch(first, firstMessages);
    reconcileAudio(first);
    returnBatch(second, secondMessages);
    reconcileAudio(second);
    expect(firstMessages.filter((message) => message.type === "clip")).toHaveLength(2);
    expect(secondMessages.filter((message) => message.type === "clip")).toHaveLength(1);
    first.dispose();
    second.dispose();
});

test("play adds a positional Sound and unknown clips return -1", () => {
    const world = new World();
    const clip = registerClip(world, new Float32Array([0.5]));
    expect(play(world, -1)).toBe(-1);
    const eid = play(world, clip, { position: [1, 2, 3], volume: 0.4, loop: true });
    expect(eid).toBeGreaterThan(0);
    expect(world.has(eid, Sound)).toBe(true);
    expect(world.has(eid, Transform)).toBe(true);
    expect(world.has(eid, GlobalTransform)).toBe(true);
    expect(world.storage(Sound).clip.get(eid)).toBe(clip);
    expect(world.storage(Sound).loop.get(eid)).toBe(1);
    expect(world.storage(Sound).volume.get(eid)).toBeCloseTo(0.4);
    expect(world.storage(Sound).paused.get(eid)).toBe(0);
    world.dispose();
});

test("a suspended context drops a once Sound and leaves a loop pending", () => {
    const world = new World();
    const clip = registerClip(world, new Float32Array([0.25]));
    setup(world);
    world.resource(Devices).audio.context = "suspended";
    const once = play(world, clip);
    const loop = play(world, clip, { loop: true });
    reconcileAudio(world);
    expect(world.exists(once)).toBe(false);
    expect(world.exists(loop)).toBe(true);
    expect(world.resource(AudioRuntime).slots.filter(Boolean)).toHaveLength(0);
    world.dispose();
});

test("a restored loop reconciles to one voice while a finished once sound is discarded", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const loop = play(world, clip, { loop: true });
    const once = play(world, clip);
    const snapshot = world.snapshot();
    reconcileAudio(world);
    const sent = lastBatch(messages);
    const data = commands(sent);
    const onceHandle = data[2 + 1 * 9 + 1];
    returnBatch(world, messages, [onceHandle]);
    expect(world.exists(once)).toBe(false);

    world.restore(snapshot);
    returnBatch(world, messages);
    reconcileAudio(world);
    expect(world.exists(loop)).toBe(true);
    expect(world.exists(once)).toBe(false);
    expect(world.resource(AudioRuntime).slots.filter(Boolean)).toHaveLength(1);
    expect(voiceFor(world, loop)).toBeDefined();
    const restored = commands(lastBatch(messages));
    expect(restored[0]).toBe(1);
    expect(restored[2]).toBe(2);
    world.dispose();
});

test("a once Sound restored after many later completions is discarded without replay", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const oldEid = play(world, clip);
    const oldSnapshot = world.snapshot();

    reconcileAudio(world);
    let data = commands(lastBatch(messages));
    returnAudioBuffer(world, data, [data[3]]);
    messages.length = 0;

    for (let i = 0; i < 65; i++) {
        play(world, clip);
        reconcileAudio(world);
        data = commands(lastBatch(messages));
        returnAudioBuffer(world, data, [data[3]]);
        messages.length = 0;
    }

    world.restore(oldSnapshot);
    expect(world.exists(oldEid)).toBe(false);
    expect(world.resource(AudioRuntime).slots.filter(Boolean)).toHaveLength(0);
    reconcileAudio(world);
    expect(commands(lastBatch(messages))[0]).toBe(0);
    world.dispose();
});

test("a once Sound with a live matching voice keeps that voice across restore", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const eid = play(world, clip);
    const snapshot = world.snapshot();
    reconcileAudio(world);
    const voice = voiceFor(world, eid)!;
    returnBatch(world, messages);

    world.restore(snapshot);
    expect(world.exists(eid)).toBe(true);
    reconcileAudio(world);
    expect(voiceFor(world, eid)).toBe(voice);
    const update = commands(lastBatch(messages));
    expect(update[0]).toBe(1);
    expect(update[2]).toBe(2);
    expect(update[3]).toBe(voice.handle);
    world.dispose();
});

test("a once Sound played after restore starts normally", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const snapshot = world.snapshot();
    world.restore(snapshot);

    const eid = play(world, clip);
    reconcileAudio(world);
    const data = commands(lastBatch(messages));
    expect(world.exists(eid)).toBe(true);
    expect(data[0]).toBe(1);
    expect(data[2]).toBe(1);
    expect(voiceFor(world, eid)).toBeDefined();
    world.dispose();
});

test("removing a loop fades and frees its voice", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const eid = play(world, clip, { loop: true });
    reconcileAudio(world);
    const voice = voiceFor(world, eid)!;
    returnBatch(world, messages);
    world.remove(eid, Sound);
    reconcileAudio(world);
    const stopped = commands(lastBatch(messages));
    expect(stopped[0]).toBe(1);
    expect(stopped[2]).toBe(3);
    expect(voice.stopping).toBe(true);
    returnBatch(world, messages, [voice.handle]);
    expect(voiceFor(world, eid)).toBeUndefined();
    expect(world.resource(AudioRuntime).slots[voice.handle % 64]).toBeNull();
    expect(world.resource(AudioRuntime).free).toContain(voice.handle % 64);
    world.dispose();
});

test("the positional command carries listener-relative azimuth for a source at the listener's left", () => {
    const world = new World();
    const messages = setup(world);
    const clip = registerClip(world, new Float32Array([0.25]));
    const listener = world.create();
    world.add(listener, Listener);
    world.add(listener, GlobalTransform);
    const source = play(world, clip, { position: [-1, 0, 0] });
    const transforms = world.storage(GlobalTransform);
    transforms.translation.set(listener, 0, 0, 0, 0);
    transforms.rotation.set(listener, 0, 0, 0, 1);
    transforms.scale.set(listener, 1, 1, 1, 0);
    transforms.translation.set(source, -1, 0, 0, 0);
    transforms.rotation.set(source, 0, 0, 0, 1);
    transforms.scale.set(source, 1, 1, 1, 0);
    reconcileAudio(world);
    const data = commands(lastBatch(messages));
    expect(data[0]).toBe(1);
    expect(data[2 + 5]).toBe(1);
    expect(data[2 + 6]).toBeCloseTo(-Math.PI / 2);
    expect(world.has(source, GlobalTransform)).toBe(true);
    world.dispose();
});

test("a context suspending drops a once Sound at its statechange, before the next frame", async () => {
    const globals = globalThis as Record<string, unknown>;
    const previous = {
        AudioContext: globals.AudioContext,
        AudioWorkletNode: globals.AudioWorkletNode,
    };
    let onStateChange: (() => void) | undefined;
    const node = () => ({ connect() {}, disconnect() {} });
    class FakeContext {
        state = "running";
        sampleRate = 48000;
        destination = node();
        audioWorklet = { addModule: async () => {} };
        addEventListener(type: string, listener: () => void) {
            if (type === "statechange") onStateChange = listener;
        }
        removeEventListener() {}
        createConstantSource() {
            return { ...node(), offset: { value: 1 }, start() {}, stop() {} };
        }
        async close() {}
    }
    class FakeWorkletNode {
        onprocessorerror: unknown = null;
        port = {
            onmessageerror: null as unknown,
            postMessage() {},
            listener: null as ((event: { data: unknown }) => void) | null,
            get onmessage() {
                return this.listener;
            },
            set onmessage(listener: ((event: { data: unknown }) => void) | null) {
                const first = this.listener === null && listener !== null;
                this.listener = listener;
                if (first) queueMicrotask(() => listener!({ data: { type: "ready" } }));
            },
        };
        connect() {}
        disconnect() {}
    }
    globals.AudioContext = FakeContext;
    globals.AudioWorkletNode = FakeWorkletNode;
    const world = new World();
    try {
        await initializeAudio(world);
        const context = world.resource(AudioRuntime).context as unknown as FakeContext;
        const clip = registerClip(world, new Float32Array([0.25]));
        const once = play(world, clip);
        const loop = play(world, clip, { loop: true });
        context.state = "suspended";
        onStateChange!();
        expect(world.resource(Devices).audio.context).toBe("suspended");
        expect(world.exists(once)).toBe(false);
        expect(world.exists(loop)).toBe(true);
    } finally {
        await disposeAudio(world);
        world.dispose();
        globals.AudioContext = previous.AudioContext;
        globals.AudioWorkletNode = previous.AudioWorkletNode;
    }
});
