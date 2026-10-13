import { expect, test } from "bun:test";
import { AudioOutput, Clips, Listener, play, registerClip, Sound } from "../../core/audio";
import { audioContextState } from "../../core/input";
import { GlobalTransform, Transform } from "../../core/transform";
import { World } from "../../engine";
import { AudioRuntime, reconcileAudio, returnAudioBuffer } from "./runtime";

interface Message {
    type: string;
    id?: number;
    buffer?: ArrayBuffer;
}

function setup(world: World): Message[] {
    const messages: Message[] = [];
    const runtime = world.resource(AudioRuntime);
    runtime.context = { sampleRate: 48000 } as AudioContext;
    runtime.node = {
        port: {
            postMessage(message: Message) {
                messages.push(message);
            },
        },
    } as unknown as AudioWorkletNode;
    world.resource(AudioOutput).context = runtime.context;
    audioContextState(world, "running");
    return messages;
}

function returnBatch(world: World, messages: Message[], finished: readonly number[] = []): void {
    const message = [...messages].reverse().find((entry) => entry.type === "batch");
    if (!message?.buffer) throw new Error("audio frame did not send a command buffer");
    returnAudioBuffer(world, new Float64Array(message.buffer), finished);
}

function commands(message: Message): Float64Array {
    if (!message.buffer) throw new Error("message has no command buffer");
    return new Float64Array(message.buffer);
}

function lastBatch(messages: Message[]): Message {
    const message = [...messages].reverse().find((entry) => entry.type === "batch");
    if (!message) throw new Error("audio frame did not send a command buffer");
    return message;
}

test("each World owns its clip registry and uploads each clip version once to its own worklet", () => {
    const first = new World();
    const second = new World();
    const firstMessages = setup(first);
    const secondMessages = setup(second);
    const firstId = registerClip(first, "shared-name", new Float32Array([0.25]));
    const secondId = registerClip(second, "shared-name", new Float32Array([0.75]));
    expect(first.resource(Clips)).not.toBe(second.resource(Clips));
    expect(first.resource(Clips).registry.get("shared-name")?.channels[0][0]).toBe(0.25);
    expect(second.resource(Clips).registry.get("shared-name")?.channels[0][0]).toBe(0.75);

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

    registerClip(first, "shared-name", new Float32Array([0.5]));
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
    const clip = registerClip(world, "positioned", new Float32Array([0.5]));
    expect(play(world, "not-registered")).toBe(-1);
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
    registerClip(world, "suspended", new Float32Array([0.25]));
    setup(world);
    audioContextState(world, "suspended");
    const once = play(world, "suspended");
    const loop = play(world, "suspended", { loop: true });
    reconcileAudio(world);
    expect(world.exists(once)).toBe(false);
    expect(world.exists(loop)).toBe(true);
    expect(world.resource(AudioRuntime).owners.size).toBe(0);
    world.dispose();
});

test("restoring a snapshot reconciles one loop voice and no finished once voice", () => {
    const world = new World();
    const messages = setup(world);
    registerClip(world, "restored", new Float32Array([0.25]));
    const loop = play(world, "restored", { loop: true });
    const once = play(world, "restored");
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
    expect(world.resource(AudioRuntime).owners.size).toBe(1);
    expect(world.resource(AudioRuntime).owners.has(loop)).toBe(true);
    expect(world.resource(AudioRuntime).slots.filter(Boolean)).toHaveLength(1);
    const restored = commands(lastBatch(messages));
    expect(restored[0]).toBe(1);
    expect(restored[2]).toBe(2);
    world.dispose();
});

test("removing a loop fades and frees its voice", () => {
    const world = new World();
    const messages = setup(world);
    registerClip(world, "removed-loop", new Float32Array([0.25]));
    const eid = play(world, "removed-loop", { loop: true });
    reconcileAudio(world);
    const voice = world.resource(AudioRuntime).owners.get(eid)!;
    returnBatch(world, messages);
    world.remove(eid, Sound);
    reconcileAudio(world);
    const stopped = commands(lastBatch(messages));
    expect(stopped[0]).toBe(1);
    expect(stopped[2]).toBe(3);
    expect(voice.stopping).toBe(true);
    returnBatch(world, messages, [voice.handle]);
    expect(world.resource(AudioRuntime).owners.has(eid)).toBe(false);
    expect(world.resource(AudioRuntime).slots[voice.handle % 64]).toBeNull();
    expect(world.resource(AudioRuntime).free).toContain(voice.handle % 64);
    world.dispose();
});

test("the positional command carries listener-relative azimuth for a source at the listener's left", () => {
    const world = new World();
    const messages = setup(world);
    registerClip(world, "left", new Float32Array([0.25]));
    const listener = world.create();
    world.add(listener, Listener);
    world.add(listener, GlobalTransform);
    const source = play(world, "left", { position: [-1, 0, 0] });
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
