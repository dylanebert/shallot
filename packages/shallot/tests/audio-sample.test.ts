import { describe, test, expect } from "bun:test";
import { State } from "../src/engine/ecs/state";
import { AudioPlugin } from "../src/standard/audio";
import { Audio, tickAudio } from "../src/standard/audio/engine";
import { instrument, instrumentRegistry } from "../src/standard/audio/instrument";
import { sample, downmixToMono } from "../src/standard/audio/sample";
import type { AudioBackend, AudioCommand } from "../src/standard/audio/backend";

function spyBackend(): AudioBackend & { calls: AudioCommand[] } {
    const calls: AudioCommand[] = [];
    return {
        calls,
        running: true,
        async init() {},
        dispose() {},
        send(cmd) {
            calls.push(cmd);
        },
        pollReadback() {},
        flush() {},
    };
}

async function setup() {
    const state = new State();
    state.register(AudioPlugin);
    await AudioPlugin.initialize!(state);
    const audio = Audio.from(state)!;
    const spy = spyBackend();
    audio.backend = spy;
    return { state, audio, spy };
}

describe("sample registration", () => {
    test("sample() returns an id and flushes a set_sample command on tick", async () => {
        const { audio, spy } = await setup();
        const data = new Float32Array([0.0, 0.25, 0.5, 0.75, 1.0]);
        const id = sample(data, "audio-sample-test/ramp");
        expect(id).toBeGreaterThanOrEqual(0);

        tickAudio(audio);

        const sampleCmds = spy.calls.filter((c) => c.type === "set_sample") as Extract<
            AudioCommand,
            { type: "set_sample" }
        >[];
        expect(sampleCmds.length).toBeGreaterThanOrEqual(1);
        const cmd = sampleCmds.find((c) => c.id === id)!;
        expect(cmd.data).toEqual(data);
    });

    test("subsequent ticks do not resend the same sample", async () => {
        const { audio, spy } = await setup();
        const data = new Float32Array([1.0, 0.5, 0.0]);
        sample(data, "audio-sample-test/dedup");

        tickAudio(audio);
        const beforeCount = spy.calls.filter((c) => c.type === "set_sample").length;

        tickAudio(audio);
        const afterCount = spy.calls.filter((c) => c.type === "set_sample").length;

        expect(afterCount).toBe(beforeCount);
    });

    test("downmix: mono passthrough copies the buffer", () => {
        const src = new Float32Array([0, 0.5, -0.5, 1, -1]);
        const out = downmixToMono([src], src.length);
        expect(Array.from(out)).toEqual(Array.from(src));
        expect(out).not.toBe(src);
    });

    test("downmix: stereo averages channels", () => {
        const L = new Float32Array([1, 1, 0, -1]);
        const R = new Float32Array([1, -1, 0, 1]);
        const out = downmixToMono([L, R], L.length);
        expect(Array.from(out)).toEqual([1, 0, 0, 0]);
    });

    test("downmix: anti-phase signals cancel", () => {
        const L = new Float32Array([0.7, -0.3, 0.5]);
        const R = new Float32Array([-0.7, 0.3, -0.5]);
        const out = downmixToMono([L, R], L.length);
        for (const v of out) expect(Math.abs(v)).toBeLessThan(1e-7);
    });

    test("downmix: N channels divide by N", () => {
        const a = new Float32Array([3]);
        const b = new Float32Array([3]);
        const c = new Float32Array([3]);
        expect(downmixToMono([a, b, c], 1)[0]).toBeCloseTo(3, 6);
    });

    test("Sample node compiles into instrument param layout", () => {
        const id = instrument(
            {
                nodes: { src: { type: "sample" } },
                output: "src",
                values: { "src.bufferId": 7, "src.rate": 1, "src.loop": 1, "src.volume": 0.8 },
            },
            "audio-sample-test/instrument",
        );
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.nodes.length).toBe(1);
        expect(compiled.nodes[0].type).toBe(7);
        expect(compiled.paramLayout.has("src.bufferId")).toBe(true);
        expect(compiled.paramLayout.has("src.rate")).toBe(true);
        expect(compiled.paramLayout.has("src.loop")).toBe(true);
        expect(compiled.paramLayout.has("src.volume")).toBe(true);
    });
});
