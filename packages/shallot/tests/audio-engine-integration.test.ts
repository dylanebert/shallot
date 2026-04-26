import { describe, test, expect, beforeEach } from "bun:test";
import {
    createAudioState,
    type AudioState,
    alloc,
    free,
    assign,
    upload,
    refresh,
    onIdle,
    handleVoiceIdle,
    allocTransport,
    freeTransport,
    setBPM,
    play,
} from "../src/standard/audio/engine";
import { instrument } from "../src/standard/audio/instrument";
import type { AudioBackend, AudioCommand, Readback } from "../src/standard/audio/backend";
import { getParamPairs, setValues, clearInstruments } from "../src/standard/audio/instrument";
import { clearPatterns } from "../src/standard/audio/pattern";

function createSpyBackend(): AudioBackend & { calls: AudioCommand[]; handler: Readback | null } {
    const calls: AudioCommand[] = [];
    let handler: Readback | null = null;
    return {
        calls,
        handler,
        running: true,
        async init(h: Readback) {
            handler = h;
        },
        dispose() {},
        send(cmd) {
            calls.push(cmd);
        },
        pollReadback() {},
        flush() {},
    };
}

const filterGraph = {
    nodes: {
        osc: { type: "oscillator" as const },
        filter: { type: "filter" as const, input: "osc" },
        env: { type: "envelope" as const, input: "filter" },
        vol: { type: "gain" as const, input: "env" },
    },
    output: "vol",
    values: {
        "osc.waveform": 1,
        "filter.cutoff": 4500,
        "filter.q": 2.5,
        "filter.mode": 1,
        "filter.mix": 1,
        "env.attack": 0.001,
        "env.decay": 0.12,
        "env.sustain": 0,
        "env.release": 0.08,
        "vol.level": 0.5,
    },
};

const noFilterGraph = {
    nodes: {
        osc: { type: "oscillator" as const },
        env: { type: "envelope" as const, input: "osc" },
        vol: { type: "gain" as const, input: "env" },
    },
    output: "vol",
};

describe("engine voice tracking through play/seek/edit", () => {
    let audio: AudioState;
    let spy: ReturnType<typeof createSpyBackend>;

    beforeEach(() => {
        clearInstruments();
        clearPatterns();

        audio = createAudioState();
        spy = createSpyBackend();
        audio.backend = spy;
    });

    test("upload reaches voices after play", () => {
        const instId = instrument(filterGraph, "synth");

        const slot = alloc(audio);
        assign(audio, slot, instId);

        spy.calls.length = 0;
        upload(audio, instId, { "filter.cutoff": 1000 });

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBe(1);
        const changes = (paramCalls[0] as { type: "params"; changes: [number, number, number][] })
            .changes;
        expect(changes.length).toBeGreaterThan(0);
        expect(changes[0][0]).toBe(slot);
    });

    test("upload reaches voices after free + realloc", () => {
        const instId = instrument(filterGraph, "synth");

        const slot1 = alloc(audio);
        assign(audio, slot1, instId);
        free(audio, slot1);

        const slot2 = alloc(audio);
        assign(audio, slot2, instId);

        spy.calls.length = 0;
        upload(audio, instId, { "filter.cutoff": 1000 });

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBe(1);
        const changes = (paramCalls[0] as { type: "params"; changes: [number, number, number][] })
            .changes;
        expect(changes.some((c) => c[0] === slot2)).toBe(true);
    });

    test("updateValues + upload reaches voices after free + realloc", () => {
        const instId = instrument(filterGraph, "synth");

        const slot1 = alloc(audio);
        assign(audio, slot1, instId);
        free(audio, slot1);

        const slot2 = alloc(audio);
        assign(audio, slot2, instId);

        spy.calls.length = 0;
        setValues(instId, { "filter.cutoff": 1000 });
        upload(audio, instId, { "filter.cutoff": 1000 });
        refresh(audio, instId);

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBeGreaterThan(0);
        const allChanges = paramCalls.flatMap(
            (c) => (c as { type: "params"; changes: [number, number, number][] }).changes,
        );
        expect(allChanges.some((c) => c[0] === slot2)).toBe(true);
    });

    test("refresh reaches voices after free + realloc", () => {
        instrument(filterGraph, "synth");

        const slot1 = alloc(audio);
        assign(audio, slot1, 0);
        free(audio, slot1);

        const slot2 = alloc(audio);
        assign(audio, slot2, 0);

        spy.calls.length = 0;
        const instId = instrument(
            { ...filterGraph, values: { ...filterGraph.values, "filter.cutoff": 999 } },
            "synth",
        );
        refresh(audio, instId);

        const instCalls = spy.calls.filter((c) => c.type === "set_voice_instrument");
        expect(instCalls.length).toBe(1);
        expect((instCalls[0] as { type: "set_voice_instrument"; voiceId: number }).voiceId).toBe(
            slot2,
        );
    });

    test("registerInstrument sends set_voice_instrument on version change", () => {
        const instId = instrument(filterGraph, "synth");

        const slot = alloc(audio);
        assign(audio, slot, instId);

        spy.calls.length = 0;
        const newId = instrument(noFilterGraph, "synth");
        refresh(audio, newId);

        const instCalls = spy.calls.filter((c) => c.type === "set_voice_instrument");
        expect(instCalls.length).toBe(1);
        const call = instCalls[0] as {
            type: "set_voice_instrument";
            voiceId: number;
            instrumentId: number;
        };
        expect(call.voiceId).toBe(slot);
        expect(call.instrumentId).toBe(instId);
    });

    test("onVoiceIdle frees voice when no callback registered", () => {
        const instId = instrument(filterGraph, "synth");
        const slot = alloc(audio);
        assign(audio, slot, instId);

        handleVoiceIdle(audio, slot);

        spy.calls.length = 0;
        upload(audio, instId, { "filter.cutoff": 1000 });

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBe(0);
    });

    test("refresh after onVoiceIdle has freed voice: no set_voice_instrument", () => {
        const instId = instrument(filterGraph, "synth");
        const slot = alloc(audio);
        assign(audio, slot, instId);

        handleVoiceIdle(audio, slot);

        spy.calls.length = 0;
        const newId = instrument(noFilterGraph, "synth");
        refresh(audio, newId);

        const instCalls = spy.calls.filter((c) => c.type === "set_voice_instrument");
        expect(instCalls.length).toBe(0);
    });

    test("sendValues after refresh sends correct param offsets for new topology", () => {
        const instId = instrument(filterGraph, "synth");

        const slot = alloc(audio);
        assign(audio, slot, instId);

        const newId = instrument(noFilterGraph, "synth");
        refresh(audio, newId);
        spy.calls.length = 0;

        const values = getParamPairs(0);
        expect(values.length).toBeGreaterThan(0);

        setValues(0, { "env.attack": 0.5 });
        upload(audio, 0, { "env.attack": 0.5 });

        const afterCalls = spy.calls.filter((c) => c.type === "params");
        const allChanges = afterCalls.flatMap(
            (c) => (c as { type: "params"; changes: [number, number, number][] }).changes,
        );
        expect(allChanges.length).toBeGreaterThan(0);
    });
});

describe("voice pool behavior", () => {
    let audio: AudioState;
    let spy: ReturnType<typeof createSpyBackend>;

    beforeEach(() => {
        clearInstruments();
        clearPatterns();
        audio = createAudioState();
        spy = createSpyBackend();
        audio.backend = spy;
    });

    test("alloc_at_capacity_returns_negative_one", () => {
        for (let i = 0; i < 64; i++) {
            expect(alloc(audio)).not.toBe(-1);
        }
        expect(alloc(audio)).toBe(-1);
    });

    test("double_free_safe", () => {
        const slot = alloc(audio);
        free(audio, slot);
        free(audio, slot);
        const next = alloc(audio);
        expect(next).toBeGreaterThanOrEqual(0);
        const next2 = alloc(audio);
        expect(next2).toBeGreaterThanOrEqual(0);
    });

    test("free_clears_idle_callback", () => {
        const instId = instrument(filterGraph, "synth");
        const slot = alloc(audio);
        assign(audio, slot, instId);

        let called = 0;
        onIdle(audio, slot, () => {
            called++;
        });
        free(audio, slot);

        handleVoiceIdle(audio, slot);
        expect(called).toBe(0);
    });

    test("idle_callback_fires_once", () => {
        const instId = instrument(filterGraph, "synth");
        const slot = alloc(audio);
        assign(audio, slot, instId);

        let called = 0;
        onIdle(audio, slot, () => {
            called++;
        });

        handleVoiceIdle(audio, slot);
        handleVoiceIdle(audio, slot);
        expect(called).toBe(1);
    });

    test("idle_without_callback_returns_to_pool", () => {
        const instId = instrument(filterGraph, "synth");
        const slot = alloc(audio);
        assign(audio, slot, instId);

        handleVoiceIdle(audio, slot);

        spy.calls.length = 0;
        upload(audio, instId, { "filter.cutoff": 1000 });
        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBe(0);
    });
});

describe("voice generation counter", () => {
    let audio: AudioState;
    let spy: ReturnType<typeof createSpyBackend>;

    beforeEach(() => {
        clearInstruments();
        clearPatterns();
        audio = createAudioState();
        spy = createSpyBackend();
        audio.backend = spy;
    });

    test("stale idle callback does not corrupt recycled slot", () => {
        const instId = instrument(filterGraph, "synth");

        const slot = alloc(audio);
        assign(audio, slot, instId);

        let oldCbCalled = false;
        onIdle(audio, slot, () => {
            oldCbCalled = true;
        });

        free(audio, slot);

        const slot2 = alloc(audio);
        expect(slot2).toBe(slot);
        assign(audio, slot2, instId);

        handleVoiceIdle(audio, slot);

        expect(oldCbCalled).toBe(false);
        const deactivateCalls = spy.calls.filter(
            (c) => c.type === "voice_active" && !(c as any).active && (c as any).voiceId === slot,
        );
        const activateCalls = spy.calls.filter(
            (c) => c.type === "voice_active" && (c as any).active && (c as any).voiceId === slot,
        );
        expect(activateCalls.length).toBe(2);
        expect(deactivateCalls.length).toBe(1);
    });
});

describe("transport lifecycle", () => {
    let audio: AudioState;
    let spy: ReturnType<typeof createSpyBackend>;

    beforeEach(() => {
        clearInstruments();
        clearPatterns();
        audio = createAudioState();
        spy = createSpyBackend();
        audio.backend = spy;
    });

    test("freeTransport sends stop and clear_events", () => {
        const tid = allocTransport(audio);
        setBPM(audio, tid, 120);
        play(audio, tid);

        spy.calls.length = 0;
        freeTransport(audio, tid);

        const stopCalls = spy.calls.filter((c) => c.type === "transport_stop");
        const clearCalls = spy.calls.filter((c) => c.type === "transport_clear_events");
        expect(stopCalls.length).toBe(1);
        expect(clearCalls.length).toBe(1);
        expect((stopCalls[0] as any).tid).toBe(tid);
        expect((clearCalls[0] as any).tid).toBe(tid);
    });

    test("freed transport can be reallocated", () => {
        const tid1 = allocTransport(audio);
        freeTransport(audio, tid1);
        const tid2 = allocTransport(audio);
        expect(tid2).toBe(tid1);
    });
});
