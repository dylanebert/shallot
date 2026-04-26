import { describe, test, expect, beforeEach } from "bun:test";
import { build } from "../src/engine";
import { State } from "../src/engine/ecs/state";
import { AudioPlugin, Sound, Listener } from "../src/standard/audio";
import { Audio, type AudioState, alloc, free, handleVoiceIdle } from "../src/standard/audio/engine";
import { noteFreq } from "../src/standard/audio/pattern";
import { instrument, instrumentRegistry, getValues } from "../src/standard/audio/instrument";
import type { AudioBackend, AudioCommand } from "../src/standard/audio/backend";
import { Transform, TransformsPlugin } from "../src/standard/transforms";

function createSpyBackend(): AudioBackend & { calls: AudioCommand[] } {
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

const testGraph = {
    nodes: {
        osc: { type: "oscillator" as const },
        env: { type: "envelope" as const, input: "osc" },
    },
    output: "env",
};

function addSound(
    state: State,
    eid: number,
    values: Partial<{
        instrument: number;
        loop: number;
        volume: number;
        pitch: number;
        spatial: number;
    }>,
) {
    state.addComponent(eid, Sound);
    if (values.instrument !== undefined) Sound.instrument[eid] = values.instrument;
    if (values.loop !== undefined) Sound.loop[eid] = values.loop;
    if (values.volume !== undefined) Sound.volume[eid] = values.volume;
    if (values.pitch !== undefined) Sound.pitch[eid] = values.pitch;
    if (values.spatial !== undefined) Sound.spatial[eid] = values.spatial;
}

async function setup() {
    const state = new State();
    state.register(AudioPlugin);
    await AudioPlugin.initialize!(state);
    const audio = Audio.from(state)!;
    const spy = createSpyBackend();
    audio.backend = spy;
    const instId = instrument(testGraph, "test");
    return { state, audio, spy, instId };
}

describe("Sound component lifecycle", () => {
    let state: State;
    let audio: AudioState;
    let spy: ReturnType<typeof createSpyBackend>;
    let instId: number;

    beforeEach(async () => {
        ({ state, audio, spy, instId } = await setup());
    });

    test("add Sound allocates voice on next update", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const activeCalls = spy.calls.filter((c) => c.type === "voice_active" && (c as any).active);
        expect(activeCalls.length).toBe(1);

        const instCalls = spy.calls.filter((c) => c.type === "set_voice_instrument");
        expect(instCalls.length).toBe(1);

        const gateCalls = spy.calls.filter((c) => c.type === "gate" && (c as any).value === 1);
        expect(gateCalls.length).toBe(1);
    });

    test("remove Sound gates off and watches idle", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        spy.calls.length = 0;
        state.removeComponent(eid, Sound);

        const gateOffCalls = spy.calls.filter((c) => c.type === "gate" && (c as any).value === 0);
        expect(gateOffCalls.length).toBe(1);

        const watchCalls = spy.calls.filter((c) => c.type === "watch_idle");
        expect(watchCalls.length).toBe(1);
    });

    test("non-looping destroys entity on idle", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const gateCalls = spy.calls.filter((c) => c.type === "gate" && (c as any).value === 1);
        const slot = (gateCalls[0] as any).voiceId;

        handleVoiceIdle(audio, slot);

        expect(state.entityExists(eid)).toBe(false);
    });

    test("looping voice stays until removal", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1 });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const watchCalls = spy.calls.filter((c) => c.type === "watch_idle");
        expect(watchCalls.length).toBe(0);

        spy.calls.length = 0;
        state.removeComponent(eid, Sound);

        const gateOffCalls = spy.calls.filter((c) => c.type === "gate" && (c as any).value === 0);
        expect(gateOffCalls.length).toBe(1);
    });

    test("pool full queues, processes on free", () => {
        const directSlots: number[] = [];
        for (let i = 0; i < 64; i++) {
            const s = alloc(audio);
            expect(s).not.toBe(-1);
            directSlots.push(s);
        }

        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const activeBeforeFree = spy.calls.filter(
            (c) =>
                c.type === "voice_active" &&
                (c as any).active &&
                !directSlots.includes((c as any).voiceId),
        );
        expect(activeBeforeFree.length).toBe(0);

        free(audio, directSlots[0]);
        spy.calls.length = 0;
        sys.update!(state);

        const activeAfterFree = spy.calls.filter(
            (c) => c.type === "voice_active" && (c as any).active,
        );
        expect(activeAfterFree.length).toBe(1);
    });

    test("rapid add/remove no leak", () => {
        const sys = AudioPlugin.systems![0];

        for (let i = 0; i < 5; i++) {
            const eid = state.addEntity();
            addSound(state, eid, { instrument: instId });
            sys.update!(state);
            state.removeComponent(eid, Sound);
        }

        const activates = spy.calls.filter((c) => c.type === "voice_active" && (c as any).active);
        expect(activates.length).toBe(5);

        for (const call of spy.calls.filter((c) => c.type === "watch_idle")) {
            handleVoiceIdle(audio, (call as any).voiceId);
        }
        const totalDeactivates = spy.calls.filter(
            (c) => c.type === "voice_active" && !(c as any).active,
        );
        expect(totalDeactivates.length).toBe(5);
    });

    test("system removal doesn't double-free", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const gateCalls = spy.calls.filter((c) => c.type === "gate" && (c as any).value === 1);
        const slot = (gateCalls[0] as any).voiceId;

        spy.calls.length = 0;
        handleVoiceIdle(audio, slot);

        const deactivates = spy.calls.filter(
            (c) => c.type === "voice_active" && !(c as any).active,
        );
        expect(deactivates.length).toBe(1);
    });

    test("pending entity removed before allocation", () => {
        const directSlots: number[] = [];
        for (let i = 0; i < 64; i++) {
            directSlots.push(alloc(audio));
        }

        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId });

        state.removeComponent(eid, Sound);

        free(audio, directSlots[0]);
        spy.calls.length = 0;

        const sys = AudioPlugin.systems![0];
        sys.update!(state);

        const activeAfter = spy.calls.filter((c) => c.type === "voice_active" && (c as any).active);
        expect(activeAfter.length).toBe(0);
    });
});

const paramGraph = {
    nodes: {
        osc: { type: "oscillator" as const },
        env: { type: "envelope" as const, input: "osc" },
        vol: { type: "gain" as const, input: "env" },
    },
    output: "vol",
    volumeParam: "vol.level",
    pitchParams: ["osc.frequency"],
};

async function setupWithParams() {
    const state = new State();
    state.register(AudioPlugin);
    await AudioPlugin.initialize!(state);
    const audio = Audio.from(state)!;
    const spy = createSpyBackend();
    audio.backend = spy;
    const instId = instrument(paramGraph, "paramtest");
    return { state, audio, spy, instId };
}

describe("Sound param upload", () => {
    let state: State;
    let spy: ReturnType<typeof createSpyBackend>;
    let instId: number;
    const sys = AudioPlugin.systems![0];

    beforeEach(async () => {
        ({ state, spy, instId } = await setupWithParams());
    });

    test("volume param sent on update with quadratic curve", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1, volume: 0.5 });
        sys.update!(state);

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        const inst = instrumentRegistry.get(instId)!;
        const volOffset = inst.paramLayout.get("vol.level")!;
        const volumeParams = paramCalls.filter(
            (c) =>
                c.type === "params" &&
                (c as any).changes.some(
                    (ch: number[]) => ch[1] === volOffset && Math.abs(ch[2] - 0.25) < 0.001,
                ),
        );
        expect(volumeParams.length).toBeGreaterThanOrEqual(1);
    });

    test("pitch param sent on update", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1, pitch: 12 });
        sys.update!(state);

        const inst = instrumentRegistry.get(instId)!;
        const freqOffset = inst.paramLayout.get("osc.frequency")!;
        const vals = getValues(instId)!;
        const baseFreq = vals.get("osc.frequency")!;
        const expectedFreq = noteFreq(baseFreq, 0, 12, 0);

        const paramCalls = spy.calls.filter(
            (c) =>
                c.type === "params" &&
                (c as any).changes.some(
                    (ch: number[]) => ch[1] === freqOffset && Math.abs(ch[2] - expectedFreq) < 0.01,
                ),
        );
        expect(paramCalls.length).toBeGreaterThanOrEqual(1);
        expect(expectedFreq).toBeCloseTo(baseFreq * 2, 5);
    });

    test("no volumeParam means no volume param sent", () => {
        const noVolGraph = {
            nodes: {
                osc: { type: "oscillator" as const },
                env: { type: "envelope" as const, input: "osc" },
            },
            output: "env",
        };
        const plainId = instrument(noVolGraph, "plain");
        const eid = state.addEntity();
        addSound(state, eid, { instrument: plainId, loop: 1, volume: 0.5 });
        sys.update!(state);

        spy.calls.length = 0;
        sys.update!(state);

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        expect(paramCalls.length).toBe(0);
    });

    test("multi-osc pitchParams sets all oscillator frequencies", () => {
        const dualOscGraph = {
            nodes: {
                osc1: { type: "oscillator" as const },
                osc2: { type: "oscillator" as const },
                mix: { type: "mix" as const, input: "osc1", inputB: "osc2" },
                env: { type: "envelope" as const, input: "mix" },
                vol: { type: "gain" as const, input: "env" },
            },
            output: "vol",
            volumeParam: "vol.level",
            pitchParams: ["osc1.frequency", "osc2.frequency"],
        };
        const dualId = instrument(dualOscGraph, "dual-osc");
        const eid = state.addEntity();
        addSound(state, eid, { instrument: dualId, loop: 1, pitch: 12 });
        sys.update!(state);

        const inst = instrumentRegistry.get(dualId)!;
        const offset1 = inst.paramLayout.get("osc1.frequency")!;
        const offset2 = inst.paramLayout.get("osc2.frequency")!;
        const vals = getValues(dualId)!;
        const base1 = vals.get("osc1.frequency")!;
        const base2 = vals.get("osc2.frequency")!;
        const expected1 = noteFreq(base1, 0, 12, 0);
        const expected2 = noteFreq(base2, 0, 12, 0);

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        const hasOsc1 = paramCalls.some((c) =>
            (c as any).changes.some(
                (ch: number[]) => ch[1] === offset1 && Math.abs(ch[2] - expected1) < 0.01,
            ),
        );
        const hasOsc2 = paramCalls.some((c) =>
            (c as any).changes.some(
                (ch: number[]) => ch[1] === offset2 && Math.abs(ch[2] - expected2) < 0.01,
            ),
        );
        expect(hasOsc1).toBe(true);
        expect(hasOsc2).toBe(true);
    });

    test("per-oscillator pitch offsets applied to frequency", () => {
        const dualOscGraph = {
            nodes: {
                osc1: { type: "oscillator" as const },
                osc2: { type: "oscillator" as const },
                mix: { type: "mix" as const, input: "osc1", inputB: "osc2" },
                env: { type: "envelope" as const, input: "mix" },
                vol: { type: "gain" as const, input: "env" },
            },
            output: "vol",
            volumeParam: "vol.level",
            pitchParams: ["osc1.frequency", "osc2.frequency"],
            values: { "osc2.octave": 1, "osc2.semitone": 7 },
        };
        const dualId = instrument(dualOscGraph, "dual-offset");
        const eid = state.addEntity();
        addSound(state, eid, { instrument: dualId, loop: 1, pitch: 0 });
        sys.update!(state);

        const inst = instrumentRegistry.get(dualId)!;
        const offset1 = inst.paramLayout.get("osc1.frequency")!;
        const offset2 = inst.paramLayout.get("osc2.frequency")!;
        const vals = getValues(dualId)!;
        const base1 = vals.get("osc1.frequency")!;
        const base2 = vals.get("osc2.frequency")!;
        const expected1 = noteFreq(base1, 0, 0, 0);
        const expected2 = noteFreq(base2, 1, 7, 0);

        const paramCalls = spy.calls.filter((c) => c.type === "params");
        const hasOsc1 = paramCalls.some((c) =>
            (c as any).changes.some(
                (ch: number[]) => ch[1] === offset1 && Math.abs(ch[2] - expected1) < 0.01,
            ),
        );
        const hasOsc2 = paramCalls.some((c) =>
            (c as any).changes.some(
                (ch: number[]) => ch[1] === offset2 && Math.abs(ch[2] - expected2) < 0.01,
            ),
        );
        expect(hasOsc1).toBe(true);
        expect(hasOsc2).toBe(true);
        expect(expected2).toBeCloseTo(base2 * 2 ** (1 + 7 / 12), 2);
    });

    test("volume change between frames", () => {
        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1, volume: 1 });
        sys.update!(state);

        spy.calls.length = 0;
        Sound.volume[eid] = 0.3;
        sys.update!(state);

        const inst = instrumentRegistry.get(instId)!;
        const volOffset = inst.paramLayout.get("vol.level")!;
        const expected = 0.3 * 0.3;
        const paramCalls = spy.calls.filter(
            (c) =>
                c.type === "params" &&
                (c as any).changes.some(
                    (ch: number[]) => ch[1] === volOffset && Math.abs(ch[2] - expected) < 0.001,
                ),
        );
        expect(paramCalls.length).toBeGreaterThanOrEqual(1);
    });
});

async function setupSpatial() {
    const state = await build({
        plugins: [TransformsPlugin, AudioPlugin],
        defaults: false,
    });
    const audio = Audio.from(state)!;
    const spy = createSpyBackend();
    audio.backend = spy;
    const instId = instrument(paramGraph, "spatial-test");
    return { state, audio, spy, instId };
}

describe("Sound spatial", () => {
    const sys = AudioPlugin.systems![0];

    test("spatial voice sends addSpatial with listener", async () => {
        const { state, spy, instId } = await setupSpatial();

        const listener = state.addEntity();
        state.addComponent(listener, Listener);
        state.addComponent(listener, Transform);

        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1, spatial: 1 });
        state.addComponent(eid, Transform);
        Transform.posX[eid] = 5;

        state.step();
        sys.update!(state);

        const spatialCalls = spy.calls.filter((c) => c.type === "spatial");
        expect(spatialCalls.length).toBe(1);
    });

    test("no listener means no spatial sent", async () => {
        const { state, spy, instId } = await setupSpatial();

        const eid = state.addEntity();
        addSound(state, eid, { instrument: instId, loop: 1, spatial: 1 });
        state.addComponent(eid, Transform);

        state.step();
        sys.update!(state);

        const spatialCalls = spy.calls.filter((c) => c.type === "spatial");
        expect(spatialCalls.length).toBe(0);
    });
});
