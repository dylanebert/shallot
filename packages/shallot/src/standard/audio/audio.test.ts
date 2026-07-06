import { beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import { State } from "../..";
import { clear, register } from "../../engine/ecs/core";
import { Transform } from "../transforms";
import { AudioPlugin, play, Sound, sfx } from "./";
import { Audio, alloc, byId, free, instrument, polar, slotOf, valid } from "./core";
import { markCooldown, withinCooldown } from "./policy";
import { flushSamples, keepChannels, resetSampleUploads, Samples } from "./sample";

// the identity listener basis: right +x, up +y, forward +z
const RIGHT = [1, 0, 0] as const;
const UP = [0, 1, 0] as const;
const FWD = [0, 0, 1] as const;

describe("voice allocator", () => {
    test("alloc returns a valid in-range handle; free invalidates it", () => {
        const h = alloc();
        expect(h).toBeGreaterThanOrEqual(0);
        expect(valid(h)).toBe(true);
        expect(slotOf(h)).toBeGreaterThanOrEqual(0);
        expect(slotOf(h)).toBeLessThan(64);
        free(h);
        expect(valid(h)).toBe(false);
    });

    test("re-claiming a slot invalidates the prior generation's handle", () => {
        const a = alloc();
        const slot = slotOf(a);
        free(a);
        // the free-list is LIFO, so the just-freed slot comes straight back
        const b = alloc();
        expect(slotOf(b)).toBe(slot);
        expect(valid(b)).toBe(true);
        expect(valid(a)).toBe(false); // stale handle no-ops
        free(b);
    });

    test("pool exhaustion returns -1 and recovers after a free", () => {
        const held: number[] = [];
        for (let i = 0; i < 64; i++) held.push(alloc());
        expect(held.every((h) => h >= 0)).toBe(true);
        expect(alloc()).toBe(-1);
        free(held[0]);
        const reused = alloc();
        expect(reused).toBeGreaterThanOrEqual(0);
        free(reused);
        for (let i = 1; i < 64; i++) free(held[i]);
    });

    test("the -1 sentinel is never valid", () => {
        expect(valid(-1)).toBe(false);
    });
});

describe("polar", () => {
    // each call reads its result immediately — polar returns a reused scratch.
    // every case is an exact special value — atan2/asin/sqrt of exact inputs
    // (±π/2, 0, 1, 5) — so the projection is bit-exact, no tolerance to derive.
    test("axis-aligned directions map to known azimuth / elevation / distance", () => {
        // straight ahead (+forward): no azimuth, no elevation
        expect(polar(0, 0, 1, ...RIGHT, ...UP, ...FWD).azimuth).toBe(0);
        expect(polar(0, 0, 1, ...RIGHT, ...UP, ...FWD).distance).toBe(1);
        // to the right (+x): azimuth +90°
        expect(polar(1, 0, 0, ...RIGHT, ...UP, ...FWD).azimuth).toBe(Math.PI / 2);
        // overhead (+y): elevation +90°
        expect(polar(0, 1, 0, ...RIGHT, ...UP, ...FWD).elevation).toBe(Math.PI / 2);
        // distance is the euclidean norm, basis-independent
        expect(polar(3, 4, 0, ...RIGHT, ...UP, ...FWD).distance).toBe(5);
    });

    test("a rotated listener basis re-frames the source", () => {
        // listener faces +x (forward = +x, right = -z), source dead ahead at +x
        expect(polar(1, 0, 0, 0, 0, -1, 0, 1, 0, 1, 0, 0).azimuth).toBe(0);
    });
});

describe("instrument compiler", () => {
    test("compiles a linear sampler chain in topological order with resolved offsets", () => {
        const id = instrument(
            {
                nodes: {
                    src: { type: "sample" },
                    env: { type: "envelope", input: "src" },
                    vol: { type: "gain", input: "env" },
                },
                output: "vol",
                volumeParam: "vol.level",
                loopParam: "src.loop",
                values: { "src.bufferId": 7, "vol.level": 0.5 },
            },
            "test-sampler",
        );
        const inst = byId(id);
        expect(inst).toBeDefined();
        if (!inst) return;
        expect(inst.paramLayout.has("src.bufferId")).toBe(true);
        expect(inst.paramLayout.has("env.attack")).toBe(true);
        expect(inst.paramLayout.has("vol.level")).toBe(true);
        // ECS-driven offsets resolved once at compile
        expect(inst.volumeOffsets).toEqual([inst.paramLayout.get("vol.level")!]);
        expect(inst.baseVolume).toBe(0.5);
        expect(inst.loopOffsets).toEqual([inst.paramLayout.get("src.loop")!]);
        // mono routes the left buffer into both bus channels
        expect(inst.outputBufR).toBe(inst.outputBuf);
        // sample(7) → envelope(3) → gain(4), dependencies before dependents
        expect(inst.nodes.map((n) => n.type)).toEqual([7, 3, 4]);
    });

    test("a stereo instrument resolves a per-chain offset and a distinct right buffer", () => {
        const id = instrument(
            {
                nodes: {
                    srcL: { type: "sample" },
                    volL: { type: "gain", input: "srcL" },
                    srcR: { type: "sample" },
                    volR: { type: "gain", input: "srcR" },
                },
                output: "volL",
                outputR: "volR",
                volumeParam: ["volL.level", "volR.level"],
                loopParam: ["srcL.loop", "srcR.loop"],
                values: { "srcR.channel": 1 },
            },
            "test-stereo",
        );
        const inst = byId(id);
        expect(inst).toBeDefined();
        if (!inst) return;
        expect(inst.volumeOffsets).toEqual([
            inst.paramLayout.get("volL.level")!,
            inst.paramLayout.get("volR.level")!,
        ]);
        expect(inst.loopOffsets).toEqual([
            inst.paramLayout.get("srcL.loop")!,
            inst.paramLayout.get("srcR.loop")!,
        ]);
        // distinct buffers feed the two bus channels — the baked width survives
        expect(inst.outputBufR).not.toBe(inst.outputBuf);
    });

    test("resolves pitch entries for an oscillator instrument", () => {
        const id = instrument(
            {
                nodes: { osc: { type: "oscillator" }, out: { type: "gain", input: "osc" } },
                output: "out",
                pitchParams: ["osc.frequency"],
                values: { "osc.frequency": 220 },
            },
            "test-osc",
        );
        const inst = byId(id);
        expect(inst).toBeDefined();
        if (!inst) return;
        expect(inst.pitchEntries.length).toBe(1);
        expect(inst.pitchEntries[0].offset).toBe(inst.paramLayout.get("osc.frequency")!);
        expect(inst.pitchEntries[0].baseFreq).toBe(220);
    });

    test("a cyclic graph throws", () => {
        expect(() =>
            instrument({
                nodes: { a: { type: "gain", input: "b" }, b: { type: "gain", input: "a" } },
                output: "a",
            }),
        ).toThrow(/cycle/);
    });
});

describe("sample decode channels", () => {
    test("preserves mono and stereo, collapses >2 channels to mono", () => {
        const l = new Float32Array([1, 0]);
        const r = new Float32Array([0, 1]);

        const mono = keepChannels([l], 2);
        expect(mono.length).toBe(1);
        expect([...mono[0]]).toEqual([1, 0]);

        const stereo = keepChannels([l, r], 2);
        expect(stereo.length).toBe(2);
        expect([...stereo[0]]).toEqual([1, 0]);
        expect([...stereo[1]]).toEqual([0, 1]);

        // 4 channels have no 2D placement → one downmixed (averaged) channel
        const quad = keepChannels(
            [
                new Float32Array([4, 0]),
                new Float32Array([0, 4]),
                new Float32Array([4, 0]),
                new Float32Array([0, 4]),
            ],
            2,
        );
        expect(quad.length).toBe(1);
        expect([...quad[0]]).toEqual([2, 2]);
    });

    test("keepChannels copies, never aliases the decoded buffers", () => {
        const src = new Float32Array([1, 2]);
        const out = keepChannels([src], 2);
        out[0][0] = 9;
        expect(src[0]).toBe(1);
    });

    test("flushSamples emits one upload per channel, then dedups by version", () => {
        resetSampleUploads();
        Samples.register({
            name: "flush-stereo",
            channels: [new Float32Array([1, 2]), new Float32Array([3, 4])],
            sampleRate: 48000,
            version: 1,
        });
        const id = Samples.id("flush-stereo")!;

        const got: [number, number, number][] = [];
        flushSamples((sid, channel, count) => {
            if (sid === id) got.push([sid, channel, count]);
        });
        // one upload per channel, each tagged with its index and the channel count
        expect(got).toEqual([
            [id, 0, 2],
            [id, 1, 2],
        ]);

        // a second flush re-sends nothing — version-tracked, not per-channel
        const again: number[] = [];
        flushSamples((sid) => {
            if (sid === id) again.push(sid);
        });
        expect(again).toEqual([]);
    });
});

type Cmd = { type: string; voiceId?: number; value?: number };

const MAX_VOICES = 64;

const TONE = {
    nodes: { osc: { type: "oscillator" }, out: { type: "gain", input: "osc" } },
    output: "out",
    values: { "osc.frequency": 440 },
} as const;

// reset the module-level Audio singleton and install a host stub: started() /
// running() then report ready and the stub captures the per-frame command batch.
// initAudio() can't stand in — headless it resets the fields then throws on
// `new AudioContext()`, and disposeAudio() calls node.disconnect() on the stub
function stubHost(): Cmd[] {
    Audio.node = null;
    Audio.ctx = null;
    Audio.queue.length = 0;
    Audio.idle.clear();
    Audio.sentInstruments.clear();
    Audio.free.length = 0;
    for (let i = MAX_VOICES - 1; i >= 0; i--) Audio.free.push(i);
    Audio.gen.fill(0);
    Audio.spatialLen = 0;

    const cmds: Cmd[] = [];
    Audio.node = {
        port: { postMessage: (m: { commands?: Cmd[] }) => m.commands && cmds.push(...m.commands) },
    } as unknown as AudioWorkletNode;
    Audio.ctx = { state: "running" } as unknown as AudioContext;
    return cmds;
}

function audioState(): State {
    clear();
    const state = new State();
    register("Transform", Transform); // Listener.requires + the spatial query resolve against it
    for (const [n, c] of Object.entries(AudioPlugin.components ?? {}))
        register(n, c, AudioPlugin.traits?.[n]);
    attach(state, AudioPlugin);
    return state;
}

describe("SoundSystem voice lifecycle", () => {
    let cmds: Cmd[];
    let state: State;

    beforeEach(() => {
        cmds = stubHost();
        state = audioState();
    });

    test("a one-shot queues its idle watch after gate-on, not before", () => {
        instrument(TONE, "tone-order");
        const eid = play(state, "tone-order");
        expect(eid).toBeGreaterThanOrEqual(0);
        state.step();

        const slot = slotOf(Sound.voice.get(eid));
        const gateOn = cmds.findIndex(
            (c) => c.type === "gate" && c.voiceId === slot && c.value !== 0,
        );
        const watch = cmds.findIndex((c) => c.type === "watch_idle" && c.voiceId === slot);
        expect(gateOn).toBeGreaterThanOrEqual(0);
        expect(watch).toBeGreaterThanOrEqual(0);
        // the worklet clears a slot from _releasing on any gate(value != 0), so a
        // watch_idle queued before the gate-on is wiped in the same batch and the
        // voice never reports idle — it leaks until steal reclaims it
        expect(watch).toBeGreaterThan(gateOn);
    });

    test("pool exhaustion steals the oldest voice so a new sound still plays", () => {
        instrument(TONE, "tone-steal");
        const Voiced = AudioPlugin.components!.Voiced;

        for (let i = 0; i < MAX_VOICES; i++) play(state, "tone-steal");
        state.step();
        expect(Audio.free.length).toBe(0);
        expect([...state.query([Sound, Voiced])].length).toBe(MAX_VOICES);

        const extra = play(state, "tone-steal");
        state.step();
        // the new sound claimed a stolen slot; the pool never exceeds its size
        expect(valid(Sound.voice.get(extra))).toBe(true);
        expect([...state.query([Sound, Voiced])].length).toBe(MAX_VOICES);
    });

    test("the global cull skips loops — a one-shot burst never steals the bed", () => {
        instrument(TONE, "tone-bed");

        // the bed is created first, so a gen-indifferent cull (a clean pool gives
        // every slot the same generation) would pick it without the loop guard
        const bed = play(state, "tone-bed", { loop: true });
        for (let i = 0; i < MAX_VOICES - 1; i++) play(state, "tone-bed");
        state.step();
        expect(Audio.free.length).toBe(0);

        const extra = play(state, "tone-bed"); // pool full → forces a cull
        state.step();
        // the cull took a one-shot; the looping bed keeps its voice
        expect(valid(Sound.voice.get(extra))).toBe(true);
        expect(state.exists(bed)).toBe(true);
        expect(valid(Sound.voice.get(bed))).toBe(true);
        expect(Sound.loop.get(bed)).toBe(1);
    });
});

describe("sfx policy", () => {
    test("cooldown predicate gates a re-trigger and self-heals a clock reset", () => {
        expect(withinCooldown("cd", 0.1, 0)).toBe(false); // no prior trigger
        markCooldown("cd", 0);
        expect(withinCooldown("cd", 0.1, 0.05)).toBe(true); // inside the window → drop
        expect(withinCooldown("cd", 0.1, 0.1)).toBe(false); // at the window edge → allowed
        markCooldown("cd", 10); // a later trigger
        // a State rebuild resets elapsed to 0 — a backwards clock reads as expired
        expect(withinCooldown("cd", 0.1, 0.05)).toBe(false);
        // a zero cooldown never gates
        expect(withinCooldown("cd-off", 0, 5)).toBe(false);
    });
});

describe("sfx policy enforcement", () => {
    let state: State;

    beforeEach(() => {
        stubHost();
        state = audioState();
    });

    test("a max cap is never exceeded by a same-frame burst", () => {
        const Voiced = AudioPlugin.components!.Voiced;
        instrument(TONE, "tone-cap");
        sfx("tone-cap", { max: 4 });

        for (let i = 0; i < 10; i++) play(state, "tone-cap");
        expect([...state.query([Sound])].length).toBe(4); // surplus triggers dropped
        state.step();
        expect([...state.query([Sound, Voiced])].length).toBe(4);
    });

    test("a cap with steal keeps a free voice for the new trigger", () => {
        const Voiced = AudioPlugin.components!.Voiced;
        instrument(TONE, "tone-cap-steal");
        sfx("tone-cap-steal", { max: 2, steal: "oldest" });

        play(state, "tone-cap-steal");
        play(state, "tone-cap-steal");
        state.step();
        expect([...state.query([Sound, Voiced])].length).toBe(2);

        const extra = play(state, "tone-cap-steal"); // at cap → culls a playing one
        state.step();
        expect(valid(Sound.voice.get(extra))).toBe(true);
        expect([...state.query([Sound, Voiced])].length).toBe(2); // cap held
    });

    test('steal "none" drops the new trigger at the cap, culling nothing', () => {
        instrument(TONE, "tone-cap-none");
        sfx("tone-cap-none", { max: 2, steal: "none" });

        const a = play(state, "tone-cap-none");
        const b = play(state, "tone-cap-none");
        state.step();

        const extra = play(state, "tone-cap-none");
        expect(extra).toBe(-1); // dropped
        expect(state.has(a, Sound)).toBe(true);
        expect(state.has(b, Sound)).toBe(true);
        expect([...state.query([Sound])].length).toBe(2);
    });

    test('steal "quietest" culls the lowest-volume instance', () => {
        instrument(TONE, "tone-cap-quiet");
        sfx("tone-cap-quiet", { max: 2, steal: "quietest" });

        const loud = play(state, "tone-cap-quiet", { volume: 0.9 });
        const quiet = play(state, "tone-cap-quiet", { volume: 0.2 });
        state.step();

        play(state, "tone-cap-quiet"); // at cap → the quietest yields
        expect(state.has(quiet, Sound)).toBe(false);
        expect(state.has(loud, Sound)).toBe(true);
        expect([...state.query([Sound])].length).toBe(2); // cap held
    });

    test("play() drops a re-trigger inside the cooldown window, then admits past it", () => {
        instrument(TONE, "tone-cd");
        sfx("tone-cd", { cooldown: 0.05 });

        expect(play(state, "tone-cd")).toBeGreaterThanOrEqual(0); // opens the window
        expect(play(state, "tone-cd")).toBe(-1); // same elapsed → inside the window
        // 0.06 s clears the 0.05 s window and stays under the ~67 ms step clamp,
        // so elapsed advances unclamped past it
        state.step(0.06);
        expect(play(state, "tone-cd")).toBeGreaterThanOrEqual(0);
    });
});

describe("stereo sampler", () => {
    let cmds: Cmd[];
    let state: State;

    beforeEach(() => {
        cmds = stubHost();
        state = audioState();
    });

    function registerSample(name: string, channels: Float32Array[]): void {
        Samples.register({ name, channels, sampleRate: 48000, version: 1 });
    }

    test("a 2-channel sample compiles two parallel chains; a mono one compiles a single chain", () => {
        registerSample("stereo-bed", [new Float32Array([1, 0]), new Float32Array([0, 1])]);
        const stereo = byId(Sound.instrument.get(play(state, "stereo-bed", { loop: true })))!;
        // distinct buffers feed the two bus channels — the baked stereo width survives
        expect(stereo.outputBufR).not.toBe(stereo.outputBuf);
        expect(stereo.volumeOffsets.length).toBe(2);
        expect(stereo.loopOffsets.length).toBe(2);

        registerSample("mono-bed", [new Float32Array([1, 0])]);
        const mono = byId(Sound.instrument.get(play(state, "mono-bed", { loop: true })))!;
        // mono routes the same buffer into both channels (today's behavior, unchanged)
        expect(mono.outputBufR).toBe(mono.outputBuf);
        expect(mono.volumeOffsets.length).toBe(1);
    });

    test("the per-frame firehose drives both gains of a stereo voice", () => {
        registerSample("stereo-fire", [new Float32Array([1, 0]), new Float32Array([0, 1])]);
        const eid = play(state, "stereo-fire", { loop: true, volume: 0.5 });
        const inst = byId(Sound.instrument.get(eid))!;
        state.step();

        const slot = slotOf(Sound.voice.get(eid));
        // level = v*v*baseVolume = 0.5*0.5*1 = 0.25, distinct from the static default 1,
        // so a match isolates the firehose write from the one-time assign
        const driven = new Set<number>();
        for (const c of cmds as { type: string; changes?: [number, number, number][] }[]) {
            if (c.type !== "params" || !c.changes) continue;
            for (const [s, off, val] of c.changes) {
                if (s === slot && Math.abs(val - 0.25) < 1e-6) driven.add(off);
            }
        }
        expect(driven.has(inst.volumeOffsets[0])).toBe(true);
        expect(driven.has(inst.volumeOffsets[1])).toBe(true);
    });
});
