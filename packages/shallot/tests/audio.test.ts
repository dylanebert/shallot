import { describe, test, expect, beforeEach } from "bun:test";
import { polar } from "../src/standard/audio/engine";
import { noteFreq } from "../src/standard/audio/pattern";
import {
    instrument,
    MAX_INSTRUMENTS,
    MAX_BUFFERS,
    type Instrument,
} from "../src/standard/audio/instrument";
import { evalCurve, type CurveMapping } from "../src/standard/audio/curve";
import {
    pattern,
    notes,
    midiFreq,
    type Pattern,
    type PatternTrack,
} from "../src/standard/audio/pattern";
import {
    instrumentRegistry,
    getValues,
    getParamPairs,
    clearInstruments,
} from "../src/standard/audio/instrument";
import { patternRegistry, clearPatterns } from "../src/standard/audio/pattern";

describe("noteFreq", () => {
    test("returns BASE_FREQ when base is 0", () => {
        const freq = noteFreq(0);
        expect(freq).toBeCloseTo(523.2511, 2);
    });

    test("returns base when positive", () => {
        expect(noteFreq(440)).toBeCloseTo(440, 4);
    });

    test("octave doubles frequency", () => {
        const f1 = noteFreq(440);
        const f2 = noteFreq(440, 1);
        expect(f2 / f1).toBeCloseTo(2, 4);
    });

    test("semitone shifts by 12th of octave", () => {
        const f1 = noteFreq(440, 0, 0);
        const f2 = noteFreq(440, 0, 12);
        expect(f2 / f1).toBeCloseTo(2, 4);
    });

    test("fine shifts by cents", () => {
        const f1 = noteFreq(440, 0, 0, 0);
        const f2 = noteFreq(440, 0, 0, 1200);
        expect(f2 / f1).toBeCloseTo(2, 4);
    });

    test("combined octave + semitone + fine", () => {
        const freq = noteFreq(440, 1, 7, 0);
        expect(freq).toBeCloseTo(440 * 2 ** (1 + 7 / 12), 2);
    });
});

describe("polar", () => {
    const Eps = 0.001;

    test("source directly ahead: azimuth 0, elevation 0", () => {
        const { azimuth, elevation, distance } = polar(0, 0, 5, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        expect(Math.abs(azimuth)).toBeLessThan(Eps);
        expect(Math.abs(elevation)).toBeLessThan(Eps);
        expect(Math.abs(distance - 5)).toBeLessThan(Eps);
    });

    test("source to the right: positive azimuth", () => {
        const { azimuth } = polar(5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        expect(azimuth).toBeGreaterThan(0);
        expect(Math.abs(azimuth - Math.PI / 2)).toBeLessThan(Eps);
    });

    test("source to the left: negative azimuth", () => {
        const { azimuth } = polar(-5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        expect(azimuth).toBeLessThan(0);
        expect(Math.abs(azimuth + Math.PI / 2)).toBeLessThan(Eps);
    });

    test("source above: positive elevation", () => {
        const { elevation } = polar(0, 5, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        expect(elevation).toBeGreaterThan(0);
        expect(Math.abs(elevation - Math.PI / 2)).toBeLessThan(Eps);
    });

    test("coincident source: zero distance, zero angles", () => {
        const { elevation, distance } = polar(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1);
        expect(distance).toBeLessThan(Eps);
        expect(Math.abs(elevation)).toBeLessThan(Eps);
    });
});

describe("instrument compilation", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });
    const defaultGraph: Instrument = {
        nodes: {
            osc: { type: "oscillator" },
            filter: { type: "filter", input: "osc" },
            env: { type: "envelope", input: "filter" },
        },
        output: "env",
    };

    const dualOscGraph: Instrument = {
        nodes: {
            osc1: { type: "oscillator" },
            osc2: { type: "oscillator" },
            mix: { type: "mix", input: "osc1", inputB: "osc2" },
            filter: { type: "filter", input: "mix" },
            env: { type: "envelope", input: "filter" },
        },
        output: "env",
    };

    test("paramLayout uses raw param names as keys", () => {
        const id = instrument(defaultGraph);
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.paramLayout.has("osc.frequency")).toBe(true);
        expect(compiled.paramLayout.has("filter.cutoff")).toBe(true);
        expect(compiled.paramLayout.has("env.attack")).toBe(true);
    });

    test("filter has 4 params", () => {
        const id = instrument(defaultGraph);
        const compiled = instrumentRegistry.get(id)!;
        const filterOffset = compiled.paramLayout.get("filter.cutoff")!;
        const envOffset = compiled.paramLayout.get("env.attack")!;
        expect(envOffset - filterOffset).toBe(4);
    });

    test("dual-osc has independent param entries per node", () => {
        const id = instrument(dualOscGraph);
        const compiled = instrumentRegistry.get(id)!;
        const osc1Freq = compiled.paramLayout.get("osc1.frequency");
        const osc2Freq = compiled.paramLayout.get("osc2.frequency");
        expect(osc1Freq).not.toBe(osc2Freq);
    });

    test("throws on cycle", () => {
        expect(() =>
            instrument({
                nodes: {
                    a: { type: "oscillator", input: "b" },
                    b: { type: "filter", input: "a" },
                },
                output: "b",
            }),
        ).toThrow("cycle");
    });

    test("throws on missing output", () => {
        expect(() =>
            instrument({
                nodes: { osc: { type: "oscillator" } },
                output: "missing",
            }),
        ).toThrow('output node "missing" not found');
    });

    test("5-mode modal graph (linear mix chain) fits within MAX_BUFFERS", () => {
        const nodes: Instrument["nodes"] = {};
        const N = 5;
        for (let i = 0; i < N; i++) {
            nodes[`osc${i}`] = { type: "oscillator" };
            nodes[`env${i}`] = { type: "envelope", input: `osc${i}` };
        }
        let prev = "env0";
        for (let i = 1; i < N; i++) {
            nodes[`mix${i}`] = { type: "mix", input: prev, inputB: `env${i}` };
            prev = `mix${i}`;
        }
        nodes.vol = { type: "gain", input: prev };
        const id = instrument({ nodes, output: "vol" });
        const compiled = instrumentRegistry.get(id)!;
        for (let i = 0; i < N; i++) {
            expect(compiled.paramLayout.has(`osc${i}.frequency`)).toBe(true);
            expect(compiled.paramLayout.has(`env${i}.decay`)).toBe(true);
        }
    });
});

describe("instrument overwrite-by-name", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("same name returns same ID", () => {
        const graph: Instrument = {
            nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
            output: "env",
        };
        const id1 = instrument(graph, "synth");
        const id2 = instrument(graph, "synth");
        expect(id2).toBe(id1);
    });

    test("overwrite bumps version", () => {
        const graph: Instrument = {
            nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
            output: "env",
        };
        const id = instrument(graph, "synth");
        const v1 = instrumentRegistry.get(id)!.version;
        instrument(graph, "synth");
        const v2 = instrumentRegistry.get(id)!.version;
        expect(v2).toBeGreaterThan(v1);
    });

    test("different names get different IDs", () => {
        const graph: Instrument = {
            nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
            output: "env",
        };
        const id1 = instrument(graph, "a");
        const id2 = instrument(graph, "b");
        expect(id2).not.toBe(id1);
    });

    test("overwrite replaces compiled data", () => {
        const graph1: Instrument = {
            nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
            output: "env",
            values: { "osc.waveform": 0 },
        };
        const graph2: Instrument = {
            nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
            output: "env",
            values: { "osc.waveform": 2 },
        };
        const id = instrument(graph1, "synth");
        instrument(graph2, "synth");
        expect(getValues(id)!.get("osc.waveform")).toBe(2);
    });
});

describe("instrument cap", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    const simpleGraph: Instrument = {
        nodes: { osc: { type: "oscillator" } },
        output: "osc",
    };

    test("registers up to MAX_INSTRUMENTS", () => {
        for (let i = 0; i < MAX_INSTRUMENTS; i++) {
            expect(instrument(simpleGraph)).toBe(i);
        }
    });

    test("returns -1 at cap", () => {
        for (let i = 0; i < MAX_INSTRUMENTS; i++) instrument(simpleGraph);
        expect(instrument(simpleGraph)).toBe(-1);
    });

    test("named overwrite still works at cap", () => {
        for (let i = 0; i < MAX_INSTRUMENTS; i++) instrument(simpleGraph, `inst${i}`);
        const id = instrument(simpleGraph, "inst0");
        expect(id).toBe(0);
    });
});

describe("instrument values", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("values stored separately from compiled instrument", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            values: {
                "osc.waveform": 1,
                "env.attack": 0.5,
            },
        });
        const values = getValues(id)!;
        expect(values.get("osc.waveform")).toBe(1);
        expect(values.get("env.attack")).toBe(0.5);
        expect(values.get("octave")).toBe(0);
        expect(values.get("semitone")).toBe(0);
        expect(values.get("fine")).toBe(0);
        expect(values.get("volume")).toBe(0.7);
    });

    test("getParamValues returns offset-value pairs", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            values: {
                "osc.waveform": 2,
                "env.sustain": 0.8,
            },
        });
        const compiled = instrumentRegistry.get(id)!;
        const pairs = getParamPairs(id);
        expect(pairs.length).toBe(compiled.paramLayout.size);
        const waveformOffset = compiled.paramLayout.get("osc.waveform")!;
        const sustainOffset = compiled.paramLayout.get("env.sustain")!;
        expect(pairs).toContainEqual([waveformOffset, 2]);
        expect(pairs).toContainEqual([sustainOffset, 0.8]);
    });

    test("no graph values still has metadata values", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
        });
        const compiled = instrumentRegistry.get(id)!;
        const values = getValues(id)!;
        expect(values.get("octave")).toBe(0);
        expect(values.get("volume")).toBe(0.7);
        const pairs = getParamPairs(id);
        expect(pairs.length).toBe(compiled.paramLayout.size);
    });

    test("per-oscillator pitch offset metadata defaults", () => {
        const id = instrument({
            nodes: {
                osc1: { type: "oscillator" },
                osc2: { type: "oscillator" },
                mix: { type: "mix", input: "osc1", inputB: "osc2" },
                env: { type: "envelope", input: "mix" },
            },
            output: "env",
        });
        const values = getValues(id)!;
        expect(values.get("osc1.octave")).toBe(0);
        expect(values.get("osc1.semitone")).toBe(0);
        expect(values.get("osc1.fine")).toBe(0);
        expect(values.get("osc2.octave")).toBe(0);
        expect(values.get("osc2.semitone")).toBe(0);
        expect(values.get("osc2.fine")).toBe(0);
    });

    test("per-oscillator pitch offsets are metadata, not WASM params", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            values: { "osc.octave": 2 },
        });
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.paramLayout.has("osc.octave")).toBe(false);
        expect(compiled.paramLayout.has("osc.semitone")).toBe(false);
        expect(compiled.paramLayout.has("osc.fine")).toBe(false);
        const values = getValues(id)!;
        expect(values.get("osc.octave")).toBe(2);
        const pairs = getParamPairs(id);
        expect(pairs.some(([, v]) => v === 2)).toBe(false);
    });

    test("per-oscillator pitch offsets overridable via values", () => {
        const id = instrument({
            nodes: {
                osc1: { type: "oscillator" },
                osc2: { type: "oscillator" },
                mix: { type: "mix", input: "osc1", inputB: "osc2" },
                env: { type: "envelope", input: "mix" },
            },
            output: "env",
            values: { "osc2.octave": 1, "osc2.semitone": 7, "osc2.fine": 50 },
        });
        const values = getValues(id)!;
        expect(values.get("osc1.octave")).toBe(0);
        expect(values.get("osc2.octave")).toBe(1);
        expect(values.get("osc2.semitone")).toBe(7);
        expect(values.get("osc2.fine")).toBe(50);
    });

    test("non-oscillator nodes have no pitch offset metadata", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
            },
            output: "env",
        });
        const values = getValues(id)!;
        expect(values.has("filter.octave")).toBe(false);
        expect(values.has("env.octave")).toBe(false);
    });

    test("non-param values are stored but not sent to worklet", () => {
        const id = instrument({
            nodes: {
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            values: { octave: 2, semitone: 3 },
        });
        const values = getValues(id)!;
        expect(values.get("octave")).toBe(2);
        expect(values.get("semitone")).toBe(3);
        expect(getParamPairs(id).find(([, v]) => v === 2)).toBeUndefined();
    });
});

describe("modulation compilation", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("modulation connection compiles correctly", () => {
        const graph: Instrument = {
            nodes: {
                lfo: { type: "oscillator" },
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            modulations: [{ source: "lfo", target: "osc", param: "frequency", mode: "semitone" }],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.modulations.length).toBe(1);
        const mod = compiled.modulations[0];
        expect(mod.mode).toBe(1);
        expect(mod.depthParam).toBeGreaterThanOrEqual(compiled.nodes.length);
    });

    test("depth param allocated after node params", () => {
        const graph: Instrument = {
            nodes: {
                lfo: { type: "oscillator" },
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            modulations: [{ source: "lfo", target: "osc", param: "frequency" }],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        const nodeParams = 4 + 4 + 7;
        expect(compiled.modulations[0].depthParam).toBe(nodeParams);
        expect(compiled.totalParams).toBe(nodeParams + 1);
    });

    test("depth param appears in paramLayout with correct key", () => {
        const graph: Instrument = {
            nodes: {
                lfo: { type: "oscillator" },
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            modulations: [{ source: "lfo", target: "osc", param: "frequency" }],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        const entry = compiled.paramLayout.get("lfo>osc.frequency");
        expect(entry).toBeDefined();
    });

    test("two modulations targeting same param both get depth params", () => {
        const graph: Instrument = {
            nodes: {
                lfo1: { type: "oscillator" },
                lfo2: { type: "oscillator" },
                osc: { type: "oscillator" },
                env: { type: "envelope", input: "osc" },
            },
            output: "env",
            modulations: [
                { source: "lfo1", target: "osc", param: "frequency" },
                { source: "lfo2", target: "osc", param: "frequency" },
            ],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.modulations.length).toBe(2);
        expect(compiled.modulations[0].depthParam).not.toBe(compiled.modulations[1].depthParam);
    });

    test("envelope without input compiles with correct param layout", () => {
        const graph: Instrument = {
            nodes: {
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
                fltEnv: { type: "envelope" },
            },
            output: "env",
            modulations: [
                { source: "fltEnv", target: "filter", param: "cutoff", mode: "semitone" },
            ],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.paramLayout.has("fltEnv.attack")).toBe(true);
        const fltEnvNode = compiled.nodes.find(
            (_, i) =>
                compiled.paramLayout.get("fltEnv.attack")! >= compiled.nodes[i].paramOffset &&
                compiled.paramLayout.get("fltEnv.attack")! < compiled.nodes[i].paramOffset + 7,
        )!;
        expect(fltEnvNode.inputBuf).toBe(0xff);
    });

    test("default instrument with modulation round-trip", () => {
        const graph: Instrument = {
            nodes: {
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
                fltEnv: { type: "envelope" },
            },
            output: "env",
            modulations: [
                {
                    source: "fltEnv",
                    target: "filter",
                    param: "cutoff",
                    mode: "semitone",
                },
            ],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.nodes.length).toBe(4);
        expect(compiled.modulations.length).toBe(1);
        expect(compiled.totalParams).toBe(4 + 4 + 7 + 7 + 1);
        expect(compiled.paramLayout.has("fltEnv.attack")).toBe(true);
        expect(compiled.paramLayout.has("fltEnv>filter.cutoff")).toBe(true);
    });

    test("throws on unknown modulation source", () => {
        expect(() =>
            instrument({
                nodes: { osc: { type: "oscillator" }, env: { type: "envelope", input: "osc" } },
                output: "env",
                modulations: [{ source: "missing", target: "osc", param: "frequency" }],
            }),
        ).toThrow('unknown source "missing"');
    });

    test("throws on unknown modulation target param", () => {
        expect(() =>
            instrument({
                nodes: {
                    lfo: { type: "oscillator" },
                    osc: { type: "oscillator" },
                    env: { type: "envelope", input: "osc" },
                },
                output: "env",
                modulations: [{ source: "lfo", target: "osc", param: "nonexistent" }],
            }),
        ).toThrow('target param "nonexistent"');
    });

    test("topo sort respects modulation edges", () => {
        const graph: Instrument = {
            nodes: {
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
                fltEnv: { type: "envelope" },
            },
            output: "env",
            modulations: [{ source: "fltEnv", target: "filter", param: "cutoff" }],
        };
        const id = instrument(graph);
        const compiled = instrumentRegistry.get(id)!;
        const fltEnvOffset = compiled.paramLayout.get("fltEnv.attack")!;
        const filterOffset = compiled.paramLayout.get("filter.cutoff")!;
        const fltEnvIdx = compiled.nodes.findIndex((n) => n.paramOffset === fltEnvOffset);
        const filterIdx = compiled.nodes.findIndex((n) => n.paramOffset === filterOffset);
        expect(fltEnvIdx).toBeLessThan(filterIdx);
    });
});

describe("midiFreq", () => {
    test("MIDI 69 = 440 Hz", () => {
        expect(midiFreq(69)).toBeCloseTo(440, 4);
    });

    test("MIDI 60 ~ 261.6 Hz", () => {
        expect(midiFreq(60)).toBeCloseTo(261.626, 1);
    });

    test("octave = 2x frequency", () => {
        const f1 = midiFreq(60);
        const f2 = midiFreq(72);
        expect(f2 / f1).toBeCloseTo(2, 4);
    });
});

describe("pattern registration", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    const simpleDef: Pattern = {
        tracks: {
            kick: {
                instrument: "kick",
                notes: [{ beat: 0, note: 36, velocity: 1, duration: 0.5 }],
            },
        },
        length: 4,
    };

    test("IDs increment", () => {
        const id1 = pattern(simpleDef);
        const id2 = pattern(simpleDef);
        expect(id2).toBe(id1 + 1);
    });

    test("same name overwrites and returns same ID", () => {
        const id1 = pattern(simpleDef, "drums");
        const id2 = pattern(simpleDef, "drums");
        expect(id2).toBe(id1);
    });

    test("overwrite bumps version", () => {
        const id = pattern(simpleDef, "drums");
        const v1 = patternRegistry.get(id)!.version;
        pattern(simpleDef, "drums");
        const v2 = patternRegistry.get(id)!.version;
        expect(v2).toBeGreaterThan(v1);
    });

    test("fresh registry clears all", () => {
        pattern(simpleDef, "drums");
        clearInstruments();
        clearPatterns();
        expect(patternRegistry.getByName("drums")).toBeUndefined();
        expect(patternRegistry.get(0)).toBeUndefined();
    });

    test("getPatternByName resolves", () => {
        const id = pattern(simpleDef, "drums");
        expect(patternRegistry.getByName("drums")).toBe(id);
    });

    test("notes sorted by beat on registration", () => {
        const def: Pattern = {
            tracks: {
                t: {
                    instrument: "x",
                    notes: [
                        { beat: 2, note: 40, velocity: 1, duration: 0.5 },
                        { beat: 0, note: 36, velocity: 1, duration: 0.5 },
                        { beat: 1, note: 38, velocity: 0.8, duration: 0.5 },
                    ],
                },
            },
            length: 4,
        };
        const id = pattern(def);
        const track = patternRegistry.get(id)!.def.tracks.t;
        expect(track.notes[0].beat).toBe(0);
        expect(track.notes[1].beat).toBe(1);
        expect(track.notes[2].beat).toBe(2);
    });

    test("compile does not mutate caller's notes array", () => {
        const originalNotes = [
            { beat: 2, note: 40, velocity: 1, duration: 0.5 },
            { beat: 0, note: 36, velocity: 1, duration: 0.5 },
            { beat: 1, note: 38, velocity: 0.8, duration: 0.5 },
        ];
        const def: Pattern = {
            tracks: { t: { instrument: "x", notes: originalNotes } },
            length: 4,
        };
        pattern(def);
        expect(originalNotes[0].beat).toBe(2);
        expect(originalNotes[1].beat).toBe(0);
        expect(originalNotes[2].beat).toBe(1);
    });

    test("trackOrder is sorted", () => {
        const def: Pattern = {
            tracks: {
                snare: { instrument: "snare", notes: [] },
                kick: { instrument: "kick", notes: [] },
                hat: { instrument: "hat", notes: [] },
            },
            length: 4,
        };
        const id = pattern(def);
        expect(patternRegistry.get(id)!.trackOrder).toEqual(["hat", "kick", "snare"]);
    });
});

describe("notes", () => {
    const track: PatternTrack = {
        instrument: "kick",
        notes: [
            { beat: 0, note: 36, velocity: 1, duration: 0.5 },
            { beat: 1, note: 36, velocity: 0.8, duration: 0.5 },
            { beat: 2, note: 36, velocity: 1, duration: 0.5 },
            { beat: 3, note: 36, velocity: 0.9, duration: 0.5 },
        ],
    };

    test("returns notes in range", () => {
        const result = notes(track, 1, 3);
        expect(result.length).toBe(2);
        expect(result[0].beat).toBe(1);
        expect(result[1].beat).toBe(2);
    });

    test("from inclusive, to exclusive", () => {
        const result = notes(track, 2, 3);
        expect(result.length).toBe(1);
        expect(result[0].beat).toBe(2);
    });

    test("out-of-range excluded", () => {
        const result = notes(track, 5, 10);
        expect(result.length).toBe(0);
    });

    test("empty track", () => {
        const empty: PatternTrack = { instrument: "x", notes: [] };
        expect(notes(empty, 0, 4).length).toBe(0);
    });
});

describe("evalCurve", () => {
    test("no curve = linear min-max", () => {
        const mapping: CurveMapping = { min: 0, max: 10 };
        expect(evalCurve(mapping, 0)).toBeCloseTo(0);
        expect(evalCurve(mapping, 0.5)).toBeCloseTo(5);
        expect(evalCurve(mapping, 1)).toBeCloseTo(10);
    });

    test("defaults min=0 max=1", () => {
        const mapping: CurveMapping = {};
        expect(evalCurve(mapping, 0.5)).toBeCloseTo(0.5);
    });

    test("piecewise linear interpolation", () => {
        const mapping: CurveMapping = {
            curve: [0, 0, 0.5, 1, 1, 0],
            min: 0,
            max: 100,
        };
        expect(evalCurve(mapping, 0)).toBeCloseTo(0);
        expect(evalCurve(mapping, 0.25)).toBeCloseTo(50);
        expect(evalCurve(mapping, 0.5)).toBeCloseTo(100);
        expect(evalCurve(mapping, 0.75)).toBeCloseTo(50);
        expect(evalCurve(mapping, 1)).toBeCloseTo(0);
    });

    test("clamps at extremes", () => {
        const mapping: CurveMapping = {
            curve: [0.2, 0, 0.8, 1],
            min: 0,
            max: 10,
        };
        expect(evalCurve(mapping, 0)).toBeCloseTo(0);
        expect(evalCurve(mapping, 1)).toBeCloseTo(10);
    });
});

describe("modulation guards", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("self_modulation_rejected", () => {
        expect(() =>
            instrument({
                nodes: {
                    osc: { type: "oscillator" },
                    env: { type: "envelope", input: "osc" },
                },
                output: "env",
                modulations: [{ source: "osc", target: "osc", param: "frequency" }],
            }),
        ).toThrow("self-modulation not allowed");
    });

    test("discrete_param_waveform_rejected", () => {
        expect(() =>
            instrument({
                nodes: {
                    lfo: { type: "oscillator" },
                    osc: { type: "oscillator" },
                    env: { type: "envelope", input: "osc" },
                },
                output: "env",
                modulations: [{ source: "lfo", target: "osc", param: "waveform" }],
            }),
        ).toThrow('cannot modulate discrete param "waveform"');
    });

    test("discrete_param_filter_mode_rejected", () => {
        expect(() =>
            instrument({
                nodes: {
                    lfo: { type: "oscillator" },
                    osc: { type: "oscillator" },
                    filter: { type: "filter", input: "osc" },
                    env: { type: "envelope", input: "filter" },
                },
                output: "env",
                modulations: [{ source: "lfo", target: "filter", param: "mode" }],
            }),
        ).toThrow('cannot modulate discrete param "mode"');
    });

    test("continuous_params_accepted", () => {
        expect(() =>
            instrument({
                nodes: {
                    lfo: { type: "oscillator" },
                    osc: { type: "oscillator" },
                    filter: { type: "filter", input: "osc" },
                    env: { type: "envelope", input: "filter" },
                    gain: { type: "gain", input: "env" },
                    mix: { type: "mix", input: "env", inputB: "gain" },
                },
                output: "mix",
                modulations: [
                    { source: "lfo", target: "osc", param: "frequency" },
                    { source: "lfo", target: "filter", param: "cutoff" },
                    { source: "lfo", target: "filter", param: "q" },
                    { source: "lfo", target: "filter", param: "mix" },
                    { source: "lfo", target: "osc", param: "volume" },
                    { source: "lfo", target: "gain", param: "level" },
                    { source: "lfo", target: "mix", param: "mix" },
                    { source: "lfo", target: "env", param: "attack" },
                ],
            }),
        ).not.toThrow();
    });

    test("every_node_type_solo", () => {
        expect(() =>
            instrument({
                nodes: { osc: { type: "oscillator" } },
                output: "osc",
            }),
        ).not.toThrow();

        expect(() =>
            instrument({
                nodes: { env: { type: "envelope" } },
                output: "env",
            }),
        ).not.toThrow();

        expect(() =>
            instrument({
                nodes: {
                    osc: { type: "oscillator" },
                    gain: { type: "gain", input: "osc" },
                },
                output: "gain",
            }),
        ).not.toThrow();

        expect(() =>
            instrument({
                nodes: {
                    osc1: { type: "oscillator" },
                    osc2: { type: "oscillator" },
                    mix: { type: "mix", input: "osc1", inputB: "osc2" },
                },
                output: "mix",
            }),
        ).not.toThrow();
    });
});

describe("constant node", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("constant_node_compiles", () => {
        const id = instrument({
            nodes: {
                c: { type: "constant" },
                gain: { type: "gain", input: "c" },
            },
            output: "gain",
        });
        const compiled = instrumentRegistry.get(id)!;
        expect(compiled.paramLayout.has("c.value")).toBe(true);
    });

    test("constant_node_round_trip", () => {
        const id = instrument({
            nodes: {
                c: { type: "constant" },
                gain: { type: "gain", input: "c" },
            },
            output: "gain",
        });
        const compiled = instrumentRegistry.get(id)!;
        const constNode = compiled.nodes.find((n) => n.type === 6);
        expect(constNode).toBeDefined();
        expect(compiled.nodes.length).toBe(2);
    });
});

describe("buffer count validation", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("throws when graph needs more than MAX_BUFFERS", () => {
        const nodes: Record<
            string,
            { type: "oscillator" | "mix"; input?: string; inputB?: string }
        > = {};
        for (let i = 0; i < MAX_BUFFERS + 1; i++) {
            nodes[`osc${i}`] = { type: "oscillator" };
        }
        expect(() => instrument({ nodes, output: "osc0" })).toThrow(`buffers (max ${MAX_BUFFERS})`);
    });

    test("graph within buffer limit compiles", () => {
        expect(() =>
            instrument({
                nodes: {
                    osc1: { type: "oscillator" },
                    osc2: { type: "oscillator" },
                    mix: { type: "mix", input: "osc1", inputB: "osc2" },
                },
                output: "mix",
            }),
        ).not.toThrow();
    });
});

describe("evaluation order invariant", () => {
    beforeEach(() => {
        clearInstruments();
        clearPatterns();
    });

    test("modulation source evaluated before target", () => {
        const id = instrument({
            nodes: {
                lfo: { type: "oscillator" },
                osc: { type: "oscillator" },
                filter: { type: "filter", input: "osc" },
                env: { type: "envelope", input: "filter" },
            },
            output: "env",
            modulations: [{ source: "lfo", target: "filter", param: "cutoff" }],
        });
        const compiled = instrumentRegistry.get(id)!;
        const lfoIdx = compiled.nodes.findIndex(
            (n) => n.paramOffset === compiled.paramLayout.get("lfo.frequency")!,
        );
        const filterIdx = compiled.nodes.findIndex(
            (n) => n.paramOffset === compiled.paramLayout.get("filter.cutoff")!,
        );
        expect(lfoIdx).toBeLessThan(filterIdx);
    });
});

describe("constants sync", () => {
    test("TS MAX_INSTRUMENTS matches Rust", async () => {
        const rs = await Bun.file("packages/shallot/rust/audio/src/graph.rs").text();
        const match = rs.match(/MAX_INSTRUMENTS:\s*usize\s*=\s*(\d+)/);
        expect(match).not.toBeNull();
        expect(MAX_INSTRUMENTS).toBe(Number(match![1]));
    });

    test("TS MAX_BUFFERS matches Rust", async () => {
        const rs = await Bun.file("packages/shallot/rust/audio/src/graph.rs").text();
        const match = rs.match(/MAX_BUFFERS:\s*usize\s*=\s*(\d+)/);
        expect(match).not.toBeNull();
        expect(MAX_BUFFERS).toBe(Number(match![1]));
    });
});
