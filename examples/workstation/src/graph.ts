import { instrument, type Instrument } from "@dylanebert/shallot";
import {
    setValues,
    upload,
    refresh,
    getValues,
    type AudioState,
} from "@dylanebert/shallot/audio/core";

type NodeType = "oscillator" | "filter" | "envelope" | "gain" | "mix" | "constant" | "sample";

export interface SlotConfig {
    id: string;
    type: NodeType;
    label: string;
    removable: boolean;
}

export interface ModRoute {
    source: string;
    target: string;
    param: string;
    depth: number;
    mode: "linear" | "semitone";
}

function nextId(prefix: string, slots: SlotConfig[]): string {
    const nums = slots
        .filter((s) => s.id.startsWith(prefix))
        .map((s) => parseInt(s.id.slice(prefix.length)));
    let n = 1;
    while (nums.includes(n)) n++;
    return `${prefix}${n}`;
}

export type AddableType = "oscillator" | "filter" | "envelope" | "constant" | "sample";

const MAX_SOURCES = 3;

function sourceCount(slots: SlotConfig[]): number {
    return slots.filter((s) => s.type === "oscillator" || s.type === "sample").length;
}

export function canAdd(slots: SlotConfig[], type: AddableType): boolean {
    if (type === "oscillator") return sourceCount(slots) < MAX_SOURCES;
    if (type === "sample") return sourceCount(slots) < MAX_SOURCES;
    if (type === "filter") return slots.filter((s) => s.type === "filter").length < 2;
    if (type === "envelope")
        return slots.filter((s) => s.type === "envelope" && s.id !== "ampEnv").length < 2;
    if (type === "constant") return slots.filter((s) => s.type === "constant").length < 2;
    return false;
}

export function createSlot(slots: SlotConfig[], type: AddableType): SlotConfig | null {
    if (!canAdd(slots, type)) return null;
    if (type === "oscillator") {
        const id = nextId("osc", slots);
        const n = slots.filter((s) => s.type === "oscillator").length + 1;
        return { id, type: "oscillator", label: `Osc ${n}`, removable: true };
    }
    if (type === "sample") {
        const id = nextId("sample", slots);
        const n = slots.filter((s) => s.type === "sample").length + 1;
        return { id, type: "sample", label: `Sample ${n}`, removable: true };
    }
    if (type === "filter") {
        const id = nextId("filter", slots);
        const n = slots.filter((s) => s.type === "filter").length + 1;
        return { id, type: "filter", label: `Filter ${n}`, removable: true };
    }
    if (type === "envelope") {
        const id = nextId("modEnv", slots);
        const n = slots.filter((s) => s.type === "envelope" && s.id !== "ampEnv").length + 1;
        return { id, type: "envelope", label: `Mod Env ${n}`, removable: true };
    }
    if (type === "constant") {
        const id = nextId("const", slots);
        const n = slots.filter((s) => s.type === "constant").length + 1;
        return { id, type: "constant", label: `Const ${n}`, removable: true };
    }
    return null;
}

export function removeSlot(
    slots: SlotConfig[],
    modRoutes: ModRoute[],
    id: string,
): { slots: SlotConfig[]; modRoutes: ModRoute[] } {
    return {
        slots: slots.filter((s) => s.id !== id),
        modRoutes: modRoutes.filter((r) => r.source !== id && r.target !== id),
    };
}

function getSources(slots: SlotConfig[]): SlotConfig[] {
    return slots.filter((s) => s.type === "oscillator" || s.type === "sample");
}

export function getMixSlots(slots: SlotConfig[]): SlotConfig[] {
    const n = getSources(slots).length;
    if (n === 2) return [{ id: "mix", type: "mix", label: "Mix", removable: false }];
    if (n >= 3)
        return [
            { id: "mix1", type: "mix", label: "Mix 1", removable: false },
            { id: "mix2", type: "mix", label: "Mix 2", removable: false },
        ];
    return [];
}

export function getDisplaySlots(slots: SlotConfig[]): SlotConfig[] {
    const sources = getSources(slots);
    const filters = slots.filter((s) => s.type === "filter");
    const ampEnv = slots.filter((s) => s.id === "ampEnv");
    const modEnvs = slots.filter((s) => s.type === "envelope" && s.id !== "ampEnv");
    const constants = slots.filter((s) => s.type === "constant");
    const vol = slots.filter((s) => s.id === "vol");
    const mixSlots = getMixSlots(slots);
    return [...sources, ...mixSlots, ...filters, ...ampEnv, ...modEnvs, ...constants, ...vol];
}

export function buildInstrument(
    slots: SlotConfig[],
    modRoutes: ModRoute[],
    values?: Record<string, number>,
): Instrument {
    const sources = getSources(slots);
    const oscs = sources.filter((s) => s.type === "oscillator");
    const filters = slots.filter((s) => s.type === "filter");
    const modEnvs = slots.filter((s) => s.type === "envelope" && s.id !== "ampEnv");
    const constants = slots.filter((s) => s.type === "constant");

    const nodes: Record<string, { type: NodeType; input?: string; inputB?: string }> = {};

    for (const src of sources) {
        nodes[src.id] = { type: src.type };
    }

    let chain: string;
    if (sources.length <= 1) {
        chain = sources[0]?.id ?? "osc1";
    } else if (sources.length === 2) {
        nodes.mix = { type: "mix", input: sources[0].id, inputB: sources[1].id };
        chain = "mix";
    } else {
        nodes.mix1 = { type: "mix", input: sources[0].id, inputB: sources[1].id };
        nodes.mix2 = { type: "mix", input: "mix1", inputB: sources[2].id };
        chain = "mix2";
    }

    for (const f of filters) {
        nodes[f.id] = { type: "filter", input: chain };
        chain = f.id;
    }

    nodes.ampEnv = { type: "envelope", input: chain };
    chain = "ampEnv";

    nodes.vol = { type: "gain", input: chain };

    for (const me of modEnvs) nodes[me.id] = { type: "envelope" };
    for (const c of constants) nodes[c.id] = { type: "constant" };

    const modulations = modRoutes
        .filter((r) => nodes[r.source] && nodes[r.target])
        .map((r) => ({
            source: r.source,
            target: r.target,
            param: r.param,
            mode: r.mode,
        }));

    const graph: Instrument = {
        nodes,
        output: "vol",
        modulations,
        volumeParam: "vol.level",
        pitchParams: oscs.length > 0 ? oscs.map((o) => `${o.id}.frequency`) : undefined,
    };

    if (values) graph.values = values;
    return graph;
}

let instId = -1;

export function getInstId(): number {
    return instId;
}

export function compile(
    slots: SlotConfig[],
    modRoutes: ModRoute[],
    values?: Record<string, number>,
): number {
    const graph = buildInstrument(slots, modRoutes, values);
    instId = instrument(graph, "synth");
    if (values) {
        const compiled = getValues(instId);
        if (compiled) {
            for (const [key, val] of compiled) {
                if (values[key] === undefined) values[key] = val;
            }
        }
    }
    return instId;
}

export function recompile(
    audio: AudioState,
    slots: SlotConfig[],
    modRoutes: ModRoute[],
    values?: Record<string, number>,
): void {
    compile(slots, modRoutes, values);
    refresh(audio, instId);
}

export function applyValues(audio: AudioState, changed: Record<string, number>): void {
    if (instId < 0) return;
    setValues(instId, changed);
    upload(audio, instId, changed);
}

export function depthKey(route: ModRoute): string {
    return `${route.source}>${route.target}.${route.param}`;
}

const NODE_PARAM_NAMES: Record<string, string[]> = {
    oscillator: ["frequency", "waveform", "wavetablePos", "volume"],
    filter: ["cutoff", "q", "mode", "mix"],
    envelope: [
        "attack",
        "decay",
        "sustain",
        "release",
        "attackCurve",
        "decayCurve",
        "releaseCurve",
    ],
    gain: ["level"],
    mix: ["mix"],
    constant: ["value"],
    sample: ["bufferId", "rate", "loop", "volume"],
};

const DISCRETE_PARAMS = new Set(["waveform", "mode", "bufferId", "loop"]);

export function getModulableParams(
    slots: SlotConfig[],
): { slotId: string; slotLabel: string; param: string }[] {
    const result: { slotId: string; slotLabel: string; param: string }[] = [];
    for (const slot of slots) {
        const params = NODE_PARAM_NAMES[slot.type];
        if (!params) continue;
        for (const p of params) {
            if (DISCRETE_PARAMS.has(p)) continue;
            result.push({ slotId: slot.id, slotLabel: slot.label, param: p });
        }
    }
    return result;
}

export function getModSources(slots: SlotConfig[]): SlotConfig[] {
    return slots.filter(
        (s) => (s.type === "envelope" && s.id !== "ampEnv") || s.type === "constant",
    );
}
