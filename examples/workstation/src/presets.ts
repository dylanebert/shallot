import type { SlotConfig, ModRoute } from "./graph";

export type KnobScale = "linear" | "log";

export type KnobDef = {
    label: string;
    field: string;
    min: number;
    max: number;
    step: number;
    default: number;
    scale?: "log";
    fmt?: (v: number) => string;
    parse?: (s: string) => number | undefined;
    toAudio?: (v: number) => number;
    fromAudio?: (v: number) => number;
};

export function valueToNorm(value: number, min: number, max: number, scale: KnobScale): number {
    if (scale === "log") {
        const logMin = Math.log(min);
        const logMax = Math.log(max);
        return (Math.log(Math.max(min, value)) - logMin) / (logMax - logMin);
    }
    return (value - min) / (max - min);
}

export function normToValue(
    norm: number,
    min: number,
    max: number,
    scale: KnobScale,
    step: number,
): number {
    if (scale === "log") {
        const logMin = Math.log(min);
        const logMax = Math.log(max);
        return Math.exp(logMin + norm * (logMax - logMin));
    }
    const raw = min + norm * (max - min);
    return Math.round(raw / step) * step;
}

export function fmtTime(v: number) {
    return v < 1 ? `${(v * 1000).toFixed(0)}ms` : `${v.toFixed(2)}s`;
}

export function parseTime(s: string): number | undefined {
    const text = s.trim();
    const m = text.match(/^(-?\d+(?:\.\d+)?)\s*(ms|s)?$/i);
    if (!m) return undefined;
    const num = parseFloat(m[1]);
    if (Number.isNaN(num)) return undefined;
    const unit = (m[2] || "").toLowerCase();
    if (unit === "s") return num;
    return num / 1000;
}

export function fmtHz(v: number) {
    return v >= 1000 ? `${(v / 1000).toFixed(1)}kHz` : `${v.toFixed(0)}Hz`;
}

export function parseHz(s: string): number | undefined {
    const text = s.trim();
    const m = text.match(/^(-?\d+(?:\.\d+)?)\s*(kHz|Hz)?$/i);
    if (!m) return undefined;
    const num = parseFloat(m[1]);
    if (Number.isNaN(num)) return undefined;
    const unit = (m[2] || "").toLowerCase();
    if (unit === "khz") return num * 1000;
    return num;
}

export function fmtPct(v: number) {
    return Math.round(v * 100) + "%";
}

export function parsePct(s: string): number | undefined {
    const m = s.trim().match(/^(-?\d+(?:\.\d+)?)\s*%?$/);
    if (!m) return undefined;
    return parseFloat(m[1]) / 100;
}

const RES_MAX_Q = 18;
const RES_CURVE = 2.5;

export function resToQ(res: number): number {
    return Math.SQRT1_2 * (RES_MAX_Q / Math.SQRT1_2) ** (res ** RES_CURVE);
}

export function qToRes(q: number): number {
    const ratio = Math.log(q / Math.SQRT1_2) / Math.log(RES_MAX_Q / Math.SQRT1_2);
    return Math.max(0, Math.min(1, ratio ** (1 / RES_CURVE)));
}

function fmtSemitone(v: number) {
    const r = Math.round(v);
    return r > 0 ? `+${r}` : `${r}`;
}

function parseSemitone(s: string): number | undefined {
    const n = parseInt(s.trim(), 10);
    return Number.isNaN(n) ? undefined : n;
}

function fmtCents(v: number) {
    const r = Math.round(v);
    return r > 0 ? `+${r}ct` : `${r}ct`;
}

function parseCents(s: string): number | undefined {
    const n = parseInt(s.trim().replace(/ct$/i, ""), 10);
    return Number.isNaN(n) ? undefined : n;
}

export const keyboardKnobs: KnobDef[] = [
    {
        label: "Oct",
        field: "octave",
        min: -4,
        max: 4,
        step: 1,
        default: 0,
        fmt: fmtSemitone,
        parse: parseSemitone,
    },
    {
        label: "Semi",
        field: "semitone",
        min: -12,
        max: 12,
        step: 1,
        default: 0,
        fmt: fmtSemitone,
        parse: parseSemitone,
    },
    {
        label: "Fine",
        field: "fine",
        min: -100,
        max: 100,
        step: 1,
        default: 0,
        fmt: fmtCents,
        parse: parseCents,
    },
    {
        label: "Volume",
        field: "volume",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.5,
        fmt: fmtPct,
        parse: parsePct,
    },
];

export const oscPitchKnobs: KnobDef[] = [
    {
        label: "Oct",
        field: "octave",
        min: -4,
        max: 4,
        step: 1,
        default: 0,
        fmt: fmtSemitone,
        parse: parseSemitone,
    },
    {
        label: "Semi",
        field: "semitone",
        min: -12,
        max: 12,
        step: 1,
        default: 0,
        fmt: fmtSemitone,
        parse: parseSemitone,
    },
    {
        label: "Fine",
        field: "fine",
        min: -100,
        max: 100,
        step: 1,
        default: 0,
        fmt: fmtCents,
        parse: parseCents,
    },
];

export const oscKnobs: KnobDef[] = [
    {
        label: "Volume",
        field: "volume",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.7,
        fmt: fmtPct,
        parse: parsePct,
    },
];

export const filterKnobs: KnobDef[] = [
    {
        label: "Cutoff",
        field: "cutoff",
        min: 20,
        max: 22050,
        step: 1,
        default: 450,
        scale: "log",
        fmt: fmtHz,
        parse: parseHz,
    },
    {
        label: "Res",
        field: "q",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0,
        fmt: fmtPct,
        parse: parsePct,
        toAudio: resToQ,
        fromAudio: qToRes,
    },
    {
        label: "Mix",
        field: "mix",
        min: 0,
        max: 1,
        step: 0.01,
        default: 1,
        fmt: fmtPct,
        parse: parsePct,
    },
];

export const ampEnvKnobs: KnobDef[] = [
    {
        label: "Attack",
        field: "attack",
        min: 0.001,
        max: 10,
        step: 0.001,
        default: 0.01,
        scale: "log",
        fmt: fmtTime,
        parse: parseTime,
    },
    {
        label: "Decay",
        field: "decay",
        min: 0.001,
        max: 10,
        step: 0.001,
        default: 0.1,
        scale: "log",
        fmt: fmtTime,
        parse: parseTime,
    },
    { label: "Sustain", field: "sustain", min: 0, max: 1, step: 0.01, default: 0.7, fmt: fmtPct },
    {
        label: "Release",
        field: "release",
        min: 0.001,
        max: 10,
        step: 0.001,
        default: 0.3,
        scale: "log",
        fmt: fmtTime,
        parse: parseTime,
    },
    {
        label: "A Crv",
        field: "attackCurve",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.5,
        fmt: fmtPct,
        parse: parsePct,
        toAudio: (v: number) => (0.5 - v) * 2,
        fromAudio: (v: number) => 0.5 - v / 2,
    },
    {
        label: "D Crv",
        field: "decayCurve",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.65,
        fmt: fmtPct,
        parse: parsePct,
        toAudio: (v: number) => (0.5 - v) * 2,
        fromAudio: (v: number) => 0.5 - v / 2,
    },
    {
        label: "R Crv",
        field: "releaseCurve",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.65,
        fmt: fmtPct,
        parse: parsePct,
        toAudio: (v: number) => (0.5 - v) * 2,
        fromAudio: (v: number) => 0.5 - v / 2,
    },
];

export const constantKnobs: KnobDef[] = [
    {
        label: "Value",
        field: "value",
        min: -1,
        max: 1,
        step: 0.01,
        default: 0,
        fmt: (v) => v.toFixed(2),
    },
];

export const gainKnobs: KnobDef[] = [
    {
        label: "Level",
        field: "level",
        min: 0,
        max: 2,
        step: 0.01,
        default: 1,
        fmt: fmtPct,
        parse: parsePct,
    },
];

export const mixKnobs: KnobDef[] = [
    {
        label: "Mix",
        field: "mix",
        min: 0,
        max: 1,
        step: 0.01,
        default: 0.5,
        fmt: fmtPct,
        parse: parsePct,
    },
];

function fmtRate(v: number): string {
    return Number.isFinite(v) ? v.toFixed(2) + "x" : "—";
}

function parseRate(s: string): number | undefined {
    const m = s.trim().match(/^(-?\d+(?:\.\d+)?)\s*x?$/i);
    if (!m) return undefined;
    const n = parseFloat(m[1]);
    return Number.isNaN(n) ? undefined : n;
}

export const sampleKnobs: KnobDef[] = [
    {
        label: "Rate",
        field: "rate",
        min: 0.25,
        max: 4,
        step: 0.01,
        default: 1,
        scale: "log",
        fmt: fmtRate,
        parse: parseRate,
    },
    {
        label: "Volume",
        field: "volume",
        min: 0,
        max: 1,
        step: 0.01,
        default: 1,
        fmt: fmtPct,
        parse: parsePct,
    },
];

export const loopOptions = ["One-shot", "Loop"];

export const waveforms = ["Sine", "Saw", "Square", "Triangle"];
export const filterModes = ["Off", "LP", "HP", "BP", "Notch"];

export interface GraphPreset {
    name: string;
    slots: SlotConfig[];
    modRoutes: ModRoute[];
    values: Record<string, number>;
}

function slot(id: string, type: string, label: string, removable = true): SlotConfig {
    return { id, type: type as SlotConfig["type"], label, removable };
}

const BASE_META = { octave: 0, semitone: 0, fine: 0, volume: 0.5 };
const BASE_ENV = {
    "ampEnv.attack": 0.01,
    "ampEnv.decay": 0.1,
    "ampEnv.sustain": 0.7,
    "ampEnv.release": 0.3,
    "ampEnv.attackCurve": 0,
    "ampEnv.decayCurve": -0.3,
    "ampEnv.releaseCurve": -0.3,
};

export const graphPresets: GraphPreset[] = [
    {
        name: "Init",
        slots: [
            slot("osc1", "oscillator", "Osc 1", false),
            slot("ampEnv", "envelope", "Amp Env", false),
            slot("vol", "gain", "Volume", false),
        ],
        modRoutes: [],
        values: {
            ...BASE_META,
            "osc1.waveform": 1,
            "osc1.volume": 0.7,
            ...BASE_ENV,
            "vol.level": 1,
        },
    },
    {
        name: "Classic Sub",
        slots: [
            slot("osc1", "oscillator", "Osc 1", false),
            slot("filter1", "filter", "Filter 1"),
            slot("ampEnv", "envelope", "Amp Env", false),
            slot("vol", "gain", "Volume", false),
        ],
        modRoutes: [],
        values: {
            ...BASE_META,
            "osc1.waveform": 1,
            "osc1.volume": 0.7,
            "filter1.cutoff": 450,
            "filter1.q": 2.5,
            "filter1.mode": 0,
            "filter1.mix": 1,
            ...BASE_ENV,
            "vol.level": 1,
        },
    },
    {
        name: "Dual Osc",
        slots: [
            slot("osc1", "oscillator", "Osc 1", false),
            slot("osc2", "oscillator", "Osc 2"),
            slot("filter1", "filter", "Filter 1"),
            slot("ampEnv", "envelope", "Amp Env", false),
            slot("vol", "gain", "Volume", false),
        ],
        modRoutes: [],
        values: {
            ...BASE_META,
            "osc1.waveform": 1,
            "osc1.volume": 0.7,
            "osc2.waveform": 3,
            "osc2.volume": 0.7,
            "osc2.fine": 10,
            "mix.mix": 0.5,
            "filter1.cutoff": 1200,
            "filter1.q": 1.5,
            "filter1.mode": 0,
            "filter1.mix": 1,
            "ampEnv.attack": 0.3,
            "ampEnv.decay": 0.5,
            "ampEnv.sustain": 0.6,
            "ampEnv.release": 0.8,
            "ampEnv.attackCurve": 0,
            "ampEnv.decayCurve": -0.3,
            "ampEnv.releaseCurve": -0.3,
            "vol.level": 1,
        },
    },
    {
        name: "Filter Sweep",
        slots: [
            slot("osc1", "oscillator", "Osc 1", false),
            slot("filter1", "filter", "Filter 1"),
            slot("ampEnv", "envelope", "Amp Env", false),
            slot("modEnv1", "envelope", "Mod Env 1"),
            slot("vol", "gain", "Volume", false),
        ],
        modRoutes: [
            { source: "modEnv1", target: "filter1", param: "cutoff", depth: 0.8, mode: "linear" },
        ],
        values: {
            ...BASE_META,
            "osc1.waveform": 1,
            "osc1.volume": 0.7,
            "filter1.cutoff": 300,
            "filter1.q": 4,
            "filter1.mode": 0,
            "filter1.mix": 1,
            ...BASE_ENV,
            "modEnv1.attack": 0.01,
            "modEnv1.decay": 0.5,
            "modEnv1.sustain": 0,
            "modEnv1.release": 0.2,
            "modEnv1.attackCurve": 0,
            "modEnv1.decayCurve": -0.3,
            "modEnv1.releaseCurve": -0.3,
            "modEnv1>filter1.cutoff": 0.8,
            "vol.level": 1,
        },
    },
];
