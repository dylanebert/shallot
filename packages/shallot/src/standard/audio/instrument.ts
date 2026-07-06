import { Registry } from "../../engine";

/** a DSP node kind in an instrument DAG: an `oscillator`, `filter`, `envelope`, `gain`, two-input `mix`,
 *  `constant`, or PCM `sample` source. */
export type NodeType =
    | "oscillator"
    | "filter"
    | "envelope"
    | "gain"
    | "mix"
    | "constant"
    | "sample";

const NODE_TYPE_ID: Record<NodeType, number> = {
    oscillator: 1,
    filter: 2,
    envelope: 3,
    gain: 4,
    mix: 5,
    constant: 6,
    sample: 7,
};

const NODE_DEFAULTS: Record<NodeType, number[]> = {
    oscillator: [440, 0, 0, 0.7],
    filter: [22050, Math.SQRT1_2, 0, 0],
    envelope: [0.01, 0.1, 0.7, 0.3, 0, -0.3, -0.3],
    gain: [1],
    mix: [0.5],
    constant: [0],
    sample: [0, 1, 0, 1, 0],
};

const NODE_PARAM_NAMES: Record<NodeType, string[]> = {
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
    sample: ["bufferId", "rate", "loop", "volume", "channel"],
};

const DISCRETE_PARAMS = new Set([
    "oscillator.waveform",
    "filter.mode",
    "sample.bufferId",
    "sample.loop",
    "sample.channel",
]);

/** a modulation route in an instrument DAG. A `source` node's output drives a `target` node's `param`,
 *  applied `linear` or per-`semitone`. */
export interface ModulationDef {
    source: string;
    target: string;
    param: string;
    mode?: "linear" | "semitone";
}

/** one node in an instrument DAG: its {@link NodeType} plus the node names feeding it: `input`, and
 *  `inputB` for the second input of a `mix`. */
export interface NodeDef {
    type: NodeType;
    input?: string;
    inputB?: string;
}

/**
 * a sound as a declarative DAG of typed nodes (oscillators, filters, envelopes,
 * samples) plus modulation routing. `instrument()` compiles it to a flat eval
 * order for the kernel. `volumeParam` / `pitchParams` / `loopParam` name the
 * params the ECS layer drives per voice (volume firehose, pitch, sample loop)
 */
export interface InstrumentDef {
    nodes: Record<string, NodeDef>;
    output: string;
    // right-channel output for a stereo instrument (two parallel chains); omit for mono
    outputR?: string;
    modulations?: ModulationDef[];
    values?: Record<string, number>;
    // a stereo instrument lists one volume/loop param per chain (left, right)
    volumeParam?: string | string[];
    pitchParams?: string[];
    loopParam?: string | string[];
}

interface Node {
    type: number;
    inputBuf: number;
    inputBufB: number;
    outputBuf: number;
    paramOffset: number;
}

export interface ModInfo {
    sourceBuf: number;
    targetNode: number;
    targetParam: number;
    depthParam: number;
    mode: number;
}

/** a per-voice frequency param resolved once at compile from the values map */
export interface PitchEntry {
    offset: number;
    baseFreq: number;
    octave: number;
    semitone: number;
    fine: number;
}

/**
 * immutable compiled topology for one instrument, carrying its registry `name`
 * + a `version` bumped on re-author (so the kernel re-receives topology only on
 * change). The ECS-driven param offsets (`volumeOffsets` / `baseVolume` /
 * `pitchEntries` / `loopOffsets`) are resolved here once so the per-frame
 * firehose reads them with no string work and no per-voice mirror. A stereo
 * instrument carries one volume/loop offset per chain; `outputBufR` equals
 * `outputBuf` for mono (the kernel reads the same buffer into both bus channels)
 */
export interface Instrument {
    name: string;
    version: number;
    nodes: Node[];
    outputBuf: number;
    outputBufR: number;
    paramLayout: Map<string, number>;
    totalParams: number;
    modulations: ModInfo[];
    volumeOffsets: number[];
    baseVolume: number;
    pitchEntries: PitchEntry[];
    loopOffsets: number[];
}

const NO_BUF = 0xff;
/**
 * the kernel's fixed instrument-table size: the number of distinct instrument *definitions* that can be
 * registered (not concurrent voices; the 64-slot pool bounds polyphony). Registering past it warns and
 * plays silent. Mirrors `rust/audio/src/graph.rs`'s `[InstrumentDef; MAX_INSTRUMENTS]`.
 */
export const MAX_INSTRUMENTS = 32;
const MAX_BUFFERS = 8;

/** every compiled instrument, keyed by name with a stable numeric ID */
export const Instruments: Registry<Instrument> = new Registry<Instrument>();

// mutable param values per instrument id, parallel to the registry. Kept apart
// from the immutable Instrument so a value tweak doesn't recompile
const instrumentValues: Map<string, number>[] = [];
let _version = 0;
let _anon = 0;

/** offset/value pairs for an instrument's WASM params (metadata params excluded) */
export function getParamPairs(id: number): [number, number][] {
    const compiled = byId(id);
    const values = instrumentValues[id];
    if (!compiled || !values) return [];
    const pairs: [number, number][] = [];
    for (const [key, value] of values) {
        const entry = compiled.paramLayout.get(key);
        if (entry !== undefined) pairs.push([entry, value]);
    }
    return pairs;
}

/** compiled instrument by numeric id. `name(id)` is an array index, `get` a map read */
export function byId(id: number): Instrument | undefined {
    const name = Instruments.name(id);
    return name === undefined ? undefined : Instruments.get(name);
}

interface GraphResult {
    sorted: string[];
    sortedIdx: Map<string, number>;
    adjacency: Map<string, string[]>;
}

function buildGraph(graph: InstrumentDef): GraphResult {
    const nodeNames = Object.keys(graph.nodes);
    const mods = graph.modulations ?? [];

    if (!graph.nodes[graph.output]) {
        throw new Error(`output node "${graph.output}" not found`);
    }

    const adjacency = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    for (const n of nodeNames) {
        adjacency.set(n, []);
        inDegree.set(n, 0);
    }

    for (const n of nodeNames) {
        const def = graph.nodes[n];
        if (def.input) {
            if (!graph.nodes[def.input])
                throw new Error(`node "${n}" references unknown input "${def.input}"`);
            adjacency.get(def.input)!.push(n);
            inDegree.set(n, inDegree.get(n)! + 1);
        }
        if (def.inputB) {
            if (!graph.nodes[def.inputB])
                throw new Error(`node "${n}" references unknown inputB "${def.inputB}"`);
            adjacency.get(def.inputB)!.push(n);
            inDegree.set(n, inDegree.get(n)! + 1);
        }
    }

    for (const mod of mods) {
        if (mod.source === mod.target) {
            throw new Error("self-modulation not allowed");
        }
        if (!graph.nodes[mod.source])
            throw new Error(`modulation references unknown source "${mod.source}"`);
        if (!graph.nodes[mod.target])
            throw new Error(`modulation references unknown target "${mod.target}"`);
        if (DISCRETE_PARAMS.has(`${graph.nodes[mod.target].type}.${mod.param}`)) {
            throw new Error(`cannot modulate discrete param "${mod.param}"`);
        }
        adjacency.get(mod.source)!.push(mod.target);
        inDegree.set(mod.target, inDegree.get(mod.target)! + 1);
    }

    const sorted: string[] = [];
    const queue: string[] = [];
    for (const n of nodeNames) {
        if (inDegree.get(n) === 0) queue.push(n);
    }
    while (queue.length > 0) {
        const n = queue.shift()!;
        sorted.push(n);
        for (const dep of adjacency.get(n)!) {
            const d = inDegree.get(dep)! - 1;
            inDegree.set(dep, d);
            if (d === 0) queue.push(dep);
        }
    }
    if (sorted.length !== nodeNames.length) {
        throw new Error("cycle detected in instrument graph");
    }

    const sortedIdx = new Map<string, number>();
    for (let j = 0; j < sorted.length; j++) sortedIdx.set(sorted[j], j);

    return { sorted, sortedIdx, adjacency };
}

function allocateBuffers(
    sorted: string[],
    graph: InstrumentDef,
    adjacency: Map<string, string[]>,
    sortedIdx: Map<string, number>,
    mods: ModulationDef[],
): Map<string, number> {
    const nodeBufOutput = new Map<string, number>();
    const lastConsumer = new Map<number, number>();
    let nextBuf = 0;

    function allocBuf(nodeIdx: number): number {
        for (const [buf, lastIdx] of lastConsumer) {
            if (lastIdx === -1) {
                lastConsumer.delete(buf);
                lastConsumer.set(buf, nodeIdx);
                return buf;
            }
        }
        const buf = nextBuf++;
        lastConsumer.set(buf, nodeIdx);
        return buf;
    }

    const modTargetToSources = new Map<string, string[]>();
    for (const mod of mods) {
        let sources = modTargetToSources.get(mod.target);
        if (!sources) {
            sources = [];
            modTargetToSources.set(mod.target, sources);
        }
        if (!sources.includes(mod.source)) sources.push(mod.source);
    }

    for (let i = 0; i < sorted.length; i++) {
        const n = sorted[i];
        const def = graph.nodes[n];

        for (const inputName of [def.input, def.inputB]) {
            if (!inputName) continue;
            const inputBuf = nodeBufOutput.get(inputName)!;
            if (adjacency.get(inputName)!.every((c) => sortedIdx.get(c)! <= i || c === n)) {
                lastConsumer.set(inputBuf, -1);
            }
        }

        const modSources = modTargetToSources.get(n);
        if (modSources) {
            for (const src of modSources) {
                const srcBuf = nodeBufOutput.get(src);
                if (srcBuf !== undefined) {
                    const cur = lastConsumer.get(srcBuf);
                    if (cur === undefined || cur < i) {
                        lastConsumer.set(srcBuf, i);
                    }
                }
            }
        }

        const outBuf = allocBuf(i);
        nodeBufOutput.set(n, outBuf);
    }

    if (nextBuf > MAX_BUFFERS) {
        throw new Error(`instrument graph requires ${nextBuf} buffers (max ${MAX_BUFFERS})`);
    }

    return nodeBufOutput;
}

interface LayoutResult {
    nodes: Node[];
    outputBuf: number;
    paramLayout: Map<string, number>;
    totalParams: number;
    modulations: ModInfo[];
    values: Map<string, number>;
}

function compileLayout(
    sorted: string[],
    graph: InstrumentDef,
    sortedIdx: Map<string, number>,
    nodeBufOutput: Map<string, number>,
): LayoutResult {
    const mods = graph.modulations ?? [];
    let paramOffset = 0;
    const paramLayout = new Map<string, number>();
    const compiledNodes: Node[] = [];

    for (const n of sorted) {
        const def = graph.nodes[n];
        const typeId = NODE_TYPE_ID[def.type];
        const paramNames = NODE_PARAM_NAMES[def.type];

        for (let pi = 0; pi < paramNames.length; pi++) {
            paramLayout.set(`${n}.${paramNames[pi]}`, paramOffset + pi);
        }

        compiledNodes.push({
            type: typeId,
            inputBuf: def.input ? nodeBufOutput.get(def.input)! : NO_BUF,
            inputBufB: def.inputB ? nodeBufOutput.get(def.inputB)! : NO_BUF,
            outputBuf: nodeBufOutput.get(n)!,
            paramOffset,
        });

        paramOffset += paramNames.length;
    }

    const values = new Map<string, number>();
    values.set("octave", 0);
    values.set("semitone", 0);
    values.set("fine", 0);
    values.set("volume", 0.7);
    for (const n of sorted) {
        const def = graph.nodes[n];
        const defaults = NODE_DEFAULTS[def.type];
        const paramNames = NODE_PARAM_NAMES[def.type];
        for (let i = 0; i < paramNames.length; i++) {
            values.set(`${n}.${paramNames[i]}`, defaults[i]);
        }
        if (def.type === "oscillator") {
            values.set(`${n}.octave`, 0);
            values.set(`${n}.semitone`, 0);
            values.set(`${n}.fine`, 0);
        }
    }

    const compiledMods: ModInfo[] = [];
    for (const mod of mods) {
        const targetNodeIdx = sortedIdx.get(mod.target)!;
        const targetParamNames = NODE_PARAM_NAMES[graph.nodes[mod.target].type];
        const paramIdx = targetParamNames.indexOf(mod.param);
        if (paramIdx === -1) {
            throw new Error(
                `modulation target param "${mod.param}" not found on node "${mod.target}"`,
            );
        }
        const targetParamOffset = compiledNodes[targetNodeIdx].paramOffset + paramIdx;
        const depthOffset = paramOffset;
        const depthKey = `${mod.source}>${mod.target}.${mod.param}`;
        paramLayout.set(depthKey, depthOffset);
        values.set(depthKey, 0);
        paramOffset += 1;

        compiledMods.push({
            sourceBuf: nodeBufOutput.get(mod.source)!,
            targetNode: targetNodeIdx,
            targetParam: targetParamOffset,
            depthParam: depthOffset,
            mode: mod.mode === "semitone" ? 1 : 0,
        });
    }

    if (graph.values) {
        for (const [key, value] of Object.entries(graph.values)) {
            values.set(key, value);
        }
    }

    for (const key of paramLayout.keys()) {
        if (!values.has(key)) throw new Error(`param "${key}" has no default value`);
    }

    return {
        nodes: compiledNodes,
        outputBuf: nodeBufOutput.get(graph.output)!,
        paramLayout,
        totalParams: paramOffset,
        modulations: compiledMods,
        values,
    };
}

// resolve the ECS-driven param offsets from the values map once at compile, so
// the per-frame firehose reads numbers, not strings
function pitchEntries(graph: InstrumentDef, layout: LayoutResult): PitchEntry[] {
    const entries: PitchEntry[] = [];
    for (const pp of graph.pitchParams ?? []) {
        const offset = layout.paramLayout.get(pp);
        if (offset === undefined) continue;
        const node = pp.slice(0, pp.indexOf("."));
        entries.push({
            offset,
            baseFreq: layout.values.get(pp) ?? 440,
            octave: layout.values.get(`${node}.octave`) ?? 0,
            semitone: layout.values.get(`${node}.semitone`) ?? 0,
            fine: layout.values.get(`${node}.fine`) ?? 0,
        });
    }
    return entries;
}

// resolve a single-or-array firehose param to its compiled offsets (one per chain),
// dropping any that don't resolve so the per-frame loop stays string-free
function offsets(param: string | string[] | undefined, layout: LayoutResult): number[] {
    if (param === undefined) return [];
    const params = typeof param === "string" ? [param] : param;
    const out: number[] = [];
    for (const p of params) {
        const off = layout.paramLayout.get(p);
        if (off !== undefined) out.push(off);
    }
    return out;
}

function compileInstrument(
    graph: InstrumentDef,
    name: string,
    version: number,
): { compiled: Instrument; values: Map<string, number> } {
    const mods = graph.modulations ?? [];
    const { sorted, sortedIdx, adjacency } = buildGraph(graph);
    const nodeBufOutput = allocateBuffers(sorted, graph, adjacency, sortedIdx, mods);
    const layout = compileLayout(sorted, graph, sortedIdx, nodeBufOutput);

    const firstVolume =
        typeof graph.volumeParam === "string" ? graph.volumeParam : graph.volumeParam?.[0];
    const { values, ...topology } = layout;
    const compiled: Instrument = {
        name,
        version,
        ...topology,
        // mono routes the left buffer into both bus channels (bit-for-bit unchanged)
        outputBufR: graph.outputR ? nodeBufOutput.get(graph.outputR)! : layout.outputBuf,
        volumeOffsets: offsets(graph.volumeParam, layout),
        baseVolume: firstVolume ? (values.get(firstVolume) ?? 1) : 1,
        pitchEntries: pitchEntries(graph, layout),
        loopOffsets: offsets(graph.loopParam, layout),
    };
    return { compiled, values };
}

/**
 * compile a sound DAG and register it under `name` (auto-named when omitted),
 * returning a stable id. Re-authoring the same name bumps the version in place;
 * a graph exceeding the kernel's instrument cap warns and returns -1
 *
 * @example
 * instrument({ nodes: { src: { type: "sample" } }, output: "src" }, "boom");
 */
export function instrument(graph: InstrumentDef, name?: string): number {
    const n = name ?? `instrument-${_anon++}`;
    if (!Instruments.has(n) && Instruments.size >= MAX_INSTRUMENTS) {
        console.warn(`audio: instrument cap reached (${MAX_INSTRUMENTS})`);
        return -1;
    }
    const { compiled, values } = compileInstrument(graph, n, ++_version);
    const id = Instruments.register(compiled);
    instrumentValues[id] = values;
    return id;
}
