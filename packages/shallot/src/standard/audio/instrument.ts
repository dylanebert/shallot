import { registry } from "../../engine";

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
    sample: [0, 1, 0, 1],
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
    sample: ["bufferId", "rate", "loop", "volume"],
};

const DISCRETE_PARAMS = new Set([
    "oscillator.waveform",
    "filter.mode",
    "sample.bufferId",
    "sample.loop",
]);

export interface ModulationDef {
    source: string;
    target: string;
    param: string;
    mode?: "linear" | "semitone";
}

export interface NodeDef {
    type: NodeType;
    input?: string;
    inputB?: string;
}

export interface Instrument {
    nodes: Record<string, NodeDef>;
    output: string;
    modulations?: ModulationDef[];
    values?: Record<string, number>;
    volumeParam?: string;
    pitchParams?: string[];
}

interface CompiledNode {
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

export interface CompiledInstrument {
    nodes: CompiledNode[];
    outputBuf: number;
    paramLayout: Map<string, number>;
    totalParams: number;
    modulations: ModInfo[];
    version: number;
    volumeParam?: string;
    pitchParams?: string[];
}

const NO_BUF = 0xff;
export const MAX_INSTRUMENTS = 16;
export const MAX_BUFFERS = 8;

export const instrumentRegistry = registry<CompiledInstrument>(MAX_INSTRUMENTS);
const instrumentValues: Map<string, number>[] = [];

export function getValues(instId: number): Map<string, number> | undefined {
    return instrumentValues[instId];
}

export function getParamPairs(instId: number): [number, number][] {
    const compiled = instrumentRegistry.get(instId);
    const values = instrumentValues[instId];
    if (!compiled || !values) return [];
    const pairs: [number, number][] = [];
    for (const [key, value] of values) {
        const entry = compiled.paramLayout.get(key);
        if (entry !== undefined) pairs.push([entry, value]);
    }
    return pairs;
}

export function setValues(instId: number, params: Record<string, number>): void {
    const values = instrumentValues[instId];
    if (!values) return;
    for (const [key, value] of Object.entries(params)) values.set(key, value);
}

interface GraphResult {
    sorted: string[];
    sortedIdx: Map<string, number>;
    adjacency: Map<string, string[]>;
}

function buildGraph(graph: Instrument): GraphResult {
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
    graph: Instrument,
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
    nodes: CompiledNode[];
    outputBuf: number;
    paramLayout: Map<string, number>;
    totalParams: number;
    modulations: ModInfo[];
    values: Map<string, number>;
}

function compileLayout(
    sorted: string[],
    graph: Instrument,
    sortedIdx: Map<string, number>,
    nodeBufOutput: Map<string, number>,
): LayoutResult {
    const mods = graph.modulations ?? [];
    let paramOffset = 0;
    const paramLayout = new Map<string, number>();
    const compiledNodes: CompiledNode[] = [];

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

function compileInstrument(
    graph: Instrument,
    version: number,
): { compiled: CompiledInstrument; values: Map<string, number> } {
    const mods = graph.modulations ?? [];
    const { sorted, sortedIdx, adjacency } = buildGraph(graph);
    const nodeBufOutput = allocateBuffers(sorted, graph, adjacency, sortedIdx, mods);
    const layout = compileLayout(sorted, graph, sortedIdx, nodeBufOutput);

    const { values, ...topology } = layout;
    const compiled: CompiledInstrument = {
        ...topology,
        version,
        volumeParam: graph.volumeParam,
        pitchParams: graph.pitchParams,
    };
    return { compiled, values };
}

export function instrument(graph: Instrument, name?: string): number {
    const { compiled, values } = compileInstrument(graph, instrumentRegistry.version + 1);
    const existing = name ? instrumentRegistry.getByName(name) : undefined;
    if (existing !== undefined) {
        instrumentRegistry.set(existing, compiled);
        instrumentValues[existing] = values;
        return existing;
    }
    if (instrumentRegistry.count() >= MAX_INSTRUMENTS) {
        console.warn(`audio: instrument cap reached (${MAX_INSTRUMENTS})`);
        return -1;
    }
    const id = instrumentRegistry.add(compiled, name);
    instrumentValues[id] = values;
    return id;
}

export function clearInstruments(): void {
    instrumentRegistry.clear();
    instrumentValues.length = 0;
}
