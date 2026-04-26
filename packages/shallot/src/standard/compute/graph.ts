import { CycleError } from "../../engine/ecs/core";

const computePassDesc: GPUComputePassDescriptor = {};

export function beginComputePass(
    encoder: GPUCommandEncoder,
    ts?: GPUComputePassTimestampWrites,
): GPUComputePassEncoder {
    computePassDesc.timestampWrites = ts;
    return encoder.beginComputePass(computePassDesc);
}

export interface ExecutionContext {
    readonly device: GPUDevice;
    readonly queue: GPUQueue;
    readonly encoder: GPUCommandEncoder;
    readonly context: GPUCanvasContext;
    readonly format: GPUTextureFormat;
    readonly canvasView: GPUTextureView;
    readonly timestampWrites?: (name: string) => GPURenderPassTimestampWrites | undefined;
    getTexture(name: string): GPUTexture | null;
    getTextureView(name: string): GPUTextureView | null;
    getBuffer(name: string): GPUBuffer | null;
    setTexture(name: string, texture: GPUTexture): void;
    setTextureView(name: string, view: GPUTextureView): void;
    setBuffer(name: string, buffer: GPUBuffer): void;
    afterSubmit(fn: () => void): void;
    readonly subGraph: string;
}

export interface ComputeNode {
    readonly name: string;
    readonly inputs: readonly string[];
    readonly outputs: readonly string[];
    readonly scope?: "frame" | "view";
    readonly sync?: boolean;
    readonly execute: (ctx: ExecutionContext) => void;
    readonly prepare?: (device: GPUDevice) => Promise<void>;
}

interface ExecutionPlan {
    readonly frame: readonly ComputeNode[];
    readonly view: readonly ComputeNode[];
}

function buildEdges(nodes: ComputeNode[]): [ComputeNode, ComputeNode][] {
    const edges: [ComputeNode, ComputeNode][] = [];
    const fresh = new Map<string, ComputeNode[]>();
    const transformers = new Map<string, ComputeNode[]>();

    for (const node of nodes) {
        for (const output of node.outputs) {
            const map = node.inputs.includes(output) ? transformers : fresh;
            let arr = map.get(output);
            if (!arr) {
                arr = [];
                map.set(output, arr);
            }
            arr.push(node);
        }
    }

    for (const [name, trans] of transformers) {
        const writers = fresh.get(name) ?? [];
        for (let i = 0; i < trans.length; i++) {
            const t = trans[i];
            for (const f of writers) {
                if (f !== t) edges.push([f, t]);
            }
            for (let j = 0; j < i; j++) {
                if (trans[j] !== t) edges.push([trans[j], t]);
            }
        }
    }

    for (const node of nodes) {
        for (const input of node.inputs) {
            if (node.outputs.includes(input)) continue;
            const trans = transformers.get(input);
            if (trans && trans.length > 0) {
                const last = trans[trans.length - 1];
                if (last !== node) edges.push([last, node]);
            } else {
                const writers = fresh.get(input);
                if (!writers) continue;
                for (const f of writers) {
                    if (f !== node) edges.push([f, node]);
                }
            }
        }
    }

    return edges;
}

function topoSort(nodes: ComputeNode[]): ComputeNode[] {
    if (nodes.length === 0) return [];

    const edges = buildEdges(nodes);
    const adjacency = new Map<ComputeNode, ComputeNode[]>();
    const inDegree = new Map<ComputeNode, number>();

    for (const node of nodes) {
        adjacency.set(node, []);
        inDegree.set(node, 0);
    }

    for (const [from, to] of edges) {
        adjacency.get(from)!.push(to);
        inDegree.set(to, inDegree.get(to)! + 1);
    }

    const queue: ComputeNode[] = [];
    for (const node of nodes) {
        if (inDegree.get(node) === 0) {
            queue.push(node);
        }
    }

    const sorted: ComputeNode[] = [];
    let i = 0;

    while (i < queue.length) {
        const node = queue[i++];
        sorted.push(node);

        for (const dep of adjacency.get(node)!) {
            const newDegree = inDegree.get(dep)! - 1;
            inDegree.set(dep, newDegree);
            if (newDegree === 0) {
                queue.push(dep);
            }
        }
    }

    if (sorted.length !== nodes.length) {
        const remaining = nodes.filter((n) => (inDegree.get(n) ?? 0) > 0);
        const lines = remaining.map((n) => {
            const incoming = nodes.filter((m) => adjacency.get(m)?.includes(n)).map((m) => m.name);
            return `  ${n.name} <- ${incoming.length ? incoming.join(", ") : "(none)"}`;
        });
        throw new CycleError(
            `Circular dependency detected in compute graph. Nodes still in cycle:\n${lines.join("\n")}`,
        );
    }

    return sorted;
}

function compileNodes(sharedNodes: ComputeNode[], subGraphNodes: ComputeNode[]): ExecutionPlan {
    const allNodes = [...sharedNodes, ...subGraphNodes];
    if (allNodes.length === 0) {
        return { frame: [], view: [] };
    }

    const frameNodes: ComputeNode[] = [];
    const viewNodes: ComputeNode[] = [];

    for (const node of allNodes) {
        if (node.scope === "frame") {
            frameNodes.push(node);
        } else {
            viewNodes.push(node);
        }
    }

    const viewOutputs = new Set<string>();
    for (const node of viewNodes) {
        for (const output of node.outputs) {
            viewOutputs.add(output);
        }
    }
    for (const node of frameNodes) {
        for (const input of node.inputs) {
            if (viewOutputs.has(input)) {
                throw new Error(
                    `Frame-scope node '${node.name}' depends on view-scope resource '${input}'`,
                );
            }
        }
    }

    return {
        frame: topoSort(frameNodes),
        view: topoSort(viewNodes),
    };
}

export class SubGraph {
    readonly nodes = new Map<string, ComputeNode>();
    check: ((cameraEid: number) => boolean) | null = null;
    private readonly _graph: ComputeGraph;

    constructor(graph: ComputeGraph) {
        this._graph = graph;
    }

    add(node: ComputeNode): void {
        if (this.nodes.has(node.name)) {
            throw new Error(`Node '${node.name}' already exists`);
        }
        this.nodes.set(node.name, node);
        this._graph.invalidate();
    }

    set(name: string, node: ComputeNode): void {
        if (node.name !== name) {
            throw new Error(`Node name '${node.name}' must match slot name '${name}'`);
        }
        this.nodes.set(name, node);
        this._graph.invalidate();
    }

    remove(name: string): boolean {
        const removed = this.nodes.delete(name);
        if (removed) {
            this._graph.invalidate();
        }
        return removed;
    }
}

export class ComputeGraph {
    readonly nodes = new Map<string, ComputeNode>();
    private readonly _subGraphs = new Map<string, SubGraph>();
    private _plans = new Map<string, ExecutionPlan>();

    get planCached(): boolean {
        return this._plans.size > 0;
    }

    get subGraphs(): ReadonlyMap<string, SubGraph> {
        return this._subGraphs;
    }

    subGraph(name: string): SubGraph {
        let sg = this._subGraphs.get(name);
        if (!sg) {
            sg = new SubGraph(this);
            this._subGraphs.set(name, sg);
        }
        return sg;
    }

    add(node: ComputeNode): void {
        if (this.nodes.has(node.name)) {
            throw new Error(`Node '${node.name}' already exists`);
        }
        this.nodes.set(node.name, node);
        this.invalidate();
    }

    set(name: string, node: ComputeNode): void {
        if (node.name !== name) {
            throw new Error(`Node name '${node.name}' must match slot name '${name}'`);
        }
        this.nodes.set(name, node);
        this.invalidate();
    }

    remove(name: string): boolean {
        const removed = this.nodes.delete(name);
        if (removed) {
            this.invalidate();
        }
        return removed;
    }

    compile(subGraphName?: string): ExecutionPlan {
        const name = subGraphName ?? "";

        const cached = this._plans.get(name);
        if (cached) return cached;

        const shared = Array.from(this.nodes.values());
        const sgNodes = subGraphName
            ? Array.from(this._subGraphs.get(subGraphName)?.nodes.values() ?? [])
            : [];

        const plan = compileNodes(shared, sgNodes);
        this._plans.set(name, plan);

        return plan;
    }

    async prepare(
        device: GPUDevice,
        onProgress?: (done: number, total: number) => void,
    ): Promise<void> {
        const allNodes: ComputeNode[] = Array.from(this.nodes.values());
        for (const sg of this._subGraphs.values()) {
            for (const node of sg.nodes.values()) {
                allNodes.push(node);
            }
        }

        const preparable = allNodes.filter((n) => n.prepare);
        const total = preparable.length;
        if (total === 0) return;

        let done = 0;

        const promises = preparable.map(async (node) => {
            await node.prepare!(device);
            done++;
            onProgress?.(done, total);
        });

        await Promise.all(promises);
    }

    invalidate(): void {
        this._plans.clear();
    }
}
