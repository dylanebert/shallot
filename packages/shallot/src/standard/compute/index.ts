import { resource, FrameSync, type Plugin, type State } from "../../engine";
import { ComputeGraph } from "./graph";
import { requestGPU } from "./device";

export { ComputeGraph, SubGraph, beginComputePass } from "./graph";
export type { ExecutionContext, ComputeNode } from "./graph";

export { type BufferView, type Binding, type GBuf, bindView, binding, gbuf, view } from "./buffer";

export interface ComputeResources {
    textures: Map<string, GPUTexture>;
    textureViews: Map<string, GPUTextureView>;
    buffers: Map<string, GPUBuffer>;
}

export interface Compute {
    readonly device: GPUDevice;
    readonly graph: ComputeGraph;
    readonly resources: ComputeResources;
    frameIndex: number;
    readonly pending: number;
    sync(): Promise<void> | null;
}

export const Compute = resource<Compute>("compute");
export const SharedDevice = resource<GPUDevice>("shared-device");

export const ComputePlugin: Plugin = {
    name: "Compute",
    async initialize(state: State, onProgress?: (progress: number) => void) {
        const existing = state.getResource(SharedDevice);
        const device = existing ?? (await requestGPU());

        const graph = new ComputeGraph();
        const resources: ComputeResources = {
            textures: new Map(),
            textureViews: new Map(),
            buffers: new Map(),
        };

        const fences: Promise<void>[] = [];
        const compute: Compute = {
            device,
            graph,
            resources,
            frameIndex: 0,
            get pending() {
                return fences.length;
            },
            sync() {
                fences.push(device.queue.onSubmittedWorkDone());
                if (fences.length >= 2) return fences.shift()!;
                return null;
            },
        };
        state.setResource(Compute, compute);
        state.setResource(FrameSync, () => compute.sync());
        onProgress?.(1);
    },

    async warm(state: State, onProgress?: (progress: number) => void) {
        const compute = Compute.from(state);
        if (!compute) return;

        await compute.graph.prepare(compute.device, (done, total) => {
            onProgress?.(done / total);
        });
    },
};
