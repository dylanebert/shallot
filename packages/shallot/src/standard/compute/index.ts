import { resource, type Plugin, type State, type System } from "../../engine";
import { requestGPU, type GPUCapabilities } from "./device";
import { ComputeGraph } from "./graph";
import { Profiler } from "./profile";

export interface Compute {
    device: GPUDevice;
    capabilities: GPUCapabilities;
    graph: ComputeGraph;
    profiler: Profiler;
}

export const Compute = resource<Compute>("compute");

/**
 * Get the precision type for shader compilation based on GPU capabilities.
 * Uses f16 if available for better performance, falls back to f32 for older GPUs.
 */
export function getPrecisionType(capabilities: GPUCapabilities): "f16" | "f32" {
    return capabilities.supportsF16 ? "f16" : "f32";
}

const ComputeSystem: System = {
    group: "compute",
    annotations: { mode: "always" },
    first: true,

    update(state: State) {
        const compute = Compute.from(state);
        if (!compute) return;
        compute.graph.execute(state, compute.device);
    },
};

export const ComputePlugin: Plugin = {
    name: "Compute",
    systems: [ComputeSystem],
    async initialize(state: State, onProgress?: (progress: number) => void) {
        const { device, capabilities } = await requestGPU();
        const profiler = new Profiler(device, capabilities.supportsTimestampQuery);
        const graph = new ComputeGraph();

        state.setResource(Compute, {
            device,
            capabilities,
            graph,
            profiler,
        });

        onProgress?.(1);
    },
};

export { ComputeGraph } from "./graph";
export type { ComputeNode, ExecutionContext } from "./graph";
export { Profiler } from "./profile";
export { createReadback, readback } from "./readback";
export { gbuf, view, type GBuf, type BufferView, CHUNK_SHIFT, CHUNK_MASK } from "./buffer";
export { write, type GPUCapabilities } from "./device";
