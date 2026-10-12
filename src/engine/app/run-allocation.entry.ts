import { type Plugin, runApp, type World } from "@dylanebert/shallot";
import { gpuAllocationExpressions } from "../runtime/allocation-site";

export let controlSink: { frame: number } | undefined;
export const control = () => {
    controlSink = { frame: 0 };
};

function draw(world: World) {
    world.frameEncoder();
}

export default async function create() {
    const requestAnimationFrameDescriptor = Object.getOwnPropertyDescriptor(
        globalThis,
        "requestAnimationFrame",
    );
    let callback: FrameRequestCallback | undefined;
    Object.defineProperty(globalThis, "requestAnimationFrame", {
        configurable: true,
        writable: true,
        value: (next: FrameRequestCallback) => {
            callback = next;
            return 0;
        },
    });

    // Keep the array generic-elements so each callback reuses a pre-boxed timestamp HeapNumber.
    const timestamps: (number | undefined)[] = [undefined];
    for (let i = 0; i < 30_000; i++) timestamps.push(((i + 0.25) * 1000) / 60);
    let timestamp = 1;
    const plugin: Plugin = {
        name: "RunAllocation",
        gpu: {},
        systems: [
            {
                name: "draw",
                group: "draw",
                update: draw,
            },
        ],
    };

    try {
        const app = await runApp({ defaults: false, plugins: [plugin] });
        const world = app.world;
        function step() {
            const next = callback!;
            callback = undefined;
            next(timestamps[timestamp++]!);
        }
        return {
            step,
            allocationExpressions: gpuAllocationExpressions,
            // Keep these seams visible so the sampler does not attribute engine work to its harness.
            allocationFunctions: [world.gpu.sync, step, draw],
            wait: () => world.gpu.device.queue.onSubmittedWorkDone(),
            dispose() {
                app.dispose();
                if (requestAnimationFrameDescriptor)
                    Object.defineProperty(
                        globalThis,
                        "requestAnimationFrame",
                        requestAnimationFrameDescriptor,
                    );
                else Reflect.deleteProperty(globalThis, "requestAnimationFrame");
            },
        };
    } catch (error) {
        if (requestAnimationFrameDescriptor)
            Object.defineProperty(
                globalThis,
                "requestAnimationFrame",
                requestAnimationFrameDescriptor,
            );
        else Reflect.deleteProperty(globalThis, "requestAnimationFrame");
        throw error;
    }
}
