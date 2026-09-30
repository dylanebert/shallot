export let controlSink: { frame: number } | undefined;
export const control = () => { controlSink = { frame: 0 }; };

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label}: 1000 ms deadline`)), 1000);
        promise.then((value) => { clearTimeout(timer); resolve(value); },
            (error) => { clearTimeout(timer); reject(error); });
    });
}

// A diagnostic control, not an engine replacement: no app, ECS, interpolation, descriptors or arrays
// created in the stepped path. The only fresh objects in copy are WebGPU's encoder/command handles.
export default async function create(mode: string) {
    const adapter = await bounded("copy adapter", navigator.gpu.requestAdapter());
    if (!adapter) throw new Error("copy allocation probe requires an adapter");
    const device = await bounded("copy device", adapter.requestDevice());
    const current = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const previous = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST });
    const descriptor: GPUCommandEncoderDescriptor = {};
    const commands: GPUCommandBuffer[] = new Array(1);
    const bytes = new Float32Array(12);
    const copy = () => {
        const encoder = device.createCommandEncoder(descriptor);
        encoder.copyBufferToBuffer(current, 0, previous, 0, 48);
        commands[0] = encoder.finish();
        device.queue.submit(commands);
    };
    const upload = () => device.queue.writeBuffer(current, 0, bytes);
    return {
        step: mode === "copy" ? copy : upload,
        wait: () => bounded("copy completion", device.queue.onSubmittedWorkDone()),
        dispose: () => { current.destroy(); previous.destroy(); device.destroy(); },
    };
}
