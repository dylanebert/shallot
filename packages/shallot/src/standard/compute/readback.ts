export async function readBuffer(
    device: GPUDevice,
    source: GPUBuffer,
    size: number,
): Promise<ArrayBuffer> {
    const staging = device.createBuffer({
        label: "staging-readback",
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();

    return data;
}

export async function readFloat32(
    device: GPUDevice,
    buffer: GPUBuffer,
    count: number,
): Promise<Float32Array> {
    const data = await readBuffer(device, buffer, count * 4);
    return new Float32Array(data);
}

export async function readUint32(
    device: GPUDevice,
    buffer: GPUBuffer,
    count: number,
): Promise<Uint32Array> {
    const data = await readBuffer(device, buffer, count * 4);
    return new Uint32Array(data);
}
