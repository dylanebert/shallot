import { RenderContext } from "../../core/rendering";
import type { World } from "../../engine";
import {
    dispatchWorkgroups,
    dispatchWorkgroupsIndirect,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import { countLayout, scatterLayout } from "./pack";
import {
    ProducerDispatch,
    ProducerParams,
    producerArgsKernel,
    producerArgsLayout,
    producerCountKernel,
    producerCullLayout,
    producerScatterKernel,
    producerTables,
} from "./producer-table";

interface BoundProducer {
    args: GPUBuffer;
    params: GPUBuffer;
    argsGroup: GPUBindGroup;
    cullGroup: GPUBindGroup;
}
const key = {
    create: () => ({
        inputs: [] as GPUBuffer[],
        tableCount: -1,
        basePairs: -1,
        argsPipe: null as GPUComputePipeline | null,
        countPipe: null as GPUComputePipeline | null,
        scatterPipe: null as GPUComputePipeline | null,
        countGroup: null as GPUBindGroup | null,
        scatterGroup: null as GPUBindGroup | null,
        producers: [] as BoundProducer[],
    }),
};

export function syncProducerPack(
    world: World,
    bounds: GPUBuffer,
    params: GPUBuffer,
    counts: GPUBuffer,
    drawArgs: GPUBuffer,
    packedEids: GPUBuffer,
    basePairs: number,
): void {
    const tables = world.resource(producerTables).tables;
    if (!tables.length) return;
    const state = world.resource(key);
    if (
        state.tableCount === tables.length &&
        state.basePairs === basePairs &&
        state.inputs[0] === bounds &&
        state.inputs[1] === params &&
        state.inputs[2] === counts &&
        state.inputs[3] === drawArgs &&
        state.inputs[4] === packedEids
    )
        return;
    const root = world.gpu.root;
    const device = world.gpu.device;
    if (!state.argsPipe) {
        state.argsPipe = root.unwrap(
            root
                .createComputePipeline({ compute: producerArgsKernel })
                .$name("shallot-producer-args"),
        );
        state.countPipe = root.unwrap(
            root
                .createComputePipeline({ compute: producerCountKernel })
                .$name("shallot-producer-count"),
        );
        state.scatterPipe = root.unwrap(
            root
                .createComputePipeline({ compute: producerScatterKernel })
                .$name("shallot-producer-scatter"),
        );
    }
    state.inputs = [bounds, params, counts, drawArgs, packedEids];
    state.basePairs = basePairs;
    state.tableCount = tables.length;
    state.countGroup = device.createBindGroup({
        layout: root.unwrap(countLayout),
        entries: [{ binding: 0, resource: { buffer: counts } }],
    });
    state.scatterGroup = device.createBindGroup({
        layout: root.unwrap(scatterLayout),
        entries: [
            { binding: 0, resource: { buffer: drawArgs } },
            { binding: 1, resource: { buffer: packedEids } },
        ],
    });
    for (let i = 0; i < tables.length; i++) {
        const table = tables[i]!.table;
        let entry = state.producers[i];
        const args =
            entry?.args ??
            root.unwrap(root.createBuffer(ProducerDispatch).$usage("storage", "indirect"));
        const producer = root.unwrap(
            root
                .createBuffer(ProducerParams, {
                    capacity: table.capacity,
                    pairOffset: (i + 1) * basePairs,
                    basePairs,
                })
                .$usage("uniform"),
        );
        const cullVolumes = world.resource(RenderContext).cullVolumes!;
        const argsGroup = device.createBindGroup({
            layout: root.unwrap(producerArgsLayout),
            entries: [
                { binding: 0, resource: { buffer: root.unwrap(table.count) } },
                { binding: 1, resource: { buffer: args } },
                { binding: 2, resource: { buffer: producer } },
                { binding: 3, resource: { buffer: params } },
            ],
        });
        const cullGroup = device.createBindGroup({
            layout: root.unwrap(producerCullLayout),
            entries: [
                { binding: 0, resource: { buffer: root.unwrap(table.meshInstances) } },
                { binding: 1, resource: { buffer: root.unwrap(table.placements) } },
                { binding: 2, resource: { buffer: bounds } },
                { binding: 3, resource: { buffer: args } },
                { binding: 4, resource: { buffer: producer } },
                { binding: 5, resource: { buffer: params } },
                { binding: 6, resource: { buffer: cullVolumes } },
            ],
        });
        if (entry) world.retireGpuBuffer(entry.params);
        entry = { args, params: producer, argsGroup, cullGroup };
        state.producers[i] = entry;
    }
}

export function countProducers(world: World, pass: GPUComputePassEncoder): void {
    const state = world.resource(key);
    for (let i = 0; i < state.producers.length; i++) {
        const producer = state.producers[i]!;
        setComputePipeline(pass, state.argsPipe!);
        setComputeBindGroup(pass, 0, producer.argsGroup);
        dispatchWorkgroups(pass, 1);
        setComputePipeline(pass, state.countPipe!);
        setComputeBindGroup(pass, 0, producer.cullGroup);
        setComputeBindGroup(pass, 1, state.countGroup!);
        dispatchWorkgroupsIndirect(pass, producer.args, 0);
    }
}
export function scatterProducers(world: World, pass: GPUComputePassEncoder): void {
    const state = world.resource(key);
    for (let i = 0; i < state.producers.length; i++) {
        const producer = state.producers[i]!;
        setComputePipeline(pass, state.scatterPipe!);
        setComputeBindGroup(pass, 0, producer.cullGroup);
        setComputeBindGroup(pass, 1, state.scatterGroup!);
        dispatchWorkgroupsIndirect(pass, producer.args, 0);
    }
}

export function producerCapacity(world: World): number {
    const tables = world.resource(producerTables).tables;
    let capacity = 0;
    for (let i = 0; i < tables.length; i++) capacity += tables[i]!.table.capacity;
    return capacity;
}
