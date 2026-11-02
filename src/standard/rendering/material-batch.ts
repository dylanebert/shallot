import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { World } from "../../engine";
import { MaterialTypes } from "./material-type";

/** Dense draw batches resolve pipelines through their material type. @internal */
export const MaterialBatches = {
    create: () => ({
        types: [] as number[],
        capacities: [] as number[],
        rows: new Uint32Array(0),
        buffer: null as ReturnType<typeof createBuffer> | null,
    }),
};

function createBuffer(world: World, rows: Uint32Array) {
    return world.gpu.root.createBuffer(d.arrayOf(d.u32, rows.length), rows).$usage("storage");
}

/** Type offsets precede row-to-batch entries in one storage binding. @internal */
export function syncMaterialBatches(world: World): boolean {
    const state = world.resource(MaterialBatches);
    const { types, sources } = world.resource(MaterialTypes);
    let changed = state.types.length !== types.length;
    let length = types.length + 2;
    for (let type = 0; type < types.length; type++) {
        const capacity = Math.max(1, sources[type]?.table.capacity ?? 0);
        if (state.capacities[type] !== capacity) changed = true;
        state.capacities[type] = capacity;
        length += capacity;
    }
    if (!changed) return false;
    const rows = new Uint32Array(length);
    rows[0] = types.length;
    let offset = types.length + 2;
    for (let type = 0; type < types.length; type++) {
        state.types[type] = type;
        rows[type + 1] = offset;
        rows.fill(type, offset, offset + state.capacities[type]!);
        offset += state.capacities[type]!;
    }
    rows[types.length + 1] = offset;
    state.rows = rows;
    if (state.buffer) state.buffer.destroy();
    state.buffer = createBuffer(world, rows);
    return true;
}

/** Resolve the same (type, material row) key used by producer and entity kernels. @internal */
export function materialBatch(world: World, type: number, row: number): number {
    const { rows } = world.resource(MaterialBatches);
    if (type >= rows[0]! || row >= rows[type + 2]! - rows[type + 1]!) return 0xffffffff;
    return rows[rows[type + 1]! + row]!;
}

export function batchResolver(layout: {
    readonly $: { readonly materialBatches: readonly number[] };
}) {
    return tgpu.fn(
        [d.u32, d.u32],
        d.u32,
    )((type, row) => {
        "use gpu";
        if (type >= layout.$.materialBatches[0]) return 0xffffffff;
        const start = layout.$.materialBatches[type + 1];
        const end = layout.$.materialBatches[type + 2];
        if (row >= end - start) return 0xffffffff;
        return layout.$.materialBatches[start + row];
    });
}
