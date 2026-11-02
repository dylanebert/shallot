import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { GpuTable, World } from "../../engine";
import { type MaterialBinding, type MaterialType, MaterialTypes } from "./material-type";

type ImageField = { name: string; sampler: string; isSrgb: boolean; offset: number };
type Rows = {
    layout: object | undefined;
    table: GpuTable<any> | undefined;
    capacity: number;
    fields: ImageField[];
    ids: Int32Array;
    next: Int32Array;
    previous: Int32Array;
    dirty: Uint8Array;
    pending: Int32Array;
    pendingCount: number;
    stop: () => void;
};

/** Dense, reusable batch slots; unreferenced slots are reused before growing the draw grid. @internal */
export const MaterialBatches = {
    create: () => ({
        types: [] as number[],
        representatives: [] as number[],
        entries: [] as Rows[],
        rows: new Uint32Array(0),
        buckets: new Int32Array(0),
        links: new Int32Array(0),
        counts: new Uint32Array(0),
        hashes: new Uint32Array(0),
        free: new Int32Array(0),
        freeCount: 0,
        bufferCapacity: 0,
        buffer: null as ReturnType<typeof createBuffer> | null,
    }),
};
type Batches = ReturnType<typeof MaterialBatches.create>;

function createBuffer(world: World, rows: Uint32Array) {
    return world.gpu.root.createBuffer(d.arrayOf(d.u32, rows.length), rows).$usage("storage");
}

function imageFields(type: MaterialType<any, any, any> | undefined): ImageField[] {
    return Object.entries(
        (type?.layout.bindingSpecs ?? {}) as Record<string, MaterialBinding>,
    ).flatMap(([name, binding]) =>
        binding.type === "image"
            ? [
                  {
                      name,
                      sampler: `${name}Sampler`,
                      isSrgb: binding.isSrgb !== false,
                      offset: d.memoryLayoutOf(type!.parameters, (row: any) => row[binding.field])
                          .offset,
                  },
              ]
            : [],
    );
}

function observe(world: World, entry: Rows): void {
    if (!entry.table || !entry.fields.length) return;
    entry.stop = entry.table.observeRows((row) => {
        // Growth is reconciled by a full rebuild before pending rows are consumed.
        if (row >= entry.capacity || entry.dirty[row]) return;
        entry.dirty[row] = 1;
        entry.pending[entry.pendingCount++] = row;
    });
    world.onDispose(entry.stop);
}

function rowStorage(capacity: number, fields: number) {
    return {
        capacity,
        ids: new Int32Array(capacity * fields),
        next: new Int32Array(capacity).fill(-1),
        previous: new Int32Array(capacity).fill(-1),
        dirty: new Uint8Array(capacity),
        pending: new Int32Array(capacity),
        pendingCount: 0,
    };
}

function entryFor(
    world: World,
    type: MaterialType<any, any, any> | undefined,
    table: GpuTable<any> | undefined,
): Rows {
    const capacity = Math.max(1, table?.capacity ?? 0);
    const fields = imageFields(type);
    const entry: Rows = {
        layout: type?.layout,
        table,
        fields,
        ...rowStorage(capacity, fields.length),
        stop: () => {},
    };
    observe(world, entry);
    return entry;
}

function readIds(entry: Rows, row: number, bytes: Uint8Array): boolean {
    let changed = false;
    const stride = entry.table!.rowBytes;
    for (let field = 0; field < entry.fields.length; field++) {
        const at = row * stride + entry.fields[field].offset;
        const id = bytes[at] | (bytes[at + 1] << 8) | (bytes[at + 2] << 16) | (bytes[at + 3] << 24);
        const index = row * entry.fields.length + field;
        if (entry.ids[index] !== id) {
            entry.ids[index] = id;
            changed = true;
        }
    }
    return changed;
}

function hashRow(type: number, entry: Rows, row: number): number {
    let hash = Math.imul(type + 1, 16777619);
    for (let field = 0; field < entry.fields.length; field++)
        hash = Math.imul(hash ^ entry.ids[row * entry.fields.length + field], 16777619);
    return hash >>> 0;
}

function sameKey(state: Batches, batch: number, type: number, entry: Rows, row: number): boolean {
    if (state.types[batch] !== type) return false;
    const representative = state.representatives[batch]!;
    for (let field = 0; field < entry.fields.length; field++)
        if (
            entry.ids[representative * entry.fields.length + field] !==
            entry.ids[row * entry.fields.length + field]
        )
            return false;
    return true;
}

function release(state: Batches, entry: Rows, row: number, batch: number): void {
    const next = entry.next[row];
    const previous = entry.previous[row];
    if (previous >= 0) entry.next[previous] = next;
    else state.representatives[batch] = next;
    if (next >= 0) entry.previous[next] = previous;
    if (--state.counts[batch]) return;
    const bucket = state.hashes[batch] & (state.buckets.length - 1);
    let prior = -1;
    let current = state.buckets[bucket];
    while (current !== batch) {
        prior = current;
        current = state.links[current];
    }
    if (prior < 0) state.buckets[bucket] = state.links[batch];
    else state.links[prior] = state.links[batch];
    state.free[state.freeCount++] = batch;
}

function assign(state: Batches, type: number, entry: Rows, row: number): number {
    const hash = hashRow(type, entry, row);
    const bucket = hash & (state.buckets.length - 1);
    let batch = state.buckets[bucket];
    while (batch >= 0 && (state.hashes[batch] !== hash || !sameKey(state, batch, type, entry, row)))
        batch = state.links[batch];
    if (batch < 0) {
        batch = state.freeCount ? state.free[--state.freeCount] : state.types.length;
        state.types[batch] = type;
        state.hashes[batch] = hash;
        state.links[batch] = state.buckets[bucket];
        state.buckets[bucket] = batch;
        state.representatives[batch] = -1;
    }
    const head = state.representatives[batch]!;
    entry.previous[row] = -1;
    entry.next[row] = head;
    if (head >= 0) entry.previous[head] = row;
    state.representatives[batch] = row;
    state.counts[batch]++;
    return batch;
}

type MovedRow = (world: World, type: number, row: number) => void;

function rebuild(
    world: World,
    state: Batches,
    typeCount: number,
    length: number,
    moved?: MovedRow,
): void {
    const previous = state.rows;
    let slots = 0;
    for (const entry of state.entries) {
        slots += entry.capacity;
    }
    let buckets = 1;
    while (buckets < slots * 2) buckets *= 2;
    state.buckets = new Int32Array(buckets).fill(-1);
    state.links = new Int32Array(slots);
    state.counts = new Uint32Array(slots);
    state.hashes = new Uint32Array(slots);
    state.free = new Int32Array(slots);
    state.freeCount = 0;
    state.types.length = 0;
    state.representatives.length = 0;
    state.rows = new Uint32Array(length);
    state.rows[0] = typeCount;
    let offset = typeCount + 2;
    for (let type = 0; type < typeCount; type++) {
        const entry = state.entries[type]!;
        state.rows[type + 1] = offset;
        entry.pendingCount = 0;
        entry.dirty.fill(0);
        entry.next.fill(-1);
        entry.previous.fill(-1);
        const bytes = entry.fields.length ? entry.table!.bytes : undefined;
        for (let row = 0; row < entry.capacity; row++) {
            if (bytes) readIds(entry, row, bytes);
            const batch = assign(state, type, entry, row);
            state.rows[offset + row] = batch;
            if (
                moved &&
                (type >= previous[0]! ||
                    row >= previous[type + 2]! - previous[type + 1]! ||
                    previous[previous[type + 1]! + row] !== batch)
            )
                moved(world, type, row);
        }
        offset += entry.capacity;
    }
    state.rows[typeCount + 1] = offset;
}

/** Type offsets precede row-to-batch entries; notifications retain every changed row until sync. @internal */
export function syncMaterialBatches(world: World, moved?: MovedRow): boolean {
    const state = world.resource(MaterialBatches);
    const { types, sources } = world.resource(MaterialTypes);
    let reset = state.entries.length !== types.length;
    let length = types.length + 2;
    for (let type = 0; type < types.length; type++) {
        const table = sources[type]?.table;
        const capacity = Math.max(1, table?.capacity ?? 0);
        let entry = state.entries[type];
        if (!entry || entry.layout !== types[type]?.layout || entry.table !== table) {
            entry?.stop();
            state.entries[type] = entry = entryFor(world, types[type], table);
            reset = true;
        } else if (entry.capacity !== capacity) {
            Object.assign(entry, rowStorage(capacity, entry.fields.length));
            reset = true;
        }
        length += entry.capacity;
    }
    let changed = reset;
    if (reset) rebuild(world, state, types.length, length, moved);
    else
        for (let type = 0; type < types.length; type++) {
            const entry = state.entries[type]!;
            if (!entry.pendingCount) continue;
            const bytes = entry.table!.bytes;
            const offset = state.rows[type + 1]!;
            for (let i = 0; i < entry.pendingCount; i++) {
                const row = entry.pending[i];
                entry.dirty[row] = 0;
                if (!readIds(entry, row, bytes)) continue;
                release(state, entry, row, state.rows[offset + row]!);
                state.rows[offset + row] = assign(state, type, entry, row);
                moved?.(world, type, row);
                changed = true;
            }
            entry.pendingCount = 0;
        }
    if (!changed) return false;
    if (!state.buffer || length > state.bufferCapacity) {
        state.buffer?.destroy();
        state.buffer = createBuffer(world, state.rows);
        state.bufferCapacity = length;
    } else world.gpu.device.queue.writeBuffer(world.gpu.root.unwrap(state.buffer), 0, state.rows);
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
