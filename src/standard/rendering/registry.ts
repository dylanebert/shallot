import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import type { MeshHandle } from "../../core/mesh";
import { Registry, type World } from "../../engine";

/** WebGPU's canonical 20-byte `DrawIndexedIndirect` record. `baseVertex` is signed by specification. */
export const DrawIndexedIndirect = d
    .struct({
        indexCount: d.u32,
        instanceCount: d.u32,
        firstIndex: d.u32,
        baseVertex: d.i32,
        firstInstance: d.u32,
    })
    .$name("DrawIndexedIndirect");

/** one indirect record or a record array (offset/view-stride producers), carrying INDIRECT usage. */
export type DrawIndirectBuffer = TgpuBuffer<
    typeof DrawIndexedIndirect | d.WgslArray<typeof DrawIndexedIndirect>
> & { usableAsIndirect: true };

// Producers register shading recipes and indirect draw records for standard's forward pipeline.
// Geometry belongs to core/mesh; core rendering knows none of these registries.

/**
 * GPU DrawIndexedIndirect record (20 bytes: indexCount, instanceCount, firstIndex, baseVertex,
 * firstInstance) at byte offset (default zero). Even a CPU-known draw supplies a GPU record, not
 * literal counts. Nonzero viewStride, in bytes, selects offset + view.slot*viewStride;
 * omitted or zero is view-independent.
 */
export interface DrawArgs {
    indirect: DrawIndirectBuffer;
    offset?: number;
    viewStride?: number;
}

/**
 * materialType is a registered material-type id, alphaPipelineKey selects its phase/variant,
 * mesh is a registered mesh handle, and args addresses its indirect record.
 * Owned material bindings resolve from mesh attributes, mesh overrides or the material type's bindings.
 */
export interface Draw {
    name: string;
    materialType: number;
    alphaPipelineKey: number;
    mesh: MeshHandle;
    args: DrawArgs;
}

/** loud runtime twin of {@link DrawIndirectBuffer}'s type contract. */
export function validateDrawIndirect(buffer: DrawIndirectBuffer): void {
    const schema = buffer.dataType;
    const compatible =
        d.deepEqual(schema, DrawIndexedIndirect) ||
        (d.isWgslArray(schema) && d.deepEqual(schema.elementType, DrawIndexedIndirect));
    if (!compatible)
        throw new Error("draw indirect buffer has the wrong DrawIndexedIndirect schema");
    if (!buffer.usableAsIndirect) throw new Error("draw indirect buffer is missing indirect usage");
}

class DrawRegistry extends Registry<Draw> {
    private _active: boolean[] = [];
    private _next: number[] = [];
    private _previous: number[] = [];
    private _head = -1;
    private _tail = -1;
    private _count = 0;

    override register(spec: Draw): number {
        validateDrawIndirect(spec.args.indirect);
        // Reserved names retain their Map slots, including while the draw is absent.
        const id = super.register(spec);
        if (!this._active[id]) {
            this._active[id] = true;
            this._previous[id] = this._tail;
            this._next[id] = -1;
            if (this._tail < 0) this._head = id;
            else this._next[this._tail] = id;
            this._tail = id;
            this._count++;
        }
        return id;
    }

    override delete(name: string): boolean {
        const id = this.id(name);
        if (id === undefined || !this._active[id]) return false;
        const previous = this._previous[id]!;
        const next = this._next[id]!;
        if (previous < 0) this._head = next;
        else this._next[previous] = next;
        if (next < 0) this._tail = previous;
        else this._previous[next] = previous;
        this._active[id] = false;
        this._count--;
        return true;
    }

    override get(name: string): Draw | undefined {
        return this.has(name) ? super.get(name) : undefined;
    }

    override has(name: string): boolean {
        const id = this.id(name);
        return id !== undefined && this._active[id] === true;
    }

    override get size(): number {
        return this._count;
    }

    override forEach(callback: (value: Draw) => void): void {
        for (let id = this._head; id >= 0; id = this._next[id]!)
            callback(super.get(this.name(id)!)!);
    }

    override *values(): IterableIterator<Draw> {
        for (let id = this._head; id >= 0; id = this._next[id]!) yield super.get(this.name(id)!)!;
    }

    override snapshot(): ReturnType<Registry<Draw>["snapshot"]> {
        const names: string[] = [];
        for (let id = 0; this.name(id) !== undefined; id++) names.push(this.name(id)!);
        return { names, values: new Map([...this.values()].map((draw) => [draw.name, draw])) };
    }

    override restore(state: ReturnType<Registry<Draw>["snapshot"]>): void {
        this.clear();
        super.restore({ names: state.names, values: new Map() });
        for (const draw of state.values.values()) this.register(draw);
    }

    override clear(): void {
        super.clear();
        this._active.length = this._next.length = this._previous.length = 0;
        this._head = this._tail = -1;
        this._count = 0;
    }
}

export const drawsKey = { create: () => createDraws() };
const createDraws = () => new DrawRegistry();

/** every registered draw in the active world's registry, keyed by name */
export const Draws: import("../../engine").Resource<Registry<Draw>> = {
    create: (world) => world.resource(drawsKey),
};

export function initializeDrawState(world: World): void {
    world.resource(drawsKey);
}
