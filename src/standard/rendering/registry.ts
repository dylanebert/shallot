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
 * Other material bindings resolve by name in world.gpu's typed-buffer, texture and sampler registries.
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
    override register(spec: Draw): number {
        validateDrawIndirect(spec.args.indirect);
        return super.register(spec);
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
