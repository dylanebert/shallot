import type { TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import { Registry } from "../../engine";

export { Surfaces } from "./contract";

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

// the render-contract registries a producer registers into and a consumer
// renderer iterates: Surfaces (shading recipes) and Draws (draw calls). Meshes
// (geometry) lives in mesh.ts alongside its packing logic. Render defines these
// but never iterates them — they're the bridge between any producer and any
// renderer.

/**
 * draw arguments: always indirect. `indirect` is a GPU buffer holding the
 * standard 20-byte `DrawIndexedIndirect` record (`{ indexCount, instanceCount,
 * firstIndex, baseVertex, firstInstance }`) at `offset`; the renderer binds the
 * mesh's index buffer and issues `pass.drawIndexedIndirect(indirect, offset)`. Everything is GPU-driven, so the draw
 * count lives in GPU memory: a CPU-known draw just `writeBuffer`s its record
 * into a small buffer rather than passing literals.
 *
 * `viewStride`, when set, makes the record per-view: a producer that culls into
 * a separate `DrawIndexedIndirect` per camera lays the records out `slot`-major, and
 * the renderer reads `offset + view.slot * viewStride` so each camera draws its
 * own culled instances. Omit it (or `0`) for a view-independent draw
 */
export interface DrawArgs {
    indirect: DrawIndirectBuffer;
    offset?: number;
    viewStride?: number;
}

/**
 * one rendered thing. `surface` references a registered Surface by name;
 * `mesh` references a registered Mesh by name: the consumer renderer pulls
 * indexed vertices from that mesh's `vertices` + `indices` buffers in WGSL.
 * `args` points at the indirect draw record. Surface bindings beyond mesh
 * resolve by name against `Compute.buffers`
 */
export interface Draw {
    name: string;
    surface: string;
    mesh: string;
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

/** every registered draw, keyed by name */
export const Draws: Registry<Draw> = new DrawRegistry();
