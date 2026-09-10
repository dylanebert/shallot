import tgpu, { type TgpuBindGroupLayout, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import { Compute, type Type } from "../../engine";

// The scatter kernel: the compute half of the slab flush, one pipeline per GPU element type. Its
// element schemas are also where a slab's mirror layout is declared, so the buffer sizes and the stager
// arithmetic derive from the same source the shader binds.

// A type absent here has no native WGSL storage primitive (`u8`, `u16`): `slab(...)` warns and skips GPU
// allocation.
const ELEMENTS: Record<string, d.AnyWgslData> = {
    f32: d.f32,
    i32: d.i32,
    u32: d.u32,
    // `shader-f16` is not on the platform floor, so this element is the consumer opt-in path: a
    // `slab(f16)` only compiles for an app whose plugin declares `shader-f16` in `Plugin.features`. No
    // engine slab takes it — `f16x4` mirrors as `vec2<u32>`.
    f16: d.f16,
    "vec2<f32>": d.vec2f,
    "vec4<f32>": d.vec4f,
    "vec2<u32>": d.vec2u,
};

/** the kernel + pipeline key: the GPU element type, not the slab's name — a packed `srgb8x4` color
 *  mirrors as a `u32`, so it shares the one `u32` scatter pipeline with every other u32 slab, the copy
 *  being identical regardless of what the bits decode to (`f16x4` keys on its own `vec2<u32>`).
 *  @internal */
export function scatterKey(type: Type): string {
    return type.gpu?.wgsl ?? type.wgsl ?? type.name;
}

/** the GPU element schema a slab of this type mirrors through, or null for a type with no native WGSL
 *  storage primitive.
 *  @internal */
export function elementOf(type: Type): d.AnyWgslData | null {
    return ELEMENTS[scatterKey(type)] ?? null;
}

/** bytes per GPU element, read off the schema — the value buffer's size and the stager's value-region
 *  offset both derive from it.
 *  @internal */
export function elementBytes(type: Type): number | null {
    const element = elementOf(type);
    return element ? d.sizeOf(element) : null;
}

/** @internal */
export type ScatterLayout = TgpuBindGroupLayout<{
    slots: { storage: (n: number) => d.WgslArray<d.U32>; access: "readonly" };
    values: { storage: (n: number) => d.WgslArray<d.AnyWgslData>; access: "readonly" };
    canonical: { storage: (n: number) => d.WgslArray<d.AnyWgslData>; access: "mutable" };
}>;

// Layout and kernel are authored together inside the factory so the body closes over *this* layout: one
// authored kernel, re-emitted per element with the element's own type in each `array<...>`.
function scatterLayout(element: d.AnyWgslData): ScatterLayout {
    return tgpu.bindGroupLayout({
        slots: { storage: d.arrayOf(d.u32), access: "readonly" },
        values: { storage: d.arrayOf(element), access: "readonly" },
        canonical: { storage: d.arrayOf(element), access: "mutable" },
    }) as ScatterLayout;
}

function scatterKernel(element: d.AnyWgslData, layout: ScatterLayout) {
    // TGSL refuses to assign one storage reference to another ("references cannot be assigned"), so the
    // element schema doubles as the copy constructor.
    const copy = element as unknown as (v: unknown) => unknown;
    return tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const i = input.gid.x;
        const count = layout.$.slots[0];
        if (i >= count) return;
        layout.$.canonical[layout.$.slots[i + 1]] = copy(layout.$.values[i]);
    });
}

/** @internal */
export interface Scatter {
    layout: ScatterLayout;
    pipeline: TgpuComputePipeline;
}

const pipelines = new Map<string, Scatter>();

/** the memoized scatter pipeline for a slab type's GPU element, created against the active root.
 *  @internal */
export function scatterPipeline(type: Type): Scatter {
    const key = scatterKey(type);
    const cached = pipelines.get(key);
    if (cached) return cached;
    const element = elementOf(type)!;
    const layout = scatterLayout(element);
    const pipeline = Compute.root
        .createComputePipeline({ compute: scatterKernel(element, layout) })
        .$name(`slab-scatter-${key}`);
    const ctx = { layout, pipeline };
    pipelines.set(key, ctx);
    return ctx;
}

/** the already-compiled pipeline for a type, or undefined — `prepare` reads it to fail loud when the
 *  owning plugin never warmed.
 *  @internal */
export function compiled(type: Type): Scatter | undefined {
    return pipelines.get(scatterKey(type));
}

/** drop every compiled pipeline. Pipelines bind to the root that created them, so a build against a new
 *  device recompiles.
 *  @internal */
export function resetPipelines(): void {
    pipelines.clear();
}

/** the emitted scatter WGSL for a slab type — the device-free structural seam its test resolves. It
 *  carries no `enable` directive: a pipeline emits those from the root device's enabled features (so a
 *  `slab(f16)` gets `enable f16;` exactly on a device that requested `shader-f16`), and this seam has no
 *  device to read.
 *  @internal */
export function scatterWgsl(type: Type): string {
    const element = elementOf(type)!;
    return tgpu.resolve([scatterKernel(element, scatterLayout(element))], { names: "strict" });
}
