import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { component, type GpuTable, type Resource, vec3, vec4, type World } from "../../engine/ecs";
import { dispatchWorkgroups, setComputeBindGroup, setComputePipeline } from "../../engine/runtime";
import { Xform } from "./xform";

/** World-space visual error applied after fixed-sample interpolation: translation is added
 * in world units and unit xyzw rotation pre-multiplies the interpolated rotation.
 * Rendering only reads it; its producer writes and decays it in a first draw system
 * ordered before PrepareViewsSystem.
 * Neither fixed placement nor scale changes. Remove it when correction ends. */
export const VisualCorrection = component(
    "VisualCorrection",
    {
        translation: vec3,
        rotation: vec4,
    },
    {
        presentation: true,
        defaults: () => ({ translation: [0, 0, 0], rotation: [0, 0, 0, 1] }),
    },
);

const correctionTerms = [VisualCorrection];
const Correction = d.struct({ translation: d.vec3f, rotation: d.vec4f });
const layout = tgpu.bindGroupLayout({
    corrections: { storage: d.arrayOf(Correction), access: "readonly" },
    active: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    rows: { storage: d.arrayOf(d.u32), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    count: { uniform: d.u32 },
});
const kernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((args) => {
        "use gpu";
        const i = args.gid.x;
        if (i >= layout.$.count) return;
        const active = layout.$.active[i];
        if (active.x >= std.arrayLength(layout.$.rows)) return;
        const mapped = layout.$.rows[active.x];
        if (mapped === 0) return;
        const row = mapped - 1;
        const correction = layout.$.corrections[active.y];
        const pose = layout.$.output[row];
        const a = correction.rotation;
        const b = pose.quat;
        layout.$.output[row] = Xform({
            pos: std.add(pose.pos, correction.translation),
            quat: d.vec4f(
                std.add(std.add(std.mul(b.xyz, a.w), std.mul(a.xyz, b.w)), std.cross(a.xyz, b.xyz)),
                a.w * b.w - std.dot(a.xyz, b.xyz),
            ),
            scale: pose.scale,
        });
    })
    .$name("visualCorrection");

type Runtime = {
    table: GpuTable<typeof Correction>;
    bound: boolean;
    pipeline: GPUComputePipeline;
    count: GPUBuffer;
    words: Uint32Array;
    group?: GPUBindGroup;
    generation: number;
};
/** @internal Rendering's sparse visual-correction pass state. */
export const VisualCorrectionState: Resource<Runtime> = {
    create(world) {
        const table = world.table("visual-correction", Correction);
        const count = world.gpu.root.createBuffer(d.u32).$usage("uniform");
        const buffer = world.gpu.root.unwrap(count);
        world.own(buffer);
        return {
            table,
            bound: false,
            count: buffer,
            words: new Uint32Array(1),
            generation: -1,
            pipeline: world.gpu.root.unwrap(
                world.gpu.root.createComputePipeline({ compute: kernel }),
            ),
        };
    },
};
/** @internal Prepare the optional correction pass during rendering initialization. */
export function prepareVisualCorrection(world: World): void {
    world.resource(VisualCorrectionState);
}
/** @internal Uploads correction writes before opening a compute pass. */
export function uploadVisualCorrections(world: World, output: GpuTable<typeof Xform>): void {
    const runtime = world.resource(VisualCorrectionState);
    const table = runtime.table;
    if (!runtime.bound) {
        for (const _eid of world.query(correctionTerms)) {
            table.bindComponent(VisualCorrection, {
                translation: "translation",
                rotation: "rotation",
            });
            runtime.bound = true;
            break;
        }
        if (!runtime.bound) return;
    }
    table.upload();
    if (!table.count) return;
    runtime.words[0] = table.count;
    world.uploadGpuTable(runtime.count, 0, runtime.words.buffer, 4);
    const generation =
        table.generation + table.activeGeneration + output.generation + output.mapGeneration;
    if (runtime.generation === generation) return;
    runtime.generation = generation;
    runtime.group = world.gpu.device.createBindGroup({
        layout: runtime.pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: table.buffer } },
            { binding: 1, resource: { buffer: table.activeRowsBuffer! } },
            { binding: 2, resource: { buffer: output.eidToRowBuffer! } },
            { binding: 3, resource: { buffer: output.buffer } },
            { binding: 4, resource: { buffer: runtime.count } },
        ],
    });
}
/** @internal Applies only correction holders, after the interpolation dispatch. */
export function applyVisualCorrections(world: World, pass: GPUComputePassEncoder): void {
    const runtime = world.resource(VisualCorrectionState);
    if (!runtime.table.count) return;
    setComputePipeline(pass, runtime.pipeline);
    setComputeBindGroup(pass, 0, runtime.group!);
    dispatchWorkgroups(pass, Math.ceil(runtime.table.count / 64));
}
/** @internal Apply the same world-space correction to a caller-owned Xform record. */
export function correctRenderTransform(world: World, eid: number, out: Float32Array): void {
    if (!world.has(eid, VisualCorrection)) return;
    const storage = world.storage(VisualCorrection);
    const p = storage.translation.column,
        q = storage.rotation.column,
        j = eid * 4;
    for (let i = 0; i < 3; i++) out[i] = out[i]! + p[j + i]!;
    const ax = q[j]!,
        ay = q[j + 1]!,
        az = q[j + 2]!,
        aw = q[j + 3]!;
    const bx = out[4]!,
        by = out[5]!,
        bz = out[6]!,
        bw = out[7]!;
    out[4] = aw * bx + ax * bw + ay * bz - az * by;
    out[5] = aw * by - ax * bz + ay * bw + az * bx;
    out[6] = aw * bz + ax * by - ay * bx + az * bw;
    out[7] = aw * bw - ax * bx - ay * by - az * bz;
}
