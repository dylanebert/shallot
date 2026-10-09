import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { GpuTable, Resource, System, World } from "../../engine/ecs";
import { Xform } from "../../engine/utils";
import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    PrepareGlobalTransformSystem,
    TransformRuntime,
} from "../transform";

const terms = [GlobalTransform];
const layout = tgpu.bindGroupLayout({
    current: { storage: d.arrayOf(Xform), access: "readonly" },
    previous: { storage: d.arrayOf(Xform), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    rows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    params: { uniform: d.vec4f },
});
const kernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((args) => {
        "use gpu";
        const i = args.gid.x;
        if (i >= d.u32(layout.$.params.y)) return;
        const row = layout.$.rows[i].y;
        const current = layout.$.current[row];
        const previous = layout.$.previous[row];
        const alpha = layout.$.params.x;
        const flip = std.select(d.f32(1), d.f32(-1), std.dot(previous.quat, current.quat) < 0);
        const blend = std.add(
            std.mul(previous.quat, flip * (1 - alpha)),
            std.mul(current.quat, alpha),
        );
        const len = std.length(blend);
        let quat = d.vec4f(0, 0, 0, 1);
        if (len > 1e-12) quat = std.div(blend, len);
        layout.$.output[row] = Xform({
            pos: std.mix(previous.pos, current.pos, alpha),
            quat,
            scale: std.mix(previous.scale, current.scale, alpha),
        });
    })
    .$name("globalTransformInterpolate");
/** @internal Rendering-owned GPU placement history. */
export interface GlobalTransformRuntime {
    current?: GpuTable<typeof Xform>;
    previous?: GpuTable<typeof Xform>;
    render?: GpuTable<typeof Xform>;
    enabled: boolean;
    tickCount: number;
    stages: (GPUBuffer | undefined)[];
    ranges: Uint32Array;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
    generation: number;
    params?: GPUBuffer;
    discontinuities: Uint32Array;
    discontinuityCount: number;
    historyNeedsPromotion: boolean;
}
/** @internal GPU history identity survives compatible renderer reloads. */
export const GlobalTransformHistory: Resource<GlobalTransformRuntime> = {
    key: Symbol.for("@dylanebert/shallot/rendering/global-transform-history"),
    create(world) {
        const runtime: GlobalTransformRuntime = {
            enabled: false,
            tickCount: 0,
            stages: new Array(2),
            ranges: new Uint32Array(4),
            generation: -1,
            discontinuities: new Uint32Array(1),
            discontinuityCount: 0,
            historyNeedsPromotion: false,
        };
        world.observeMembership(GlobalTransform, (eid, present) => {
            if (present && runtime.enabled) queueDiscontinuity(runtime, eid);
        });
        return runtime;
    },
};
/** Captures initial derived placement before every ordinary fixed system. */
export const GlobalTransformHistoryStartSystem: System = {
    group: "fixed",
    name: "global-transform-history-start",
    boundary: "before",
    after: [GlobalTransformTickStartSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        if (!runtime.enabled) return;
        if (runtime.tickCount && runtime.ranges[3])
            runtime.current!.markRange(
                runtime.ranges[2] / runtime.current!.rowBytes,
                runtime.ranges[3] / runtime.current!.rowBytes,
            );
        captureCurrent(world, 0);
        runtime.discontinuityCount = 0;
        world.resource(TransformRuntime).discontinuityCount = 0;
    },
};
/** Captures completed derived placement after every ordinary fixed writer. */
export const GlobalTransformHistoryEndSystem: System = {
    group: "fixed",
    name: "global-transform-history-end",
    boundary: "after",
    after: [GlobalTransformTickEndSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        if (runtime.enabled) {
            runtime.tickCount = 1;
            captureCurrent(world, 1);
        }
    },
};
/** Stages simulation placement before ordinary draw systems use the World's encoder. */
export const PrepareGlobalTransformHistorySystem: System = {
    group: "draw",
    name: "prepare-global-transform-history",
    boundary: "before",
    after: [PrepareGlobalTransformSystem],
    update(world) {
        const runtime = world.resource(GlobalTransformHistory);
        if (!runtime.enabled) return;
        captureCurrent(world, 1);
        runtime.render!.upload();
    },
};
/** @internal Discards presentation history after restoration of fixed placement. */
export function recoverGlobalTransformHistory(world: World) {
    const runtime = world.resource(GlobalTransformHistory);
    return {
        snapshot: () => undefined,
        restore() {
            runtime.tickCount = 0;
            runtime.ranges.fill(0);
            runtime.historyNeedsPromotion = false;
            runtime.discontinuityCount = 0;
            if (runtime.enabled)
                for (const eid of world.query(terms)) queueDiscontinuity(runtime, eid);
        },
    };
}
/** Rendering-owned interpolated dense placement rows. Requires RenderingPlugin;
 * history and interpolation record through `world.frameEncoder()` before draw passes. */
export function globalTransformTable(world: World): GpuTable<typeof Xform> {
    const runtime = world.resource(GlobalTransformHistory);
    if (!runtime.enabled && !world.hasSystem(GlobalTransformHistoryEndSystem))
        throw new Error("globalTransformTable requires RenderingPlugin");
    if (!runtime.enabled) {
        runtime.enabled = true;
        const current = (runtime.current = world.table("global-transform", Xform));
        const previous = (runtime.previous = world.table("global-transform-previous-tick", Xform, {
            gpuOnly: true,
        }));
        const render = (runtime.render = world.table("global-transform-interpolated", Xform, {
            gpuOnly: true,
        }));
        current.bindComponent(GlobalTransform, {
            pos: "translation",
            quat: "rotation",
            scale: "scale",
        });
        previous.bindMembership(GlobalTransform);
        render.bindMembership(GlobalTransform);
        render.subscribe((buffer) => {
            world.gpu.buffers.set("globalTransforms", buffer);
            world.gpu.typed.set("globalTransforms", render.typed);
        });
        render.enableEidLookup();
        render.subscribeMap((buffer) => {
            world.gpu.buffers.set("globalTransformRows", buffer);
            world.gpu.typed.set("globalTransformRows", render.eidToRowTyped!);
        });
        if (!runtime.params) {
            const params = world.gpu.root.createBuffer(d.vec4f).$usage("uniform");
            runtime.params = world.gpu.root.unwrap(params);
            world.own(runtime.params);
        }
        runtime.pipeline = world.gpu.root.unwrap(
            world.gpu.root.createComputePipeline({ compute: kernel }),
        );
        for (const eid of world.query(terms)) queueDiscontinuity(runtime, eid);
    }
    return runtime.render!;
}
function queueDiscontinuity(runtime: GlobalTransformRuntime, eid: number): void {
    for (let i = 0; i < runtime.discontinuityCount; i++)
        if (runtime.discontinuities[i] === eid) return;
    if (runtime.discontinuityCount === runtime.discontinuities.length) {
        const next = new Uint32Array(runtime.discontinuities.length * 2);
        next.set(runtime.discontinuities);
        runtime.discontinuities = next;
    }
    runtime.discontinuities[runtime.discontinuityCount++] = eid;
}
function captureCurrent(world: World, phase: number): void {
    const runtime = world.resource(GlobalTransformHistory);
    const placement = world.resource(TransformRuntime);
    for (let i = 0; i < placement.discontinuityCount; i++)
        queueDiscontinuity(runtime, placement.discontinuities[i]);
    const current = runtime.current!;
    current.prepareUpload();
    // Reused stages retain earlier changed rows, including gaps in their range.
    if (current.pendingUploadSize && runtime.ranges[phase * 2 + 1])
        current.markRange(
            runtime.ranges[phase * 2] / current.rowBytes,
            runtime.ranges[phase * 2 + 1] / current.rowBytes,
        );
    const size = current.pendingUploadSize;
    let buffer = runtime.stages[phase];
    if (size && (!buffer || buffer.size < size)) {
        if (buffer) world.retireGpuBuffer(buffer);
        buffer = world.gpu.device.createBuffer({
            label: `global-transform-tick-${phase}`,
            size,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        world.own(buffer);
        runtime.stages[phase] = buffer;
    }
    if (size && buffer) {
        current.upload(buffer, true);
        runtime.ranges[phase * 2] = current.lastUploadOffset;
        runtime.ranges[phase * 2 + 1] = size;
    }
}
function copyPhase(world: World, encoder: GPUCommandEncoder | undefined, phase: number): void {
    const runtime = world.resource(GlobalTransformHistory);
    const offset = runtime.ranges[phase * 2],
        size = runtime.ranges[phase * 2 + 1];
    if (size && !runtime.stages[phase])
        throw new Error(
            `GlobalTransform history phase ${phase} has ${size} bytes but no staging buffer`,
        );
    if (size) {
        (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
            runtime.stages[phase]!,
            0,
            runtime.current!.buffer,
            offset,
            size,
        );
        runtime.historyNeedsPromotion = true;
    }
    if (phase === 1)
        for (let i = 0; i < runtime.discontinuityCount; i++) {
            const current = runtime.current!,
                row = current.rowIndex(runtime.discontinuities[i]);
            if (row >= 0)
                (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
                    current.buffer,
                    row * current.rowBytes,
                    runtime.previous!.buffer,
                    row * current.rowBytes,
                    current.rowBytes,
                );
        }
}
/** @internal Records pending history work through the World's draw encoder. */
export function prepareGlobalTransformFrame(world: World, encoder?: GPUCommandEncoder): void {
    const runtime = world.resource(GlobalTransformHistory);
    if (!runtime.enabled) return;
    if (runtime.tickCount) {
        copyPhase(world, encoder, 0);
        if (runtime.historyNeedsPromotion) {
            (encoder ?? world.frameEncoder()!).copyBufferToBuffer(
                runtime.current!.buffer,
                0,
                runtime.previous!.buffer,
                0,
                runtime.current!.buffer.size,
            );
            runtime.historyNeedsPromotion = false;
        }
    }
    copyPhase(world, encoder, 1);
    runtime.tickCount = 0;
    runtime.discontinuityCount = 0;
    runtime.ranges.fill(0);
    world.resource(TransformRuntime).discontinuityCount = 0;
    if (!runtime.current!.count) {
        runtime.historyNeedsPromotion = false;
        return;
    }
    const generation =
        runtime.current!.generation +
        runtime.previous!.generation +
        runtime.render!.generation +
        runtime.current!.activeGeneration;
    if (generation !== runtime.generation) {
        runtime.group = world.gpu.device.createBindGroup({
            layout: runtime.pipeline!.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: runtime.current!.buffer } },
                { binding: 1, resource: { buffer: runtime.previous!.buffer } },
                { binding: 2, resource: { buffer: runtime.render!.buffer } },
                { binding: 3, resource: { buffer: runtime.current!.activeRowsBuffer! } },
                { binding: 4, resource: { buffer: runtime.params! } },
            ],
        });
        runtime.generation = generation;
    }
}
