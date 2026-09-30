import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { eulerAlias, Xform } from "../utils";
import { vec4 } from "./component";
import { field } from "./field";
import type { State } from "./state";
import type { ComponentStorage } from "./storage";
import type { GpuTable } from "./table";

/** Authored world placement. A simulated body excludes this producer. There is no hierarchy. */
export const Transform = { pos: field(vec4), rot: field(vec4), scale: field(vec4) };
/** Derived fixed-tick world placement. Gameplay and physics queries read this, never Transform. */
export const Pose = { pos: field(vec4), quat: field(vec4), scale: field(vec4), vel: field(vec4) };
export const poseTraits = {
    derived: true,
    defaults: () => ({
        pos: [0, 0, 0, 0],
        quat: [0, 0, 0, 1],
        scale: [1, 1, 1, 0],
        vel: [0, 0, 0, 0],
    }),
};
const transformTerms = [Transform];
const layout = tgpu.bindGroupLayout({
    current: { storage: d.arrayOf(Xform), access: "readonly" },
    previous: { storage: d.arrayOf(Xform), access: "readonly" },
    output: { storage: d.arrayOf(Xform), access: "mutable" },
    rows: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    params: { uniform: d.vec4f },
});
const kernel = tgpu.computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })(
    (args) => {
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
    },
);

/** @internal All pose residency, including history, belongs to this world. */
export interface PoseRuntime {
    current: GpuTable<typeof Xform>;
    previous: GpuTable<typeof Xform>;
    render: GpuTable<typeof Xform>;
    fresh: Uint32Array;
    freshCount: number;
    pipeline: GPUComputePipeline;
    group: GPUBindGroup | undefined;
    generation: number;
    params: GPUBuffer;
    words: Float32Array;
    placement: ComponentStorage<typeof Transform>;
    pose: ComponentStorage<typeof Pose>;
    commands: GPUCommandBuffer[];
    encoderDescriptor: GPUCommandEncoderDescriptor;
    passDescriptor: GPUComputePassDescriptor;
}

/** @internal Register the built-in schemas; plugins cannot opt out of world pose. */
export function registerPose(state: State): void {
    state.registry.register("Pose", Pose, poseTraits);
    state.registry.register("Transform", Transform, {
        defaults: () => ({ pos: [0, 0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1, 1] }),
        aliases: { rot: eulerAlias("rot") },
        provides: [Pose],
    });
}
/** @internal Install once after columns have been allocated and before scene/setup authoring. */
export function initializePose(state: State): void {
    const current = state.table("pose", Xform);
    const previous = state.table("pose-previous", Xform, { gpuOnly: true });
    const render = state.table("transforms", Xform, { gpuOnly: true });
    current.bindComponent(Pose, { pos: "pos", quat: "quat", scale: "scale" });
    previous.bindMembership(Pose);
    render.bindMembership(Pose);
    render.enableEidLookup();
    render.subscribeMap((buffer) => {
        state.gpu.buffers.set("transformRows", buffer);
        state.gpu.typed.set("transformRows", render.eidToRowTyped!);
    });
    state.gpu.buffers.set("transformRows", render.eidToRowBuffer!);
    state.gpu.typed.set("transformRows", render.eidToRowTyped!);
    const params = state.gpu.root.createBuffer(d.vec4f).$usage("uniform");
    const rawParams = state.gpu.root.unwrap(params);
    state.own(rawParams);
    const runtime: PoseRuntime = {
        current,
        previous,
        render,
        fresh: new Uint32Array(current.capacity),
        freshCount: 0,
        pipeline: state.gpu.root.unwrap(state.gpu.root.createComputePipeline({ compute: kernel })),
        group: undefined,
        generation: -1,
        params: rawParams,
        words: new Float32Array(4),
        placement: state.of(Transform),
        pose: state.of(Pose),
        commands: new Array(1),
        encoderDescriptor: {},
        passDescriptor: {},
    };
    state.poseRuntime = runtime;
    state.observeMembership(Pose, (eid, present) => {
        if (!present) return;
        if (runtime.freshCount === runtime.fresh.length) {
            const fresh = new Uint32Array(runtime.fresh.length * 2);
            fresh.set(runtime.fresh);
            runtime.fresh = fresh;
        }
        runtime.fresh[runtime.freshCount++] = eid;
    });
    state.observeMembership(Transform, (eid, present) => {
        if (present) state.add(eid, Pose);
        else if (state.has(eid, Pose)) state.remove(eid, Pose);
    });
}

/** The engine's interpolated dense renderer rows. Current/previous inputs are not authored. */
export function transformTable(state: State): GpuTable<typeof Xform> {
    if (!state.poseRuntime)
        throw new Error("world pose is unavailable before engine initialization");
    return state.poseRuntime.render;
}

/** @internal Gather authored placement into the fixed world column without per-row callbacks. */
export function deriveTransforms(state: State): void {
    if (!state.poseRuntime) return;
    const pose = state.poseRuntime.pose;
    const sources = state.poseRuntime.placement;
    // Resolve growing column references once per gather, not per entity.
    const pp = sources.pos.column,
        pq = sources.rot.column,
        ps = sources.scale.column;
    const op = pose.pos.column,
        oq = pose.quat.column,
        os = pose.scale.column;
    const pd = pose.pos.dirty,
        qd = pose.quat.dirty,
        sd = pose.scale.dirty;
    for (const eid of state.query(transformTerms)) {
        const word = eid >>> 5,
            mask = 1 << (eid & 31);
        if (
            ((sources.pos.dirty[word] | sources.rot.dirty[word] | sources.scale.dirty[word]) &
                mask) ===
            0
        )
            continue;
        const offset = eid * 4;
        let posChanged = false,
            quatChanged = false,
            scaleChanged = false;
        for (let lane = 0; lane < 4; lane++) {
            const j = offset + lane;
            if (!Object.is(op[j], pp[j])) {
                op[j] = pp[j];
                posChanged = true;
            }
            if (!Object.is(oq[j], pq[j])) {
                oq[j] = pq[j];
                quatChanged = true;
            }
            if (!Object.is(os[j], ps[j])) {
                os[j] = ps[j];
                scaleChanged = true;
            }
        }
        if (posChanged) pd[word] |= mask;
        if (quatChanged) qd[word] |= mask;
        if (scaleChanged) sd[word] |= mask;
    }
}

function uploadCurrent(state: State): void {
    const runtime = state.poseRuntime!;
    runtime.current.upload();
    if (!runtime.freshCount) return;
    const encoder = state.gpu.device.createCommandEncoder(runtime.encoderDescriptor);
    for (let i = 0; i < runtime.freshCount; i++) {
        const row = runtime.current.rowIndex(runtime.fresh[i]);
        if (row < 0) continue;
        encoder.copyBufferToBuffer(
            runtime.current.buffer,
            row * 48,
            runtime.previous.buffer,
            row * 48,
            48,
        );
    }
    runtime.freshCount = 0;
    runtime.commands[0] = encoder.finish();
    state.gpu.device.queue.submit(runtime.commands);
}
/** @internal The snapshot runs before every individual fixed tick, including catch-up ticks. */
export function beginPoseTick(state: State): void {
    const runtime = state.poseRuntime;
    if (!runtime) return;
    deriveTransforms(state);
    uploadCurrent(state);
    const encoder = state.gpu.device.createCommandEncoder(runtime.encoderDescriptor);
    encoder.copyBufferToBuffer(
        runtime.current.buffer,
        0,
        runtime.previous.buffer,
        0,
        runtime.current.buffer.size,
    );
    runtime.commands[0] = encoder.finish();
    state.gpu.device.queue.submit(runtime.commands);
}
/** @internal Only changed current rows upload after producers finish the tick. */
export function endPoseTick(state: State): void {
    if (!state.poseRuntime) return;
    deriveTransforms(state);
    uploadCurrent(state);
}
/** @internal Derive the renderer rows once per frame, before any draw reader. */
export function presentPose(state: State): void {
    const runtime = state.poseRuntime;
    if (!runtime) return;
    deriveTransforms(state);
    uploadCurrent(state);
    runtime.render.upload();
    const generation =
        runtime.current.generation +
        runtime.previous.generation +
        runtime.render.generation +
        runtime.current.activeGeneration;
    if (generation !== runtime.generation) {
        const device = state.gpu.device;
        runtime.group = device.createBindGroup({
            layout: runtime.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: runtime.current.buffer } },
                { binding: 1, resource: { buffer: runtime.previous.buffer } },
                { binding: 2, resource: { buffer: runtime.render.buffer } },
                { binding: 3, resource: { buffer: runtime.current.activeRowsBuffer! } },
                { binding: 4, resource: { buffer: runtime.params } },
            ],
        });
        runtime.generation = generation;
    }
    runtime.words[0] = state.time.fixedAlpha;
    runtime.words[1] = runtime.current.count;
    state.gpu.device.queue.writeBuffer(runtime.params, 0, runtime.words);
    if (!runtime.current.count) return;
    const encoder = state.gpu.device.createCommandEncoder(runtime.encoderDescriptor);
    const pass = encoder.beginComputePass(runtime.passDescriptor);
    pass.setPipeline(runtime.pipeline);
    pass.setBindGroup(0, runtime.group!);
    pass.dispatchWorkgroups(Math.ceil(runtime.current.count / 64));
    pass.end();
    runtime.commands[0] = encoder.finish();
    state.gpu.device.queue.submit(runtime.commands);
}

/** Compose fixed-tick world placement for CPU camera and query readers. */
export function composeTransform(eid: number, out: Float32Array): Float32Array {
    const { pos, quat: rot, scale } = Pose;
    const px = pos.x.get(eid),
        py = pos.y.get(eid),
        pz = pos.z.get(eid);
    const qx = rot.x.get(eid),
        qy = rot.y.get(eid),
        qz = rot.z.get(eid),
        qw = rot.w.get(eid);
    const sx = scale.x.get(eid),
        sy = scale.y.get(eid),
        sz = scale.z.get(eid);
    const x2 = qx + qx,
        y2 = qy + qy,
        z2 = qz + qz;
    const xx = qx * x2,
        xy = qx * y2,
        xz = qx * z2,
        yy = qy * y2,
        yz = qy * z2,
        zz = qz * z2;
    const wx = qw * x2,
        wy = qw * y2,
        wz = qw * z2;
    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}
