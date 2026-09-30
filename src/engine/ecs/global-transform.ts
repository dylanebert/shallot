import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { eulerAlias, Xform } from "../utils";
import { type Component, idOf, vec4 } from "./component";
import { field } from "./field";
import type { State } from "./state";
import type { ComponentStorage } from "./storage";
import type { GpuTable } from "./table";

/** Authored world placement. A simulated body excludes this producer. There is no hierarchy. */
export const Transform = { pos: field(vec4), rot: field(vec4), scale: field(vec4) };
/** Derived fixed-tick world placement. Gameplay and physics queries read this, never Transform. */
export const GlobalTransform = {
    pos: field(vec4),
    quat: field(vec4),
    scale: field(vec4),
    vel: field(vec4),
};
export const globalTransformTraits = {
    derived: true,
    defaults: () => ({
        pos: [0, 0, 0, 0],
        quat: [0, 0, 0, 1],
        scale: [1, 1, 1, 0],
        vel: [0, 0, 0, 0],
    }),
};
const transformTerms = [Transform];
const globalTransformTerms = [GlobalTransform];
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

/** @internal World-owned current placement, GPU-only history and interpolated rows. */
export interface GlobalTransformRuntime {
    current?: GpuTable<typeof Xform>;
    previous?: GpuTable<typeof Xform>;
    render?: GpuTable<typeof Xform>;
    fresh: Uint32Array;
    freshCount: number;
    enabled: boolean;
    tickCount: number;
    captureIndex: number;
    stages: (GPUBuffer | undefined)[];
    ranges: Uint32Array;
    pipeline?: GPUComputePipeline;
    group?: GPUBindGroup;
    generation: number;
    params?: GPUBuffer;
    placement: ComponentStorage<typeof Transform>;
    global: ComponentStorage<typeof GlobalTransform>;
    producers: Map<number, Set<number>>;
    pendingRemoval: Set<number>;
    discontinuities: Uint32Array;
    discontinuityPhases: Uint8Array;
    discontinuityCount: number;
    historyNeedsPromotion: boolean;
}

/** @internal Register the built-in schemas; plugins cannot opt out of world placement. */
export function registerGlobalTransform(state: State): void {
    state.registry.register("GlobalTransform", GlobalTransform, globalTransformTraits);
    state.registry.register("Transform", Transform, {
        defaults: () => ({ pos: [0, 0, 0, 0], rot: [0, 0, 0, 1], scale: [1, 1, 1, 1] }),
        aliases: { rot: eulerAlias("rot") },
        provides: [GlobalTransform],
    });
}
/** @internal Install once before scene/setup authoring. GPU residency waits for a reader. */
export function initializeGlobalTransform(state: State): void {
    const runtime: GlobalTransformRuntime = {
        fresh: new Uint32Array(1),
        freshCount: 0,
        enabled: false,
        tickCount: 0,
        captureIndex: 0,
        // Initial placement, four bounded catch-up ticks, and post-simulation placement.
        stages: new Array(6),
        ranges: new Uint32Array(12),
        group: undefined,
        generation: -1,
        placement: state.of(Transform),
        global: state.of(GlobalTransform),
        producers: new Map(),
        pendingRemoval: new Set(),
        discontinuities: new Uint32Array(1),
        discontinuityPhases: new Uint8Array(1),
        discontinuityCount: 0,
        historyNeedsPromotion: false,
    };
    state.globalTransformRuntime = runtime;
    state.observeMembership(GlobalTransform, (eid, present) => {
        if (present && runtime.enabled) queueFresh(runtime, eid);
    });
}

/** The engine's interpolated dense GlobalTransform rows. Request before stepping a renderer. */
export function globalTransformTable(state: State): GpuTable<typeof Xform> {
    const runtime = state.globalTransformRuntime;
    if (!runtime) throw new Error("GlobalTransform is unavailable before engine initialization");
    if (!runtime.enabled) {
        runtime.enabled = true;
        const current = (runtime.current = state.table("global-transform", Xform));
        const previous = (runtime.previous = state.table("global-transform-previous-tick", Xform, {
            gpuOnly: true,
        }));
        const render = (runtime.render = state.table("global-transform-interpolated", Xform, {
            gpuOnly: true,
        }));
        current.bindComponent(GlobalTransform, { pos: "pos", quat: "quat", scale: "scale" });
        previous.bindMembership(GlobalTransform);
        render.bindMembership(GlobalTransform);
        render.subscribe((buffer) => {
            state.gpu.buffers.set("globalTransforms", buffer);
            state.gpu.typed.set("globalTransforms", render.typed);
        });
        render.enableEidLookup();
        render.subscribeMap((buffer) => {
            state.gpu.buffers.set("globalTransformRows", buffer);
            state.gpu.typed.set("globalTransformRows", render.eidToRowTyped!);
        });
        if (!runtime.params) {
            const params = state.gpu.root.createBuffer(d.vec4f).$usage("uniform");
            const buffer = state.gpu.root.unwrap(params);
            runtime.params = buffer;
            state.own(buffer);
        }
        runtime.pipeline = state.gpu.root.unwrap(
            state.gpu.root.createComputePipeline({ compute: kernel }),
        );
        for (const eid of state.query(globalTransformTerms)) queueFresh(runtime, eid);
    }
    return runtime.render!;
}

/** @internal Producer membership and derived-component lifetime belong to the engine. */
export function globalTransformProducerChanged(
    state: State,
    component: Component,
    eid: number,
    present: boolean,
): void {
    if (!state.registry.provides(component, GlobalTransform)) return;
    const runtime = state.globalTransformRuntime;
    if (!runtime) return;
    let producers = runtime.producers.get(eid);
    if (present) {
        if (!producers) runtime.producers.set(eid, (producers = new Set()));
        producers.add(idOf(component));
        runtime.pendingRemoval.delete(eid);
        if (!state.has(eid, GlobalTransform)) state.add(eid, GlobalTransform);
    } else {
        producers?.delete(idOf(component));
        if (producers?.size === 0) runtime.pendingRemoval.add(eid);
    }
}

/** @internal A producer cannot remove a derived row still owned by another producer. */
export function retainsGlobalTransform(state: State, eid: number, component: Component): boolean {
    if (component !== GlobalTransform) return false;
    return (state.globalTransformRuntime?.producers.get(eid)?.size ?? 0) > 0;
}

/** @internal Destruction clears owner state along with the entity's component membership. */
export function forgetGlobalTransformEntity(state: State, eid: number): void {
    const runtime = state.globalTransformRuntime;
    runtime?.producers.delete(eid);
    runtime?.pendingRemoval.delete(eid);
}

/** @internal Reconcile one-frame producer gaps before world placement is derived for draw. */
export function reconcileGlobalTransformProducers(state: State): void {
    const runtime = state.globalTransformRuntime;
    if (!runtime) return;
    for (const eid of runtime.pendingRemoval) {
        if (runtime.producers.get(eid)?.size) continue;
        runtime.pendingRemoval.delete(eid);
        runtime.producers.delete(eid);
        if (state.has(eid, GlobalTransform)) state.remove(eid, GlobalTransform);
    }
}

function queueFresh(runtime: GlobalTransformRuntime, eid: number): void {
    if (runtime.freshCount === runtime.fresh.length) {
        const fresh = new Uint32Array(runtime.fresh.length * 2);
        fresh.set(runtime.fresh);
        runtime.fresh = fresh;
    }
    runtime.fresh[runtime.freshCount++] = eid;
}

/** @internal Gather authored placement into the fixed world column without per-row callbacks. */
export function deriveTransforms(state: State): void {
    const runtime = state.globalTransformRuntime;
    if (!runtime) return;
    const global = runtime.global,
        source = runtime.placement;
    const pp = source.pos.column,
        pq = source.rot.column,
        ps = source.scale.column;
    const op = global.pos.column,
        oq = global.quat.column,
        os = global.scale.column;
    const pd = global.pos.dirty,
        qd = global.quat.dirty,
        sd = global.scale.dirty;
    const spd = source.pos.dirty,
        sqd = source.rot.dirty,
        ssd = source.scale.dirty;
    for (const eid of state.query(transformTerms)) {
        const word = eid >>> 5,
            mask = 1 << (eid & 31);
        if (((spd[word] | sqd[word] | ssd[word]) & mask) === 0) continue;
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

function captureCurrent(state: State, phase: number): void {
    const runtime = state.globalTransformRuntime!;
    const current = runtime.current!;
    current.prepareUpload();
    const size = current.pendingUploadSize;
    let buffer = runtime.stages[phase];
    if (size && (!buffer || buffer.size < size)) {
        if (buffer) state.retireGpuBuffer(buffer);
        buffer = state.gpu.device.createBuffer({
            label: `global-transform-tick-${phase}`,
            size,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        state.own(buffer);
        runtime.stages[phase] = buffer;
    }
    if (size && buffer) current.upload(buffer, true);
    runtime.ranges[phase * 2] = size ? current.lastUploadOffset : 0;
    runtime.ranges[phase * 2 + 1] = size;
    runtime.captureIndex = phase + 1;
}
/** @internal Initial placement precedes this frame's fixed ticks; no reader means no GPU work. */
export function beginGlobalTransformTick(state: State): void {
    deriveTransforms(state);
    const runtime = state.globalTransformRuntime;
    if (runtime?.enabled && runtime.tickCount === 0) captureCurrent(state, 0);
}
/** @internal Stage changed current rows; each catch-up tick has distinct immutable GPU bytes. */
export function endGlobalTransformTick(state: State): void {
    deriveTransforms(state);
    const runtime = state.globalTransformRuntime;
    if (runtime?.enabled) captureCurrent(state, ++runtime.tickCount);
}
/** @internal Gather post-simulation placement; recording waits for the renderer's frame encoder. */
export function prepareGlobalTransform(state: State): void {
    reconcileGlobalTransformProducers(state);
    deriveTransforms(state);
    const runtime = state.globalTransformRuntime;
    if (!runtime?.enabled) return;
    captureCurrent(state, runtime.tickCount ? runtime.tickCount + 1 : 0);
    runtime.render!.upload();
}

function copyPhase(state: State, encoder: GPUCommandEncoder, phase: number): void {
    const runtime = state.globalTransformRuntime!;
    const offset = runtime.ranges[phase * 2],
        size = runtime.ranges[phase * 2 + 1];
    if (size && !runtime.stages[phase]) {
        throw new Error(
            `GlobalTransform history phase ${phase} has ${size} bytes but no staging buffer`,
        );
    }
    if (size) {
        encoder.copyBufferToBuffer(
            runtime.stages[phase]!,
            0,
            runtime.current!.buffer,
            offset,
            size,
        );
        runtime.historyNeedsPromotion = true;
    }
    for (let i = 0; i < runtime.discontinuityCount; i++) {
        if (runtime.discontinuityPhases[i] !== phase) continue;
        const row = runtime.current!.rowIndex(runtime.discontinuities[i]);
        if (row >= 0)
            encoder.copyBufferToBuffer(
                runtime.current!.buffer,
                row * 48,
                runtime.previous!.buffer,
                row * 48,
                48,
            );
    }
}
/** @internal Record history copies and bind data before renderer compute passes. */
export function prepareGlobalTransformFrame(state: State, encoder: GPUCommandEncoder): void {
    const runtime = state.globalTransformRuntime;
    if (!runtime?.enabled) return;
    copyPhase(state, encoder, 0);
    for (let tick = 1; tick <= runtime.tickCount; tick++) {
        if (runtime.historyNeedsPromotion) {
            encoder.copyBufferToBuffer(
                runtime.current!.buffer,
                0,
                runtime.previous!.buffer,
                0,
                runtime.current!.buffer.size,
            );
            runtime.historyNeedsPromotion = false;
        }
        copyPhase(state, encoder, tick);
    }
    if (runtime.tickCount) copyPhase(state, encoder, runtime.tickCount + 1);
    for (let i = 0; i < runtime.freshCount; i++) {
        const row = runtime.current!.rowIndex(runtime.fresh[i]);
        if (row >= 0)
            encoder.copyBufferToBuffer(
                runtime.current!.buffer,
                row * 48,
                runtime.previous!.buffer,
                row * 48,
                48,
            );
    }
    runtime.tickCount = 0;
    runtime.captureIndex = 0;
    runtime.freshCount = 0;
    runtime.discontinuityCount = 0;
    runtime.ranges.fill(0);
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
        runtime.group = state.gpu.device.createBindGroup({
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

/** Compose fixed-tick GlobalTransform for CPU camera and query readers. */
export function composeTransform(eid: number, out: Float32Array): Float32Array {
    const { pos, quat: rot, scale } = GlobalTransform;
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
